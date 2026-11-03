"""CPU contracts for adaptive controls, foreground protection and paired screening.

Runnable with unittest + NumPy; codec and detector integration is explicit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from prism.codec.qp_maps import project_adaptive_qp_map, project_zero_mean_qp_map, qg_area_weights
from prism.preprocessing.background_filter import apply_background_smoothing
from scripts.screen_m2_enhancements import (
    MockEncoder, parse_args, run_screen, saved_map, synthetic_frame, variants,
)
from tier1a.oracle_candidates import RECIPE, NAMES, candidates_hash, generate_candidates


class AdaptiveQPTests(unittest.TestCase):
    def test_compensates_jensen_penalty_and_crf_schedule(self):
        raw = np.tile([-6., -2., 2., 6.], (4, 1))
        base = project_zero_mean_qp_map(raw, quantize_int8=False)
        self.assertGreater(float(np.exp2(-base / 6).mean()), 1.0)
        neutral = project_adaptive_qp_map(raw, delta_rate_offset=0)
        self.assertAlmostEqual(float(np.exp2(-neutral / 6).mean()), 1.0, places=6)
        low = project_adaptive_qp_map(raw, crf=20)
        high = project_adaptive_qp_map(raw, crf=36)
        self.assertTrue(np.all(low >= high))
        self.assertGreater(float(low.mean()), float(high.mean()))
        self.assertLess(float(np.exp2(-low / 6).mean()), 1.0)

    def test_offset_adapts_to_content_and_uniform_has_no_evidence(self):
        mild = project_adaptive_qp_map(np.tile([-1., 1.], (4, 2)), delta_rate_offset=0)
        strong = project_adaptive_qp_map(np.tile([-4., 4.], (4, 2)), delta_rate_offset=0)
        self.assertGreater(float(strong.mean()), float(mild.mean()))
        np.testing.assert_array_equal(project_adaptive_qp_map(np.ones((4, 4))), 0)

    def test_qg16_partial_edges_and_output_contract(self):
        for width, height in ((1920, 1080), (66, 66)):
            w = qg_area_weights(width, height)
            self.assertAlmostEqual(float(w.sum()) * 256, width * height)
            raw = np.random.default_rng(5).normal(0, 6, w.shape)
            for quantize in (False, True):
                q = project_adaptive_qp_map(raw, weights=w, high=8, quantize_int8=quantize)
                self.assertEqual(q.shape, w.shape)
                self.assertEqual(q.dtype, np.int8 if quantize else np.float32)
                self.assertTrue(q.flags.c_contiguous)
                self.assertTrue(np.all((q >= -4) & (q <= 8)))
                np.testing.assert_array_equal(q[w == 0], 0)
        self.assertEqual(qg_area_weights().shape, (68, 120))

    def test_mask_and_empty_support(self):
        known = np.eye(4, dtype=bool)
        q = project_adaptive_qp_map(np.arange(16).reshape(4, 4), mask=known)
        np.testing.assert_array_equal(q[~known], 0)
        np.testing.assert_array_equal(project_adaptive_qp_map(q, weights=np.zeros((4, 4))), 0)

    def test_rejects_invalid_maps_and_parameters(self):
        q = np.zeros((4, 4))
        cases = [dict(crf=np.nan), dict(crf=52), dict(weights=np.ones((4, 1))),
                 dict(weights=-np.ones((4, 4))), dict(mask=np.ones((1, 4))),
                 dict(low=1), dict(high=np.inf), dict(delta_rate_offset=-1),
                 dict(high=128, quantize_int8=True)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                project_adaptive_qp_map(q, **kwargs)
        with self.assertRaises(ValueError):
            project_adaptive_qp_map(np.full((4, 4), np.nan))


class BackgroundFilterTests(unittest.TestCase):
    def test_foreground_exact_background_smoother_and_no_mutation(self):
        rgb = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        original = rgb.copy()
        q = np.full((4, 4), 6.)
        q[:, :2] = -4
        out = apply_background_smoothing(rgb, q)
        np.testing.assert_array_equal(out[:, :32], rgb[:, :32])
        np.testing.assert_array_equal(rgb, original)
        self.assertLess(float(out[:, 40:].var()), float(rgb[:, 40:].var()))
        self.assertEqual(out.dtype, rgb.dtype)
        np.testing.assert_array_equal(apply_background_smoothing(rgb, q, strength=0), rgb)

    def test_yuv420_partial_padding_and_saliency(self):
        rng = np.random.default_rng(4)
        yuv = (rng.integers(16, 236, (66, 66), dtype=np.uint8),
               rng.integers(16, 241, (33, 33), dtype=np.uint8),
               rng.integers(16, 241, (33, 33), dtype=np.uint8))
        q = np.full((8, 8), 6.)
        q[4, 4] = -4  # Partial 2x2 luma QG must stay at pixel (64,64), not be rescaled.
        out = apply_background_smoothing(yuv, q)
        self.assertIsInstance(out, tuple)
        np.testing.assert_array_equal(out[0][64:, 64:], yuv[0][64:, 64:])
        self.assertEqual(out[1][32, 32], yuv[1][32, 32])
        self.assertEqual(out[2][32, 32], yuv[2][32, 32])
        saliency = np.zeros((8, 8))
        saliency[4, 4] = 1
        np.testing.assert_array_equal(apply_background_smoothing(yuv[0], saliency,
            mask_kind="saliency"), out[0])

    def test_float_preservation_and_validation(self):
        x = np.random.default_rng(3).random((64, 64, 3))
        q = np.full((4, 4), 3.)
        q[0, 0] = 0
        out = apply_background_smoothing(x, q)
        self.assertEqual(out.dtype, x.dtype)
        self.assertGreaterEqual(out.min(), x.min())
        self.assertLessEqual(out.max(), x.max())
        np.testing.assert_array_equal(out[:16, :16], x[:16, :16])
        constant = np.full((64, 64), .1, np.float64)
        np.testing.assert_array_equal(apply_background_smoothing(constant, q), constant)
        for kwargs in (dict(strength=-1), dict(sigma=0), dict(mask_kind="auto")):
            with self.assertRaises(ValueError):
                apply_background_smoothing(x, q, **kwargs)
        with self.assertRaises(ValueError):
            apply_background_smoothing(x, np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            apply_background_smoothing((x[:, :, 0], x[:, :, 0], x[:, :, 0]), q)


class FusionTests(unittest.TestCase):
    def test_joint_intersection_unknown_and_legacy_hash(self):
        sam = np.zeros((4, 4), bool)
        sam[0, :] = True
        sam[3, 3] = True
        saliency = np.arange(16, 0, -1).reshape(4, 4)
        known = np.ones((4, 4), bool)
        known[1, 0] = False
        old = generate_candidates(64, 64, sam, saliency, sample_id="test", observed=known)
        new = generate_candidates(64, 64, sam, saliency, sample_id="test", observed=known,
                                  include_fusion=True)
        self.assertEqual(new.shape, (9, 4, 4))
        self.assertEqual(NAMES[-1], "sam_dino_fusion_minus4_plus6")
        np.testing.assert_array_equal(old, new[:8])
        np.testing.assert_array_equal(new[8, 0], -4)
        self.assertEqual(new[8, 3, 3], 0)
        self.assertEqual(new[8, 1, 0], 0)
        self.assertEqual(new[8, 2, 2], 6)
        h = hashlib.sha256(json.dumps(RECIPE, sort_keys=True).encode())
        h.update(str(old.shape).encode())
        h.update(old.tobytes())
        self.assertEqual(candidates_hash(old), h.hexdigest())
        self.assertNotEqual(candidates_hash(old), candidates_hash(new))


class ScreeningTests(unittest.TestCase):
    def test_variants_share_map_and_preserve_input(self):
        frame, payload, raw = synthetic_frame(0, 64, 64, 0)
        ablations = variants(payload, raw, 64, 64, 20)
        self.assertEqual(ablations[0][1], ablations[1][1])
        self.assertEqual(ablations[0][1], ablations[2][1])
        self.assertIs(ablations[2][2], ablations[3][2])
        self.assertNotEqual(ablations[2][1], ablations[3][1])
        self.assertIsNone(ablations[0][2])

    def test_mock_report_balanced_accounting_and_no_map_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(["--mock", "--limit", "2", "--width", "66", "--height", "66",
                               "--crfs", "20", "36", "--output", tmp])
            report = run_screen(args)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["physical_encodes"], 18)
            self.assertFalse(report["rate_is_hevc"])
            self.assertFalse(report["bd_rate_claimed"])
            self.assertEqual(len(report["summary"]), 8)
            for row in report["summary"]:
                self.assertIsNone(row["map_50_95"])
                self.assertEqual(row["images"], 2)
                self.assertGreater(row["total_bytes"], 0)
                if row["variant"] == "M1":
                    self.assertEqual(row["bitrate_delta_pct_vs_M1"], 0)
            with self.assertRaises(FileExistsError):
                run_screen(args)

    def test_saved_map_requires_exact_input_binding_and_crf(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = {"sample_id": "1", "input_yuv_sha256": "abc"}
            np.savez(Path(tmp) / "1.npz", raw_qg_map=np.ones((2, 4, 4)), crfs=[20, 36],
                     input_yuv_sha256="abc")
            raw, known = saved_map(Path(tmp), frame, 20)
            self.assertEqual(raw.shape, (4, 4))
            self.assertIsNone(known)
            with self.assertRaises(ValueError):
                saved_map(Path(tmp), frame, 28)
            with self.assertRaises(ValueError):
                saved_map(Path(tmp), dict(frame, input_yuv_sha256="wrong"), 20)

    def test_identity_failure_is_persisted(self):
        class BrokenEncoder(MockEncoder):
            def __call__(self, *args, **kwargs):
                result = super().__call__(*args, **kwargs)
                if kwargs.get("stock"):
                    result["bitstream_sha256"] = "mismatch"
                return result
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(["--mock", "--limit", "1", "--width", "64", "--height", "64", "--output", tmp])
            with patch("scripts.screen_m2_enhancements.MockEncoder", BrokenEncoder), self.assertRaisesRegex(ValueError, "identity"):
                run_screen(args)
            report = json.loads((Path(tmp) / "report.json").read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["summary"])


if __name__ == "__main__":
    unittest.main()
