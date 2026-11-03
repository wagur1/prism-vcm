"""Offline 64x64 oracle tests: no downloads, model weights, or codec binaries."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prism.codec.x265_bridge import qg_shape
from prism.data.splits import SplitEntry, SplitLedger
from prism.training.manifest import EncodeManifest
from prism.training.oracle_sweep import OracleConfig, OracleSweep, rank_task_candidates
from tier1a.common import read_json
from tier1a.oracle_candidates import cached_candidates, candidates_hash, generate_candidates, project_crop
from tier1a.oracle_codec import ClaimConfig, yuv_to_rgb
from tier1a.oracle_detector import source_box
from tier1a.run_oracle_sweep import FrameSweep, choose_scope, load_frame, projection
from tier1a.yuv import letterbox_yuv, save_yuv
from tier1a.teachers import save_npz


class StubEncoder:
    def __init__(self, fail_index=None, decode_fail=False, stock_mismatch=False):
        self.calls = []
        self.fail_index, self.decode_fail, self.stock_mismatch = fail_index, decode_fail, stock_mismatch

    def __call__(self, payload, offsets, frame, directory, *, stock=False):
        self.calls.append((offsets, stock))
        index = sum(not s for _, s in self.calls) - 1
        if not stock and index == self.fail_index:
            raise RuntimeError("deliberate stub encode failure")
        size = 100 if stock else 100 + index * 10
        return {"whole_bitstream_bytes": size, "bits": 8 * size, "frame_bits": 8 * size - 16,
                "encode_seconds": .01, "bitstream_sha256": "b" * 64 if stock and self.stock_mismatch else "a" * 64}

    def decode(self, result, frame):
        if self.decode_fail:
            raise RuntimeError("deliberate decoder failure")
        return np.full((64, 64, 3), result["whole_bitstream_bytes"], dtype=np.uint8)


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entries = [SplitEntry("1", "a" * 64, "train"), SplitEntry("2", "b" * 64, "train"),
                        SplitEntry("3", "c" * 64, "dev")]
        draft = SplitLedger(self.entries)
        self.ledger = SplitLedger(self.entries, draft.digest)
        self.manifest = EncodeManifest(self.root / "encodes.jsonl", repo_sha="stub", config_sha="stub", build={}, seed=0)
        self.sweep = FrameSweep(self.root, self.ledger, self.manifest, {"test": True})
        rgb = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        self.payload, self.geometry = letterbox_yuv(rgb, 64, 64)
        self.frame = {"sample_id": "1", "image_id": 1, "source_sha256": "a" * 64, "geometry": self.geometry}
        self.sam = np.zeros((4, 4), dtype=bool)
        self.sam[:2, :2] = True
        self.saliency = np.arange(16).reshape(4, 4)
        self.maps = generate_candidates(64, 64, self.sam, self.saliency, sample_id="1")

    def test_determinism_independent_of_global_rng(self):
        np.random.seed(987)
        again = generate_candidates(64, 64, self.sam, self.saliency, sample_id="1", seed=0)
        np.testing.assert_array_equal(self.maps, again)
        self.assertEqual(candidates_hash(self.maps), candidates_hash(again))

    def test_seed_and_identity_change_only_random_proposal(self):
        different = generate_candidates(64, 64, self.sam, self.saliency, sample_id="2", seed=9)
        np.testing.assert_array_equal(self.maps[:6], different[:6])
        np.testing.assert_array_equal(self.maps[7], different[7])
        self.assertFalse(np.array_equal(self.maps[6], different[6]))

    def test_all_designs_integer_bounds_and_1080_grid(self):
        self.assertEqual(self.maps.shape, (8, 4, 4))
        self.assertTrue(np.issubdtype(self.maps.dtype, np.integer))
        self.assertLessEqual(np.abs(self.maps).max(), 6)
        np.testing.assert_array_equal(self.maps[0], 0)
        np.testing.assert_array_equal(self.maps[1], 3)
        np.testing.assert_array_equal(self.maps[2], -3)
        np.testing.assert_array_equal(self.maps[3], np.where(self.sam, -3, 2))
        np.testing.assert_array_equal(self.maps[7], np.where(self.sam, -4, 1))
        self.assertEqual(int((self.maps[4] == -3).sum()), 4)
        self.assertEqual(qg_shape(1920, 1080), (68, 120))
        full = generate_candidates(1920, 1080, np.zeros((68, 120)), np.zeros((68, 120)), sample_id="x")
        self.assertEqual(full.shape, (8, 68, 120))
        self.assertLessEqual(np.abs(full).max(), 6)

    def test_fusion_k9_is_encoded_named_and_ranked(self):
        maps = generate_candidates(64, 64, self.sam, self.saliency,
                                   sample_id="1", include_fusion=True)
        encoder = StubEncoder()
        row, skipped = self.sweep.process(self.frame, self.payload, maps, encoder,
                                          lambda rgb, frame: .5, smoke=True)
        self.assertFalse(skipped)
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(row["candidate_count"], 9)
        self.assertEqual(row["candidates"][8]["name"], "sam_dino_fusion_minus4_plus6")
        self.assertTrue(row["candidates"][8]["decode_ok"])
        self.assertIsNotNone(row["candidates"][8]["J"])
        self.assertEqual(len(encoder.calls), 10)  # nine proposals plus stock identity

    def test_unknown_crop_offsets_zero(self):
        known = np.zeros((4, 4), dtype=bool)
        known[1:3, 1:3] = True
        maps = generate_candidates(64, 64, self.sam, self.saliency, sample_id="1", observed=known)
        for i in (3, 4, 6, 7):
            self.assertTrue((maps[i][~known] == 0).all())

    def test_crop_projection_and_inverse_boxes(self):
        geometry = dict(self.geometry, source_width=128, source_height=64, active_width=64,
                        active_height=32, left=0, top=16, scale_x=.5, scale_y=.5)
        grid, known = project_crop(np.ones((4, 4)), np.array([32, 0, 96, 64]), geometry)
        self.assertEqual(int(known.sum()), 4)
        self.assertEqual(int(grid.sum()), 4)
        self.assertEqual(source_box([16, 16, 48, 48], geometry), [32., 0., 64., 64.])

    def test_invalid_teacher_and_candidate_inputs(self):
        with self.assertRaises(ValueError):
            generate_candidates(64, 64, self.sam, self.saliency * np.nan, sample_id="1")
        with self.assertRaises(ValueError):
            candidates_hash(np.full((8, 4, 4), 7))

    def test_j_math_direction_runner_up_and_ties(self):
        j, order = rank_task_candidates([.5, .52, .49], [800, 1600, 400], .05)
        np.testing.assert_allclose(j, [.55, .545, .59])
        self.assertEqual(order, [2, 0, 1])
        self.assertEqual(rank_task_candidates([.5, .5], [100, 100])[1], [0, 1])
        for scores, bits, rate in [([np.nan], [1], .05), ([.5], [0], .05), ([1.1], [1], .05),
                                    ([.5], [1], -1), ([.5], [1], np.inf)]:
            with self.assertRaises(ValueError):
                rank_task_candidates(scores, bits, rate)

    def test_real_frame_flow_stub_encoder_png_and_runner_up(self):
        encoder = StubEncoder()
        scored = []
        def scorer(rgb, frame):
            scored.append(int(rgb[0, 0, 0]))
            return float(rgb[0, 0, 0]) / 255
        row, skipped = self.sweep.process(self.frame, self.payload, self.maps, encoder, scorer, smoke=True)
        self.assertFalse(skipped)
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(len(scored), 8)
        self.assertEqual(row["winner_map_index"], 7)
        self.assertEqual(row["runner_up_map_index"], 6)
        self.assertEqual(self.manifest.counts()["oracle_initial"], 8)
        self.assertEqual(self.manifest.counts()["gate0"], 1)
        self.assertTrue(Path(row["winner_png"]).exists())
        self.assertEqual(len(list(self.root.rglob("*.png"))), 1)
        self.assertFalse(list(self.root.rglob("*.yuv")))
        self.assertTrue(all(c["decode_ok"] for c in row["candidates"]))
        self.assertTrue(all(c["frame_bits"] != c["bits"] for c in row["candidates"]))

    def test_idempotency_across_process_objects_and_candidate_hash_refresh(self):
        encoder = StubEncoder()
        self.sweep.process(self.frame, self.payload, self.maps, encoder, lambda *_: .5, smoke=True)
        resumed = FrameSweep(self.root, self.ledger, self.manifest, {"test": True})
        row, skipped = resumed.process(self.frame, self.payload, self.maps, encoder, lambda *_: .5, smoke=True)
        self.assertTrue(skipped)
        self.assertEqual(len(encoder.calls), 9)
        changed = self.maps.copy()
        changed[6, 0, 0] = 0
        resumed.process(self.frame, self.payload, changed, encoder, lambda *_: .5)
        self.assertEqual(self.manifest.counts()["oracle_refresh"], 8)
        self.assertEqual(len(resumed.data["history"]), 1)

    def test_contract_change_refused(self):
        with self.assertRaises(ValueError):
            FrameSweep(self.root, self.ledger, self.manifest, {"test": True}, lambda_rate=.06)

    def test_encode_failure_recorded_all_other_candidates_attempted_then_next_frame(self):
        encoder = StubEncoder(fail_index=2)
        row, _ = self.sweep.process(self.frame, self.payload, self.maps, encoder, lambda *_: .5)
        self.assertEqual(row["status"], "FAIL")
        self.assertIsNone(row["winner_map_index"])
        self.assertEqual(len(encoder.calls), 8)
        self.assertIsNone(row["candidates"][2]["D"])
        self.assertIn("deliberate", row["candidates"][2]["reason"])
        row2, _ = self.sweep.process(dict(self.frame, sample_id="2", image_id=2, source_sha256="b"*64),
                                    self.payload, self.maps, encoder, lambda *_: .5)
        self.assertEqual(row2["status"], "PASS")
        _, skipped = self.sweep.process(self.frame, self.payload, self.maps, encoder, lambda *_: .5)
        self.assertTrue(skipped)
        self.assertEqual(len(encoder.calls), 16)
        self.assertEqual(read_json(self.sweep.path)["labels"]["1"]["status"], "FAIL")

    def test_decode_and_nan_failures_never_fabricate_score(self):
        row, _ = self.sweep.process(self.frame, self.payload, self.maps, StubEncoder(decode_fail=True), lambda *_: .9)
        self.assertFalse(row["candidates"][0]["decode_ok"])
        self.assertIsNone(row["candidates"][0]["D"])
        second = dict(self.frame, sample_id="2", image_id=2, source_sha256="b"*64)
        row, _ = self.sweep.process(second, self.payload, self.maps, StubEncoder(), lambda *_: float("nan"))
        self.assertEqual(row["status"], "FAIL")
        self.assertIsNone(read_json(self.sweep.path)["labels"]["2"]["candidates"][0]["D"])

    def test_interrupted_frame_is_failed_without_retry(self):
        row, _ = self.sweep.process(self.frame, self.payload, self.maps, StubEncoder(), lambda *_: .5)
        row["status"] = "RUNNING"
        row["candidates"][3]["status"] = "RUNNING"
        self.sweep.flush()
        encoder = StubEncoder()
        row, skipped = self.sweep.process(self.frame, self.payload, self.maps, encoder, lambda *_: .5)
        self.assertTrue(skipped)
        self.assertEqual(row["status"], "FAIL")
        self.assertFalse(encoder.calls)

    def test_missing_winner_is_failure_without_reencode(self):
        row, _ = self.sweep.process(self.frame, self.payload, self.maps, StubEncoder(), lambda *_: .5)
        Path(row["winner_png"]).unlink()
        encoder = StubEncoder()
        row, skipped = self.sweep.process(self.frame, self.payload, self.maps, encoder, lambda *_: .5)
        self.assertTrue(skipped)
        self.assertEqual(row["status"], "FAIL")
        self.assertFalse(encoder.calls)

    def test_train_only_before_encode(self):
        encoder = StubEncoder()
        with self.assertRaises(ValueError):
            self.sweep.process(dict(self.frame, sample_id="3", image_id=3, source_sha256="c"*64),
                               self.payload, self.maps, encoder, lambda *_: .5)
        self.assertFalse(encoder.calls)

    def test_stock_mismatch_fails_zero_and_frame(self):
        row, _ = self.sweep.process(self.frame, self.payload, self.maps, StubEncoder(stock_mismatch=True),
                                    lambda *_: .5, smoke=True)
        self.assertEqual(row["status"], "FAIL")
        self.assertEqual(self.sweep.data["zero_stock_smoke"]["status"], "FAIL")

    def test_projection_includes_scoring_and_failure_is_not_timing_evidence(self):
        row = {"status": "PASS", "wall_seconds": 80, "encode_seconds": 8, "candidate_count": 8,
               "candidates": [{"encode_seconds": 1}] * 8}
        result = projection([row] * 10)
        self.assertAlmostEqual(result["projected_500x8_encode_hours"], 4000 / 3600)
        self.assertAlmostEqual(result["projected_500x8_wall_hours"], 50000 / 3600)
        self.assertFalse(result["config_f_200_image_measurement"])
        self.assertTrue(projection([row] * 200)["config_f_200_image_measurement"])
        with self.assertRaises(ValueError):
            projection([dict(row, status="FAIL")])

    def test_claim_config_and_memory_yuv_decode(self):
        options = ClaimConfig().options(self.root / "frame.csv")
        for value in ("crf=20.0", "aq-mode=1", "aq-strength=1.0", "hevc-aq=0", "cutree=0",
                      "keyint=1", "bframes=0", "qg-size=16", "frame-threads=1", "wpp=0"):
            self.assertIn(value, options.splitlines())
        black = bytes([16]) * 4096 + bytes([128]) * 2048
        self.assertFalse(yuv_to_rgb(black, 64, 64).any())
        with self.assertRaises(ValueError):
            yuv_to_rgb(black[:-1], 64, 64)

    def test_scope_reduction_and_setup_allowance(self):
        self.assertEqual(choose_scope({"projected_500x8_wall_hours": 5}, 60, 6), 8)
        self.assertEqual(choose_scope({"projected_500x8_wall_hours": 6.1}, 0, 6), 6)
        self.assertEqual(choose_scope({"projected_500x8_wall_hours": 5.9}, 600, 6), 6)

    def test_original_distortion_oracle_keeps_minimization_contract(self):
        original = OracleSweep(OracleConfig("squared error", .01, 1000), self.ledger, self.manifest)
        maps = [np.full((4, 4), i) for i in range(-4, 4)]
        row = original.label("1", "d"*64, maps, lambda m: (float((m*m).sum()), 100), lambda _: True)
        self.assertFalse(row.delta_qp.any())
        self.assertAlmostEqual(row.objective, .001)

    def test_cached_proposals_verify_source_binding_and_npz_hash(self):
        teacher_root = self.root / "teachers"
        binding = {"source_sha256": "a"*64, "ledger_sha256": self.ledger.digest, "selection_sha256": "s"*64}
        crop = np.array([0, 0, 64, 64])
        dino = {"crop_box": crop, "cls_token": np.ones(768, dtype=np.float16),
                "patch_tokens": np.ones((1369, 768), dtype=np.float16)}
        sam = {"crop_box": crop, "masks": np.zeros((0, 518, 518), dtype=bool)}
        for name, arrays in (("dinov2", dino), ("sam21", sam)):
            path = teacher_root / name / "1.npz"
            path.parent.mkdir(parents=True)
            save_npz(path, arrays, binding)
        def load(source="a"*64):
            return cached_candidates(teacher_root, "1", self.geometry, source_sha=source,
                                     ledger_sha=self.ledger.digest, selection_sha="s"*64)
        maps, provenance = load()
        self.assertEqual(maps.shape, (8, 4, 4))
        self.assertEqual(provenance["observed_qgs"], 16)
        with self.assertRaises(ValueError):
            load("b"*64)
        with (teacher_root / "sam21/1.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            load()

    def test_load_frame_supports_gzip_and_kaggle_auto_gunzipped_yuv(self):
        from tier1a.common import sha256, write_json
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        payload, geometry = letterbox_yuv(rgb, 1920, 1080)
        gz_path = self.root / "coco500/yuv/1.yuv.gz"
        save_yuv(gz_path, payload)
        meta = {"image_id": 1, "sample_id": "1", "source_sha256": "a" * 64,
                "sha256": sha256(gz_path), "geometry": geometry}
        write_json(gz_path.with_name("1.yuv.gz.json"), meta)
        row = {"image_id": 1, "sample_id": "1", "source_sha256": "a" * 64, "yuv_metadata": meta}
        binding = {"source_sha256": "a" * 64, "ledger_sha256": self.ledger.digest, "selection_sha256": "s" * 64}
        crop = np.array([0, 0, 32, 32])
        for name, arrays in (("dinov2", {"crop_box": crop, "cls_token": np.ones(768, dtype=np.float16),
                                         "patch_tokens": np.ones((1369, 768), dtype=np.float16)}),
                             ("sam21", {"crop_box": crop, "masks": np.zeros((0, 518, 518), dtype=bool)})):
            save_npz(self.root / "teachers" / name / "1.npz", arrays, binding)
        frame1, loaded1, maps1 = load_frame(self.root, row, self.ledger, "s" * 64)
        self.assertEqual(loaded1, payload)
        # Simulate Kaggle Dataset mount auto-gunzipping 1.yuv.gz -> 1.yuv
        raw_path = gz_path.with_suffix("")
        raw_path.write_bytes(payload)
        gz_path.unlink()
        frame2, loaded2, maps2 = load_frame(self.root, row, self.ledger, "s" * 64)
        self.assertEqual(loaded2, payload)
        np.testing.assert_array_equal(maps1, maps2)
        self.assertEqual(frame1["geometry"], frame2["geometry"])
        # Tampered raw YUV must still be rejected by raw_sha256 check
        bad = bytearray(payload)
        bad[0] ^= 1
        raw_path.write_bytes(bytes(bad))
        with self.assertRaisesRegex(ValueError, "Raw input SHA mismatch"):
            load_frame(self.root, row, self.ledger, "s" * 64)


if __name__ == "__main__":
    unittest.main()
