#!/usr/bin/env python3
"""Unit tests for MOT17 harvester and merge evaluation pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.merge_mot17_results import (
    SEQUENCES, VARIANTS, CRFS,
    aggregate_observations, compute_sequence_bd_rates,
    compute_overall_bd_rates, compute_bootstrap_ci,
    generate_latex_tables, merge_and_report, validate_report
)
from scripts.auto_harvester_mot17 import parse_kernel_status


def make_mock_sequence_report(seq_name: str, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    observations = []

    for v_idx, v in enumerate(VARIANTS):
        # Anchor base bitrate and scores
        base_rate = {22: 1200.0, 27: 750.0, 32: 450.0, 37: 250.0}
        base_hota = {22: 60.0, 27: 56.0, 32: 50.0, 37: 42.0}
        base_mota = {22: 68.0, 27: 62.0, 32: 54.0, 37: 44.0}
        base_idf1 = {22: 72.0, 27: 67.0, 32: 59.0, 37: 49.0}

        rate_factor = 1.0 - (0.02 * v_idx)  # candidates save bitrate
        score_offset = 0.2 * v_idx

        for c in CRFS:
            kbps = base_rate[c] * rate_factor + rng.normal(0, 5.0)
            hota = base_hota[c] + score_offset + rng.normal(0, 0.2)
            mota = base_mota[c] + score_offset + rng.normal(0, 0.3)
            idf1 = base_idf1[c] + score_offset + rng.normal(0, 0.3)

            observations.append({
                "variant": v,
                "crf": c,
                "num_frames": 100,
                "total_bytes": int(kbps * 1000 * (100 / 30.0) / 8),
                "bits": int(kbps * 1000 * (100 / 30.0)),
                "bitrate_kbps": float(kbps),
                "metrics": {
                    "HOTA": float(hota),
                    "MOTA": float(mota),
                    "IDF1": float(idf1),
                    "DetA": float(hota - 2.0),
                    "AssA": float(hota + 2.0),
                }
            })

    return {
        "schema_version": 1,
        "status": "complete",
        "sequence": seq_name,
        "fps": 30.0,
        "seq_length": 100,
        "num_frames": 100,
        "duration_seconds": 100 / 30.0,
        "width": 1920,
        "height": 1080,
        "crfs": list(CRFS),
        "variants": list(VARIANTS),
        "observations": observations,
    }


class TestMergeMOT17(unittest.TestCase):
    def test_parse_kernel_status(self):
        self.assertEqual(parse_kernel_status("KernelWorkerStatus.RUNNING"), "RUNNING")
        self.assertEqual(parse_kernel_status('foo has status "KernelWorkerStatus.COMPLETE"'), "COMPLETE")
        self.assertEqual(parse_kernel_status("foo has status 'FAILED'"), "ERROR")
        self.assertEqual(parse_kernel_status("something unknown"), "UNKNOWN")

    def test_validate_report(self):
        rep = make_mock_sequence_report(SEQUENCES[0])
        validate_report(rep, SEQUENCES[0])
        with self.assertRaises(ValueError):
            validate_report(rep, "WRONG_SEQ")

        rep_bad = dict(rep, status="running")
        with self.assertRaises(ValueError):
            validate_report(rep_bad, SEQUENCES[0])

    def test_full_pipeline_mock(self):
        with tempfile.TemporaryDirectory() as td:
            reports_dir = Path(td) / "reports"
            merged_dir = Path(td) / "merged"
            reports_dir.mkdir(parents=True)

            for idx, seq in enumerate(SEQUENCES):
                seq_dir = reports_dir / seq
                seq_dir.mkdir()
                rep = make_mock_sequence_report(seq, seed=idx)
                (seq_dir / "mot17_seq_report.json").write_text(json.dumps(rep), encoding="utf-8")

            final_report_path = merge_and_report(reports_dir, merged_dir, resamples=200)
            self.assertTrue(final_report_path.is_file())

            with open(final_report_path, encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["num_sequences"], 7)
            self.assertIn("overall_bd_rates", data)
            self.assertIn("bootstrap_ci95", data)

            # Check that table tex files exist
            self.assertTrue((merged_dir / "table_mot17_ablation.tex").is_file())
            self.assertTrue((merged_dir / "table_mot17_per_sequence.tex").is_file())


if __name__ == "__main__":
    unittest.main()
