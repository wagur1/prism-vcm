"""Dependency-free numerical and evidence-integrity tests for telemetry proposals."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gate0.calibrate_bands import CONDITIONS, calibrate, percentile, read_floors

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = ROOT / "gate0/criteria_frozen.json"
TELEMETRY_FIELDS = ("record_type", "frame", "poc", "encode_order", "slice_type", "base_qp_rc",
                    "qp_aq_offset_applied", "weighted_satd_or_blurred_complexity", "qCompress",
                    "rc_mode", "x", "y", "requested_offset", "final_qp", "clipping_flag",
                    "base_qp_at_cu", "has_residual", "skip")


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "proposal.json"
        self.criteria_sha = hashlib.sha256(CRITERIA.read_bytes()).hexdigest()
        self.fixture()

    def fixture(self, effect=6):
        for gop in (1, 10):
            for condition in CONDITIONS:
                prefix = self.root / f"qp32_gop{gop}_{condition}"
                job = {"map": condition, "config": {"qp": 32, "gop": gop, "rc_mode": "CRF", "crf": 20}}
                prefix.with_suffix(".job.json").write_text(json.dumps(job))
                rows, trace = [], []
                drift = {"zero": 0, "plus6": 1, "minus6": -2, "checkerboard": 1}[condition]
                base = 30 + drift
                for poc, frame in enumerate((100, 110)):
                    identity = {"frame": frame, "poc": poc, "encode_order": poc,
                                "slice_type": "I" if gop == 1 or poc == 0 else "P"}
                    rows.append({"record_type": "frame", **identity, "base_qp_rc": base,
                                 "weighted_satd_or_blurred_complexity": 1234, "qCompress": .6, "rc_mode": "CRF"})
                    for x in (0, 16, 32, 48):
                        requested = {"zero": 0, "plus6": 6, "minus6": -6,
                                     "checkerboard": 6 if x < 32 else -6}[condition]
                        local = base + 2 + (effect if requested > 0 else -effect if requested < 0 else 0)
                        cu = {"record_type": "cu", **identity, "x": x, "y": 0,
                              "requested_offset": requested, "final_qp": local,
                              "base_qp_at_cu": base, "has_residual": 1, "skip": 0,
                              "clipping_flag": int(x == 48)}
                        rows.append(cu)
                        trace.append({"frame": frame, "x": x, "y": 0, "qp": local,
                                      "has_residual": 1, "skip": 0})
                self.write_csv(prefix.with_suffix(".telemetry.csv"), TELEMETRY_FIELDS, rows)
                self.write_csv(prefix.with_suffix(".trace.csv"), trace[0].keys(), trace)

    def write_csv(self, path, fields, rows):
        with path.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)

    def run_calibration(self, criteria=CRITERIA):
        return calibrate(self.root, criteria, self.output, qp_groups=(32,), expected_frames=2)

    def test_net_effect_cancels_base_drift_and_keeps_pts_separate_from_poc(self):
        proposal = self.run_calibration()
        for group in proposal["groups"]:
            self.assertEqual(group["median"], 6)
            self.assertEqual(group["p10"], 6)
            self.assertEqual(group["p90"], 6)
            self.assertEqual(group["fidelity"], 1)
        frame = next(f for f in proposal["per_frame"] if f["condition"] == "plus6")
        self.assertEqual((frame["frame"], frame["poc"]), (100, 0))
        self.assertEqual((frame["base_qp"], frame["zero_base_qp"]), (31, 30))
        self.assertEqual(frame["net_offset_effect"]["median"], 6)  # raw difference was 7
        self.assertEqual(frame["clipping_rate"], .25)
        self.assertEqual({g["gop_class"] for g in proposal["groups"]},
                         {"gop1-all-intra", "gop10-first-frame", "gop10-inter"})
        with self.output.with_suffix(".net_effects.csv").open() as source:
            rows = list(csv.DictReader(source))
        negative = [r for r in rows if r["condition"] == "minus6"]
        self.assertTrue(all(float(r["net_offset_effect"]) == -6 for r in negative))
        zero = [f for f in proposal["per_frame"] if f["condition"] == "zero"]
        self.assertTrue(all(f["net_offset_effect"]["median"] == 0 for f in zero))
        self.assertTrue(all(a["plus_median_minus_minus_median"] == 0 for a in proposal["asymmetry"]))

    def test_floor_conflict_is_reported_without_widening_upper_bound(self):
        self.fixture(effect=2)
        proposal = self.run_calibration()
        floors = read_floors(CRITERIA)
        for group in proposal["groups"]:
            lower, upper = group["proposed_band"]
            self.assertGreaterEqual(lower, floors["aggregate"])
            self.assertGreaterEqual(lower, floors[f"{group['condition']}:{group['direction']}"])
            self.assertEqual(upper, 2)
            self.assertTrue(group["floor_conflict"])
        self.assertEqual(hashlib.sha256(CRITERIA.read_bytes()).hexdigest(), self.criteria_sha)

    def test_proposal_is_explicit_and_never_auto_applied(self):
        proposal = self.run_calibration()
        self.assertEqual(proposal["status"], "PROPOSAL")
        self.assertTrue(proposal["council_ratification_required"])
        self.assertFalse(proposal["auto_apply"])
        self.assertTrue(all(g["status"] == "PROPOSAL" for g in proposal["groups"]))

    def test_only_floors_are_consumed_from_criteria(self):
        frozen = json.loads(CRITERIA.read_text())
        for profile in list(frozen["uniform_maps"].values()) + [frozen["checkerboard"]]:
            for key, value in profile.items():
                if isinstance(value, list):
                    value[1] = "UNREADABLE_UPPER_THRESHOLD"
                elif key == "fidelity":
                    profile[key] = "UNREADABLE_FIDELITY_THRESHOLD"
        criteria = self.root / "floors.json"
        criteria.write_text(json.dumps(frozen))
        self.assertEqual(self.run_calibration(criteria)["status"], "PROPOSAL")

    def test_missing_condition_rejects_partial_matrix(self):
        (self.root / "qp32_gop10_minus6.telemetry.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_calibration()
        self.assertFalse(self.output.exists())

    def test_duplicate_cu_and_mismatch_are_rejected(self):
        path = self.root / "qp32_gop1_zero.telemetry.csv"
        text = path.read_text()
        path.write_text(text + text.splitlines()[2] + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate telemetry CU"):
            self.run_calibration()
        self.fixture()
        path = self.root / "qp32_gop1_plus6.trace.csv"
        path.write_text(path.read_text().replace(",39,", ",40,"))
        with self.assertRaisesRegex(ValueError, "final QP mismatch"):
            self.run_calibration()

    def test_nonfinite_base_rejected(self):
        path = self.root / "qp32_gop1_zero.telemetry.csv"
        path.write_text(path.read_text().replace(",30,", ",nan,", 1))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.run_calibration()

    def test_quantiles_use_linear_interpolation(self):
        self.assertEqual(percentile([0, 10], .1), 1)
        self.assertEqual(percentile([0, 10], .9), 9)

    def test_asymmetry_subtracts_oriented_minus_median(self):
        for gop in (1, 10):
            prefix = self.root / f"qp32_gop{gop}_minus6"
            for suffix, field in ((".telemetry.csv", "final_qp"), (".trace.csv", "qp")):
                path = prefix.with_suffix(suffix)
                with path.open() as source:
                    reader = csv.DictReader(source)
                    fields, rows = reader.fieldnames, list(reader)
                for row in rows:
                    if row[field]:
                        row[field] = float(row[field]) + 2  # net minus magnitude is now 4
                self.write_csv(path, fields, rows)
        proposal = self.run_calibration()
        uniform = [row for row in proposal["asymmetry"] if row["map_family"] == "uniform"]
        self.assertTrue(all(row["plus_median_minus_minus_median"] == 2 for row in uniform))


if __name__ == "__main__":
    unittest.main()
