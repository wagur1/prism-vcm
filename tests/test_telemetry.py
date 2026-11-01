"""Dependency-free numerical and evidence-integrity tests for telemetry proposals."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gate0.calibrate_bands import CONDITIONS, calibrate, percentile, read_floors
from gate0.recalibrate_offline import input_map, proposed_band, recalibrate
from prism.codec.gate0 import (project_telemetry_qg, qg_net_samples,
                              qg_statistics, verify_net_effect_qg)

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


QG_FIELDS = TELEMETRY_FIELDS + ("qg_x", "qg_y", "qg_size")


class QGTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.zero, self.condition = self.root / "zero.csv", self.root / "condition.csv"

    def write(self, path, regions, frames=(0,)):
        """regions: (x,y,size,final_qp,active), optionally dict overrides."""
        rows = []
        for frame in frames:
            identity = {"frame": frame, "poc": frame, "encode_order": frame, "slice_type": "I"}
            rows.append({"record_type": "frame", **identity})
            for region in regions:
                x, y, size, qp, active, *overrides = region
                rows.append({"record_type": "cu", **identity, "x": x, "y": y,
                             "qg_x": x, "qg_y": y, "qg_size": size, "final_qp": qp,
                             "base_qp_at_cu": 30, "has_residual": int(active), "skip": 0,
                             **(overrides[0] if overrides else {})})
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=QG_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def verify(self, offsets, width, height, **kwargs):
        parameters = {"bands": {"first_frame": {-1: (5.5, 6.5), 1: (5.5, 6.5)},
                                "compensated": {-1: (4.9, 7), 1: (4.9, 7)}},
                      "min_fidelity": .98, "min_aggregate_median": 4.9, "i_frames": {0}}
        parameters.update(kwargs)
        return verify_net_effect_qg(self.zero, self.condition, offsets, width=width, height=height, **parameters)

    def test_partition_divergence_and_leaf_multiplicity_are_invariant(self):
        offsets = [[-6, -6, 6, 6]] * 2
        self.write(self.zero, [(0, 0, 64, 32, True)])
        self.write(self.condition, [(0, 0, 32, 26, True)] * 7 + [(32, 0, 32, 38, True)])
        divergent = self.verify(offsets, 64, 32)
        fine_zero = [(x, y, 16, 32, True) for y in (0, 16) for x in range(0, 64, 16)]
        fine_condition = [(x, y, 16, 26 if x < 32 else 38, True)
                          for y in (0, 16) for x in range(0, 64, 16)]
        self.write(self.zero, fine_zero)
        self.write(self.condition, fine_condition)
        self.assertEqual(divergent, self.verify(offsets, 64, 32))
        self.assertEqual(divergent["aggregate"]["1"]["count"], 4)
        self.assertEqual(divergent["aggregate"]["-1"]["median_abs"], 6)

    def test_clipped_edge_cells_are_fully_covered_and_padding_is_excluded(self):
        self.write(self.zero, [(0, 0, 32, 30, True), (32, 0, 16, 30, True)])
        self.write(self.condition, [(0, 0, 32, 36, True), (32, 0, 16, 1, True)])
        result = self.verify([[6, 6, -6], [6, 6, -6]], 32, 24)
        self.assertEqual(result["aggregate"]["1"]["count"], 4)
        self.assertEqual(result["per_frame"]["0:1"]["coverage"]["joint_fraction"], 1)
        projected = project_telemetry_qg(self.zero, 32, 24)
        self.assertEqual(projected["cell_area"], [256, 256, 128, 128])

    def test_inactive_support_is_excluded_and_all_coverage_categories_reported(self):
        self.write(self.zero, [(x, 0, 16, 30, x < 32) for x in (0, 16, 32, 48)])
        self.write(self.condition, [(x, 0, 16, 36, x in (0, 32)) for x in (0, 16, 32, 48)])
        row = qg_net_samples(project_telemetry_qg(self.zero, 64, 16),
                             project_telemetry_qg(self.condition, 64, 16), [[6] * 4])[0]
        self.assertEqual(row["values"], [6])
        for category in ("both", "zero_only", "condition_only", "neither"):
            self.assertEqual(row["coverage"][category], 1)
        with self.assertRaisesRegex(ValueError, "INVALID.*coverage"):
            self.verify([[6] * 4], 64, 16)

    def test_exactly_eighty_percent_joint_coverage_is_allowed(self):
        self.write(self.zero, [(x, 0, 16, 30, True) for x in range(0, 80, 16)])
        self.write(self.condition, [(x, 0, 16, 36, x != 64) for x in range(0, 80, 16)])
        result = self.verify([[6] * 5], 80, 16)
        self.assertEqual(result["per_frame"]["0:1"]["coverage"]["joint_fraction"], .8)
        self.assertEqual(result["aggregate"]["1"]["count"], 4)

    def test_skip_leaf_does_not_activate_qg_but_any_residual_leaf_does(self):
        self.write(self.zero, [(0, 0, 16, 30, True)])
        self.write(self.condition, [(0, 0, 16, 36, True, {"skip": 1})])
        with self.assertRaisesRegex(ValueError, "INVALID"):
            self.verify([[6]], 16, 16)
        self.write(self.condition, [(0, 0, 16, 36, True, {"skip": 1}), (0, 0, 16, 36, True)])
        self.assertEqual(self.verify([[6]], 16, 16)["1"], 6)

    def test_dedup_inconsistent_final_or_base_qp_raises_even_for_inactive_leaf(self):
        for override in ({"final_qp": 35}, {"base_qp_at_cu": 31}):
            with self.subTest(override=override):
                self.write(self.condition, [(0, 0, 16, 36, True), (0, 0, 16, 36, False, override)])
                with self.assertRaisesRegex(ValueError, "Inconsistent QP"):
                    project_telemetry_qg(self.condition, 16, 16)

    def test_missing_overlapping_and_unaligned_coverage_raise(self):
        for regions, message in (([(0, 0, 16, 30, True)], "Missing QG coverage"),
                                 ([(0, 0, 32, 30, True), (0, 0, 16, 30, True)], "Overlapping"),
                                 ([(8, 0, 32, 30, True)], "Invalid effective QG")):
            with self.subTest(message=message):
                self.write(self.zero, regions)
                with self.assertRaisesRegex(ValueError, message):
                    project_telemetry_qg(self.zero, 32, 16)

    def test_map_sign_ignores_effective_qg_average_request(self):
        self.write(self.zero, [(0, 0, 32, 30, True)])
        self.write(self.condition, [(0, 0, 32, 36, True, {"requested_offset": 0})])
        rows = qg_net_samples(project_telemetry_qg(self.zero, 32, 16),
                              project_telemetry_qg(self.condition, 32, 16), [[-6, 6]])
        self.assertEqual([r["values"] for r in rows], [[-6], [6]])
        with self.assertRaisesRegex(ValueError, "Sign-fidelity"):
            self.verify([[-6, 6]], 32, 16)

    def test_feedback_class_is_not_i_slice_class(self):
        self.write(self.zero, [(0, 0, 16, 30, True)], frames=(0, 1))
        self.write(self.condition, [(0, 0, 16, 36, True)], frames=(0, 1))
        # Only the second frame changes; all slice labels remain I.
        lines = self.condition.read_text().splitlines()
        lines[-1] = lines[-1].replace(",36,", ",35,")
        self.condition.write_text("\n".join(lines) + "\n")
        result = self.verify([[6]], 16, 16, i_frames={0, 1},
                             bands={"first_frame": {1: (5.5, 6.5)}, "compensated": {1: (4.9, 5.1)}})
        self.assertEqual(result["per_frame"]["1:1"]["frame_class"], "compensated")
        self.assertEqual(result["per_frame"]["1:1"]["kind"], "I")

    def test_explicit_bands_and_aggregate_floor_are_enforced(self):
        self.write(self.zero, [(0, 0, 16, 30, True)])
        self.write(self.condition, [(0, 0, 16, 36, True)])
        with self.assertRaisesRegex(ValueError, "Missing explicit QG band"):
            self.verify([[6]], 16, 16, bands={})
        with self.assertRaisesRegex(ValueError, "Aggregate QG effect"):
            self.verify([[6]], 16, 16, min_aggregate_median=6.1)
        with self.assertRaisesRegex(ValueError, "outside QG band"):
            self.verify([[6]], 16, 16, bands={"first_frame": {1: (4, 5)}, "compensated": {1: (4, 5)}})

    def test_frame_poc_mismatch_rejected(self):
        self.write(self.zero, [(0, 0, 16, 30, True)])
        self.write(self.condition, [(0, 0, 16, 36, True)])
        text = self.condition.read_text().replace(",0,0,0,I", ",0,7,0,I")
        self.condition.write_text(text)
        with self.assertRaisesRegex(ValueError, "frame/POC"):
            self.verify([[6]], 16, 16)

    def test_qg_type7_quantiles_and_magnitude_are_distinct(self):
        stats = qg_statistics([-10, 0])
        self.assertEqual((stats["p10"], stats["p90"]), (-9, -1))
        self.assertEqual((stats["median"], stats["median_abs"]), (-5, 5))

    def test_first_frame_fidelity_is_one_even_with_compensated_threshold_point98(self):
        self.write(self.zero, [(x, 0, 16, 30, True) for x in range(0, 1600, 16)])
        self.write(self.condition, [(x, 0, 16, 36 if x else 24, True) for x in range(0, 1600, 16)])
        with self.assertRaisesRegex(ValueError, "Sign-fidelity.*0.99 < 1.0"):
            self.verify([[6] * 100], 1600, 16)

    def test_qg_proposal_retains_frozen_floor_and_reports_conflict(self):
        profile = {"first_frame": [5.5, 6.5], "compensated": [4.9, 7]}
        result = proposed_band(qg_statistics([2, 2]), profile, "compensated", 4.9)
        self.assertEqual(result["proposed_band"], [4.9, 2])
        self.assertTrue(result["floor_conflict"])
        self.assertEqual(result["status"], "PROPOSAL")

    def test_offline_driver_uses_qg_samples_and_keeps_proposal_separate(self):
        evidence = self.root / "evidence"
        evidence.mkdir()
        criteria = ROOT / "gate0/criteria_frozen_v2.json"
        frozen_sha = hashlib.sha256(criteria.read_bytes()).hexdigest()
        for gop in (1, 10):
            for condition in CONDITIONS:
                prefix = evidence / f"qp32_gop{gop}_{condition}"
                offsets = input_map(128, 16, condition)
                self.write(prefix.with_suffix(".telemetry.csv"),
                           [(x * 16, 0, 16, 30 + v, True) for x, v in enumerate(offsets[0])], frames=(0, 1))
                prefix.with_suffix(".job.json").write_text(json.dumps({
                    "width": 128, "height": 16, "map": condition, "pin": {},
                    "config": {"qp": 32, "gop": gop, "crf": 20}}))
        output = self.root / "proposal.json"
        result = recalibrate(evidence, output, qp_groups=(32,), expected_frames=2)
        self.assertEqual(result["status"], "PROPOSAL")
        self.assertFalse(result["auto_apply"])
        self.assertTrue(result["council_ratification_required"])
        self.assertIsNone(result["gate_verdict"])
        self.assertEqual(result["coverage_problems"], [])
        for band in result["bands"]:
            self.assertEqual(band["proposed_band"], [5.5, 6.5] if band["frame_class"] == "first_frame" else [6, 6])
        self.assertEqual(hashlib.sha256(criteria.read_bytes()).hexdigest(), frozen_sha)
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            recalibrate(evidence, criteria, qp_groups=(32,), expected_frames=2)
        (evidence / "qp32_gop10_minus6.telemetry.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            recalibrate(evidence, self.root / "missing.json", qp_groups=(32,), expected_frames=2)


if __name__ == "__main__":
    unittest.main()
