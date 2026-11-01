"""Offline data/identity/storage/cache contracts. All downloads are mocked."""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from prism.data.cached_dataset import CachedDataset
from prism.data.splits import SplitEntry, SplitLedger
from prism.data.transforms import center_crop_box
from tier1a.common import StepFailure, download, exclusive, read_json, sha256, write_json
from tier1a.prepare_coco500 import annotations_file, freeze_ledger, prepare, select_images
from tier1a.resources import ResourceBudget
from tier1a.teachers import crop_prompts, load_npz, save_npz
from tier1a.yuv import letterbox_yuv, load_yuv, save_yuv

FIXTURE = Path(__file__).parent / "fixtures/coco500_small.json"


def expanded_fixture() -> dict:
    """Expand six fixture templates into 900 candidates, preserving overlap."""
    fixture = read_json(FIXTURE)
    result = {"images": [], "annotations": [], "categories": fixture["categories"]}
    for batch in range(150):
        for image in fixture["images"][:6]:
            image_id = batch * 10 + image["id"]
            result["images"].append(dict(image, id=image_id, file_name=f"{image_id:012d}.jpg"))
            for ann in fixture["annotations"]:
                if ann["image_id"] == image["id"]:
                    result["annotations"].append(dict(ann, id=batch * 100 + ann["id"], image_id=image_id))
    return result


class Response(io.BytesIO):
    def __init__(self, payload: bytes, length: int | None = None):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload) if length is None else length)}


class Tier1aStep1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Default downloader captured urlopen at definition time, so also patch
        # its transport entrypoint. Tests explicitly inject mock openers.
        self.network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("NETWORK FORBIDDEN"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def test_same_seed_selects_exact_500_disjoint_ids_and_strata(self) -> None:
        data = expanded_fixture()
        first = select_images(data)
        self.assertEqual(first, select_images(data, seed=0))
        self.assertEqual(len(first), 500)
        self.assertEqual(len({r["image_id"] for r in first}), 500)
        self.assertEqual(Counter(r["stratum"] for r in first), {"person": 200, "vehicle": 150, "animal": 150})
        self.assertTrue(all(r["annotation_count"] >= 3 for r in first))
        self.assertEqual([r["image_id"] for r in first], sorted(r["image_id"] for r in first))
        self.assertNotEqual(first, select_images(data, seed=1))

    def test_selection_ignores_input_order_and_rejects_insufficient_pool(self) -> None:
        data = expanded_fixture()
        first = select_images(data)
        data["images"].reverse()
        data["annotations"].reverse()
        self.assertEqual(first, select_images(data))
        with self.assertRaisesRegex(StepFailure, "need 200"):
            select_images(read_json(FIXTURE))
        small = select_images(read_json(FIXTURE), quotas={"person": 2, "vehicle": 2, "animal": 2})
        self.assertEqual([r["image_id"] for r in small], [1, 2, 3, 4, 5, 6])

    def test_ledger_uses_existing_canonical_schema_and_never_refreezes(self) -> None:
        entries = [SplitEntry("2", "b" * 64, "train"), SplitEntry("1", "a" * 64, "train")]
        path = self.root / "ledger.json"
        ledger, status = freeze_ledger(path, entries)
        self.assertEqual(status, "CREATED")
        self.assertTrue(ledger.frozen)
        self.assertEqual(SplitLedger.read(path).require("1", "train"), entries[1])
        expected = hashlib.sha256(SplitLedger(entries).canonical_bytes()).hexdigest()
        self.assertEqual(read_json(path)["ledger_sha256"], expected)
        original = path.read_bytes()
        self.assertEqual(freeze_ledger(path, list(reversed(entries)))[1], "SKIPPED")
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(StepFailure):
            freeze_ledger(path, [SplitEntry("1", "c" * 64, "train")])
        tampered = read_json(path)
        tampered["entries"][0]["sha256"] = "d" * 64
        write_json(path, tampered)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            SplitLedger.read(path)

    def test_download_retry_sha_and_skip(self) -> None:
        path = self.root / "image.jpg"
        payload = b"mock bytes"
        calls = []
        def opener(url: str, timeout: int) -> Response:
            calls.append(url)
            if len(calls) <= 3:
                raise OSError("temporary failure")
            return Response(payload)
        created = download("http://fixture/image.jpg", path, opener=opener, sleep=lambda _: None)
        self.assertEqual(len(calls), 4)
        self.assertEqual(created["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(download("http://fixture/image.jpg", path, offline=True)["status"], "SKIPPED")
        path.write_bytes(b"tamper")
        with self.assertRaisesRegex(StepFailure, "SHA-256 mismatch"):
            download("http://fixture/image.jpg", path, opener=opener)

    def test_offline_miss_and_truncated_download_fail_with_url(self) -> None:
        url = "http://fixture/missing"
        with self.assertRaisesRegex(StepFailure, url):
            download(url, self.root / "missing", offline=True)
        with self.assertRaisesRegex(StepFailure, "Incomplete response"):
            download(url, self.root / "missing", opener=lambda *a, **kw: Response(b"x", 9),
                     retries=0, sleep=lambda _: None)
        self.assertFalse((self.root / "missing").exists())

    def test_zip_reads_only_train_member(self) -> None:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("annotations/instances_train2017.json", FIXTURE.read_bytes())
            bundle.writestr("annotations/instances_val2017.json", "SEALED DO NOT READ")
        def fake_download(url: str, path: Path, **kwargs: object) -> dict:
            return download(url, path, opener=lambda *a, **kw: Response(archive.getvalue()))
        with patch("tier1a.prepare_coco500.download", side_effect=fake_download):
            path = annotations_file(self.root)
            self.assertEqual(read_json(path), read_json(FIXTURE))
            self.assertEqual(annotations_file(self.root), path)
        self.assertFalse((self.root / "instances_val2017.json").exists())

    def test_letterbox_planes_dimensions_levels_and_lossless_storage(self) -> None:
        rgb = np.full((100, 100, 3), 255, dtype=np.uint8)
        payload, geometry = letterbox_yuv(rgb)
        self.assertEqual(len(payload), 1920 * 1080 * 3 // 2)
        self.assertEqual((geometry["active_width"], geometry["active_height"]), (1080, 1080))
        self.assertEqual((geometry["left"], geometry["top"]), (420, 0))
        y = np.frombuffer(payload[:1920 * 1080], np.uint8).reshape(1080, 1920)
        self.assertTrue(np.all(y[:, :420] == 16))
        self.assertTrue(np.all(y[:, 420:1500] == 235))
        self.assertTrue(np.all(np.frombuffer(payload[1920 * 1080:], np.uint8) == 128))
        path = self.root / "frame.yuv.gz"
        save_yuv(path, payload)
        self.assertEqual(load_yuv(path), payload)
        self.assertFalse(list(self.root.glob("*.yuv")))
        with self.assertRaises(ValueError):
            letterbox_yuv(rgb, width=1919)

    def test_crop_prompts_preserve_coordinates_and_cap_at_32(self) -> None:
        row = select_images(read_json(FIXTURE), quotas={"person": 1, "vehicle": 0, "animal": 0})[0]
        row = copy.deepcopy(row)
        row["annotations"] = [dict(row["annotations"][0], id=i) for i in range(40)]
        rois, boxes, ids, crop = crop_prompts(row)
        self.assertEqual(crop, center_crop_box(12, 16))
        self.assertEqual(len(boxes), 32)
        self.assertEqual(len(ids), 32)
        np.testing.assert_allclose(boxes[:, [0, 2]], (rois[:, [1, 3]] - crop[0]) * 518 / 12)

    def test_npz_fp16_roundtrip_and_cached_dataset_contract(self) -> None:
        ledger_path = self.root / "ledger.json"
        ledger, _ = freeze_ledger(ledger_path, [SplitEntry("1", "a" * 64, "train")])
        path = self.root / "sample.npz"
        binding = {"ledger_sha256": ledger.digest, "source_sha256": "a" * 64}
        arrays = {"x_low": np.zeros((3, 256, 256), np.float16), "dino_features": np.zeros((2, 768), np.float16),
                  "sam_masks": np.ones((2, 8, 8), bool), "rois": np.zeros((2, 5), np.float16),
                  "cls_token": np.zeros(768, np.float16), "patch_tokens": np.zeros((1369, 768), np.float16)}
        save_npz(path, arrays, binding)
        loaded = load_npz(path, binding)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], loaded[key])
            self.assertEqual(loaded[key].dtype, arrays[key].dtype)
        write_json(self.root / "index.json", {"teachers": ["DINOv2-Base", "SAM 2.1-Base"],
            "split_ledger_sha256": ledger.digest, "samples": [{"sample_id": "1", "source_sha256": "a" * 64,
            "shard": path.name, "cache_sha256": sha256(path)}]})
        dataset = CachedDataset(self.root / "index.json", ledger)
        self.assertEqual(dataset[0]["dino_features"].dtype, np.float16)
        with self.assertRaises(StepFailure):
            load_npz(path, {"different": "binding"})
        path.write_bytes(b"corrupted")
        with self.assertRaises(StepFailure):
            load_npz(path, binding)

    def test_prepare_end_to_end_idempotency_with_mock_images(self) -> None:
        output, cache = self.root / "tier1a", self.root / "cache"
        ledger = self.root / "prism/data/splits/coco500_ledger.json"
        requests = []
        def fake_download(url: str, path: Path, **kwargs: object) -> dict:
            def opener(*a: object, **kw: object) -> Response:
                requests.append(url)
                image_id = int(path.stem)
                stream = io.BytesIO()
                Image.new("RGB", (16, 12), (image_id * 30, 80, 100)).save(stream, format="JPEG")
                return Response(stream.getvalue())
            return download(url, path, opener=opener, **kwargs)
        with patch("tier1a.prepare_coco500.download", side_effect=fake_download):
            args = dict(working=self.root, annotation_path=FIXTURE,
                        quotas={"person": 2, "vehicle": 2, "animal": 2})
            first = prepare(output, cache, ledger, **args)
            ledger_bytes = ledger.read_bytes()
            second = prepare(output, cache, ledger, **args)
        self.assertEqual(len(requests), 6)
        self.assertEqual(first["images_created"], 6)
        self.assertEqual(second["images_skipped"], 6)
        self.assertEqual(second["yuv_skipped"], 6)
        self.assertEqual(second["ledger_status"], "SKIPPED")
        self.assertEqual(ledger.read_bytes(), ledger_bytes)
        self.assertFalse(list(output.rglob("*.yuv")))

    def test_budget_projection_resume_and_failed_jobs_are_charged(self) -> None:
        now = [0.0]
        path = self.root / "gpu_hours.json"
        first = ResourceBudget(path, devices=1, clock=lambda: now[0], prior_gpu_hours=50)
        self.assertLess(first.check(500)["projected_pilot_gpu_hours"], 60)
        now[0] = 1800
        first.finish("FAIL")
        self.assertAlmostEqual(read_json(path)["gpu_hours"], .5)
        second = ResourceBudget(path, devices=1, clock=lambda: now[0], prior_gpu_hours=58)
        with self.assertRaisesRegex(StepFailure, "Projected usage"):
            second.check(500)
        second.finish("FAIL")
        interrupted = ResourceBudget(path, devices=1, clock=lambda: now[0], prior_gpu_hours=58)
        self.assertIn("active", read_json(path))
        with self.assertRaisesRegex(StepFailure, "No job allowance"):
            ResourceBudget(path, devices=1, clock=lambda: now[0], prior_gpu_hours=58)
        self.assertEqual(read_json(path)["jobs"][-1]["status"], "INTERRUPTED_CHARGED_RESERVATION")

    def test_writer_lock_and_unfinished_npz_are_not_complete(self) -> None:
        lock = self.root / "writer.lock"
        with exclusive(lock):
            with self.assertRaises(StepFailure):
                with exclusive(lock):
                    pass
        self.assertFalse(lock.exists())
        path = self.root / "partial.npz"
        path.write_bytes(b"partial")
        self.assertIsNone(load_npz(path, {}))

    def test_existing_extractor_resume_and_fp16_with_mock_teacher(self) -> None:
        """Exercise the real extractor's I/O/resume flow without installing torch."""
        from scripts.cache_teacher_targets import extract

        class Tensor:
            def __init__(self, values: np.ndarray):
                self.values = values
                self.shape = values.shape

            def permute(self, *axes: int) -> Tensor:
                return Tensor(self.values.transpose(axes))

            def float(self) -> Tensor:
                return Tensor(self.values.astype(np.float32))

            def __getitem__(self, key: object) -> Tensor:
                return Tensor(self.values[key])

            def __truediv__(self, value: float) -> Tensor:
                return Tensor(self.values / value)

            def cpu(self) -> Tensor:
                return self

            def numpy(self) -> np.ndarray:
                return self.values

        source = self.root / "image.jpg"
        Image.new("RGB", (16, 12), "red").save(source)
        ledger, _ = freeze_ledger(self.root / "ledger.json", [SplitEntry("1", sha256(source), "train")])
        calls = []
        def teacher(*args: object, **kwargs: object) -> dict:
            calls.append(1)
            return {"dino_features": Tensor(np.zeros((1, 768, 37, 37), np.float32)),
                    "sam_masks": Tensor(np.ones((1, 2, 8, 8), bool)),
                    "rois": Tensor(np.zeros((2, 5), np.float32))}
        fake_torch = SimpleNamespace(from_numpy=Tensor, inference_mode=nullcontext)
        resized = (Tensor(np.zeros((1, 3, 256, 256))), Tensor(np.zeros((1, 3, 518, 518))))
        with patch.dict(sys.modules, {"torch": fake_torch}), \
                patch("prism.data.transforms.resize_inputs", return_value=resized), \
                patch("prism.data.transforms.sample_roi_targets", return_value=Tensor(np.zeros((2, 768, 1, 1)))):
            kwargs = dict(fp16=True, center_crop=True, resume=True)
            first = extract(ledger, {"1": str(source)}, self.root / "cache", teacher, **kwargs)
            second = extract(ledger, {"1": str(source)}, self.root / "cache", teacher, **kwargs)
            self.assertEqual(first["created"], 1)
            self.assertEqual(second["skipped"], 1)
            self.assertEqual(len(calls), 1)
            dataset = CachedDataset(self.root / "cache/index.json", ledger)
            self.assertEqual(dataset[0]["x_low"].dtype, np.float16)
            with self.assertRaisesRegex(ValueError, "different provenance"):
                extract(ledger, {"1": str(source)}, self.root / "cache", teacher, resume=True)

    def test_runner_freezes_before_teachers_and_reports_failures(self) -> None:
        from tier1a.run_tier1a_step1 import main
        repo = Path(__file__).resolve().parents[1]
        events = []
        def fake_prepare(output: Path, cache: Path, ledger: Path, **kwargs: object) -> dict:
            events.append("freeze")
            # No writes to the real repo ledger in tests.
            return {"status": "PASS", "ledger_sha256": "a" * 64}
        def fake_teachers(*args: object, **kwargs: object) -> dict:
            self.assertEqual(events[-1], "freeze")
            events.append("teachers")
            raise StepFailure("sam21_load", "weights URL=https://fixture/weights.pt; offline miss")
        argv = ["run_tier1a_step1.py", "--working", str(self.root), "--repo", str(repo), "--skip-install"]
        with patch.object(sys, "argv", argv), patch("tier1a.run_tier1a_step1.device_count", return_value=0), \
                patch("tier1a.prepare_coco500.prepare", side_effect=fake_prepare), \
                patch("tier1a.cache_teachers.cache_teachers", side_effect=fake_teachers), \
                patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(main(), 1)
        report = read_json(self.root / "tier1a/tier1a_step1_report.json")
        self.assertEqual(events, ["freeze", "teachers"])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["failing_step"], "sam21_load")
        self.assertFalse(report["step1_complete"])
        self.assertIn("https://fixture/weights.pt", report["reason"])
        account = read_json(self.root / "tier1a/teachers/gpu_hours.json")
        self.assertNotIn("active", account)
        self.assertEqual(account["jobs"][-1]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
