#!/usr/bin/env python3
"""Acquire the 500-image train2017 carve-out, then freeze real source identities."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism.data.splits import SplitEntry, SplitLedger
from tier1a.common import StepFailure, check_storage, download, exclusive, read_json, sha256, write_json

REPO = Path(__file__).resolve().parents[1]
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL = "http://images.cocodataset.org/train2017/"
QUOTAS = {"person": 200, "vehicle": 150, "animal": 150}
CATEGORIES = {"person": {1}, "vehicle": {2, 3, 4, 5, 6, 7, 8, 9},
              "animal": {16, 17, 18, 19, 20, 21, 22, 23, 24, 25}}
TARGET_IDS = set.union(*CATEGORIES.values())


def select_images(data: dict, seed: int = 0, quotas: dict[str, int] | None = None) -> list[dict]:
    """Disjoint strata; person-rich means >=2 non-crowd people.

    Sort candidates by image_id before seeded shuffling. Assign person, vehicle,
    animal in that order, excluding already assigned images. Sort final output.
    """
    quotas = QUOTAS if quotas is None else quotas
    if set(quotas) != set(QUOTAS) or any(n < 0 for n in quotas.values()):
        raise ValueError("Expected nonnegative person/vehicle/animal quotas")
    images = {row["id"]: row for row in data["images"]}
    if len(images) != len(data["images"]):
        raise ValueError("Duplicate COCO image IDs")
    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    pools: dict[str, list[int]] = {name: [] for name in QUOTAS}
    for image_id in sorted(images):
        rows = annotations[image_id]
        if len(rows) < 3:
            continue
        counts = Counter(row["category_id"] for row in rows if not row.get("iscrowd", 0))
        for name, category_ids in CATEGORIES.items():
            if sum(counts[c] for c in category_ids) >= (2 if name == "person" else 1):
                pools[name].append(image_id)
    rng, selected, assigned = random.Random(seed), [], set()
    for name in QUOTAS:
        candidates = [image_id for image_id in pools[name] if image_id not in assigned]
        rng.shuffle(candidates)
        if len(candidates) < quotas[name]:
            raise StepFailure("selection", f"{name}: need {quotas[name]}, have {len(candidates)}")
        for image_id in candidates[:quotas[name]]:
            row = images[image_id]
            filename = row["file_name"]
            if filename != f"{image_id:012d}.jpg":
                raise StepFailure("selection", f"Unexpected train2017 filename: {filename}")
            prompts = [a for a in annotations[image_id] if a["category_id"] in TARGET_IDS
                       and not a.get("iscrowd", 0) and a["bbox"][2] > 0 and a["bbox"][3] > 0]
            prompts.sort(key=lambda a: (-a.get("area", a["bbox"][2] * a["bbox"][3]), a["id"]))
            selected.append({"image_id": image_id, "sample_id": str(image_id), "file_name": filename,
                "width": row["width"], "height": row["height"], "stratum": name,
                "annotation_count": len(annotations[image_id]),
                "annotations": [{key: a[key] for key in ("id", "category_id", "bbox")} for a in prompts]})
            assigned.add(image_id)
    return sorted(selected, key=lambda row: row["image_id"])


def annotations_file(cache: Path, *, offline: bool = False) -> Path:
    archive = cache / "annotations_trainval2017.zip"
    download(ANNOTATIONS_URL, archive, offline=offline)
    target = cache / "instances_train2017.json"
    receipt = target.with_suffix(".sha256.json")
    if target.exists() and receipt.exists():
        meta = read_json(receipt)
        if sha256(target) != meta["sha256"] or sha256(archive) != meta["archive_sha256"]:
            raise StepFailure("annotations", "Cached train annotations changed")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    # The ZIP contains val annotations, but only this train member is read.
    with zipfile.ZipFile(archive) as bundle:
        with bundle.open("annotations/instances_train2017.json") as source, temporary.open("wb") as dest:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                dest.write(block)
    write_json(receipt, {"sha256": sha256(temporary), "archive_sha256": sha256(archive)})
    temporary.replace(target)
    return target


def freeze_ledger(path: Path, entries: list[SplitEntry]) -> tuple[SplitLedger, str]:
    draft = SplitLedger(entries)
    if path.exists():
        ledger = SplitLedger.read(path)
        if not ledger.frozen or ledger.digest != draft.digest:
            raise StepFailure("ledger", "Refusing to replace a frozen ledger or accept an unfrozen file")
        if read_json(path).get("ledger_sha256", ledger.digest) != ledger.digest:
            raise StepFailure("ledger", "ledger_sha256 alias mismatch")
        return ledger, "SKIPPED"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".freeze")
    # A .freeze file is never used by consumers. Recover interrupted publication.
    temporary.unlink(missing_ok=True)
    draft.freeze(temporary)
    data = read_json(temporary)
    data["ledger_sha256"] = draft.digest  # Existing reader uses frozen_sha256.
    write_json(temporary, data)
    temporary.replace(path)
    return SplitLedger.read(path), "CREATED"


def prepare(output: Path, cache: Path, ledger_path: Path, *, working: Path,
            offline: bool = False, annotation_path: Path | None = None,
            quotas: dict[str, int] | None = None) -> dict:
    import numpy as np
    from PIL import Image
    from tier1a.yuv import letterbox_yuv, load_yuv, save_yuv

    check_storage(working, 1_500_000_000)
    annotation_path = annotation_path or annotations_file(cache, offline=offline)
    records = select_images(read_json(annotation_path), quotas=quotas)
    selection = {"schema_version": 1, "source": "train2017", "seed": 0,
        "quotas": QUOTAS if quotas is None else quotas, "annotations_sha256": sha256(annotation_path),
        "person_rich_min_non_crowd": 2, "stratum_priority": list(QUOTAS), "images": records}
    selection_path = output / "coco500_selection.json"
    if selection_path.exists() and read_json(selection_path) != selection:
        raise StepFailure("selection", "Existing selection differs; refusing to change identities")
    if not selection_path.exists():
        write_json(selection_path, selection)
    frozen = SplitLedger.read(ledger_path) if ledger_path.exists() else None
    if frozen and (not frozen.frozen or set(frozen.by_id) != {r["sample_id"] for r in records}):
        raise StepFailure("ledger", "Existing ledger is unfrozen or selects different images")
    entries, manifests = [], []
    counts = {"images_created": 0, "images_skipped": 0, "yuv_created": 0, "yuv_skipped": 0}
    for row in records:
        check_storage(working, 10_000_000)
        image_path = output / "coco500/images" / row["file_name"]
        expected = frozen.require(row["sample_id"], "train").sha256 if frozen else None
        result = download(IMAGE_URL + row["file_name"], image_path, expected_sha256=expected, offline=offline)
        counts["images_" + result["status"].lower()] += 1
        with Image.open(image_path) as image:
            image.load()
            if image.size != (row["width"], row["height"]):
                raise StepFailure("image_decode", f"COCO dimensions disagree for {image_path}")
            rgb = np.asarray(image.convert("RGB")).copy()
        entry = SplitEntry(row["sample_id"], result["sha256"], "train", "train2017")
        entries.append(entry)
        yuv = output / "coco500/yuv" / f"{row['image_id']}.yuv.gz"
        metadata = yuv.with_name(yuv.name + ".json")
        if yuv.exists() and metadata.exists():
            meta = read_json(metadata)
            if (meta["source_sha256"] != entry.sha256 or meta["sha256"] != sha256(yuv)
                    or meta["geometry"]["raw_sha256"] != hashlib.sha256(load_yuv(yuv)).hexdigest()):
                raise StepFailure("yuv", f"Cached YUV changed: {yuv}")
            counts["yuv_skipped"] += 1
        else:
            payload, geometry = letterbox_yuv(rgb)
            save_yuv(yuv, payload)
            meta = {"source_sha256": entry.sha256, "sha256": sha256(yuv), "geometry": geometry}
            write_json(metadata, meta)
            counts["yuv_created"] += 1
        manifests.append(dict(row, source_sha256=entry.sha256, image=str(image_path),
                              yuv=str(yuv), yuv_metadata=meta))
    ledger, status = freeze_ledger(ledger_path, entries)
    freeze_ledger(output / "coco500_ledger.json", entries)
    write_json(output / "coco500/manifest.json", {"ledger_sha256": ledger.digest, "images": manifests})
    write_json(output / "coco500/sources.json", {r["sample_id"]: r["image"] for r in manifests})
    report = {"status": "PASS", "images": len(records), "ledger_status": status,
              "ledger_sha256": ledger.digest, "ledger": str(ledger_path), **counts}
    write_json(output / "prepare_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/tier1a"))
    parser.add_argument("--cache", type=Path, default=Path("/kaggle/working/cache"))
    parser.add_argument("--ledger", type=Path, default=REPO / "prism/data/splits/coco500_ledger.json")
    parser.add_argument("--working", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run-annotations", type=Path, help="Selection only; never downloads or freezes")
    parser.add_argument("--dry-run-quotas", type=int, nargs=3, metavar=("PERSON", "VEHICLE", "ANIMAL"))
    args = parser.parse_args()
    try:
        if args.dry_run_annotations:
            quotas = dict(zip(QUOTAS, args.dry_run_quotas)) if args.dry_run_quotas else None
            rows = select_images(read_json(args.dry_run_annotations), quotas=quotas)
            print(json.dumps({"status": "DRY_RUN", "count": len(rows), "image_ids": [r["image_id"] for r in rows]}))
        else:
            if args.dry_run_quotas:
                raise StepFailure("arguments", "Quotas can only change in a dry run")
            with exclusive(args.output / ".prepare.lock"):
                print(json.dumps(prepare(args.output, args.cache, args.ledger,
                                         working=args.working, offline=args.offline)))
        return 0
    except Exception as exc:
        report = {"status": "FAIL", "step": getattr(exc, "step", "prepare"), "reason": str(exc)}
        write_json(args.output / "prepare_report.json", report)
        print(json.dumps(report), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
