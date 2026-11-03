#!/usr/bin/env python3
"""Paired dev/pilot screening of M1 and three M2 variants.

Examples (no downloads or training):
  python scripts/screen_m2_enhancements.py --mock --limit 10
  python scripts/screen_m2_enhancements.py --images /data/train2017 \
    --checkpoint student_seed_0.pt --gate /build/gate0 --limit 50 \
    --detector yolo --annotations instances_train2017.json --weights yolo11l.pt

Mock uses synthetic frames, scalar quantization and zlib; its byte deltas and
pixel similarity are NOT HEVC gains or detector mAP. Real mode uses source-SHA
bound train2017 dev/pilot entries, the pinned stock/patched GATE-0 pair, FFmpeg
decode and optional pooled COCO YOLO11-L AP@[.50:.95]. No val2017 tuning.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
from pathlib import Path
import sys
import tempfile
import time
import zlib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from prism.codec.ffmpeg_stream import frame_bytes
from prism.codec.qp_maps import (
    RATE_OFFSET_SCHEDULE, project_adaptive_qp_map, project_zero_mean_qp_map, qg_area_weights,
)
from prism.codec.x265_bridge import qg_shape
from prism.data.splits import SplitLedger
from prism.preprocessing.background_filter import apply_background_smoothing
from tier1a.common import read_json, sha256, write_json
from tier1a.oracle_codec import Gate0Encoder, yuv_to_rgb

VARIANTS = ("M1", "M2_baseline", "M2_asym", "M2_asym_prefilter")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def smooth_yuv(payload, qmap, width, height, strength):
    if len(payload) != frame_bytes(width, height):
        raise ValueError("Expected one YUV420p8 frame")
    raw, n = np.frombuffer(payload, np.uint8), width * height
    planes = (raw[:n].reshape(height, width),
              raw[n:n * 5 // 4].reshape(height // 2, width // 2),
              raw[n * 5 // 4:].reshape(height // 2, width // 2))
    return b"".join(p.tobytes() for p in apply_background_smoothing(planes, qmap, strength))


def variants(payload, raw_map, width, height, crf, *, high=6., strength=1.,
             delta_rate_offset=None, quantize_int8=False, known=None):
    """Use the same unrounded student output in every QP ablation."""
    if np.shape(raw_map) != qg_shape(width, height):
        raise ValueError("Raw controls must match the bridge QG16 shape")
    weights = qg_area_weights(width, height)
    baseline = project_zero_mean_qp_map(raw_map, weights=weights, mask=known)
    asym = project_adaptive_qp_map(raw_map, crf=crf, weights=weights, mask=known,
        high=high, delta_rate_offset=delta_rate_offset, quantize_int8=quantize_int8)
    # Preserve original spatial foreground even if the adaptive DC crosses zero.
    spatial = project_zero_mean_qp_map(raw_map, weights=weights, mask=known, quantize_int8=False)
    filter_map = np.where(spatial > 0, asym, 0)
    filtered = smooth_yuv(payload, filter_map, width, height, strength)
    return (("M1", payload, None), ("M2_baseline", payload, baseline),
            ("M2_asym", payload, asym), ("M2_asym_prefilter", filtered, asym))


class MockEncoder:
    """Deterministic functional substitute, deliberately not an HEVC estimator."""
    def __call__(self, payload, offsets, frame, directory, *, stock=False, crf=28.):
        width, height = frame["geometry"]["width"], frame["geometry"]["height"]
        raw = np.frombuffer(payload, np.uint8).copy()
        q = np.zeros(qg_shape(width, height), np.float32) if offsets is None else offsets
        if stock and offsets is not None:
            raise ValueError("Stock encoder must receive no offsets")
        if q.shape != qg_shape(width, height) or not np.isfinite(q).all():
            raise ValueError("Invalid mock QG16 shape")
        step = np.exp2((crf + q.repeat(16, 0).repeat(16, 1)[:height, :width] - 4) / 6)
        y = raw[:width * height].reshape(height, width)
        y[:] = np.rint(np.rint(y / step) * step).clip(16, 235).astype(np.uint8)
        self.decoded = raw.tobytes()
        encoded = zlib.compress(self.decoded)
        return {"whole_bitstream_bytes": len(encoded), "bits": len(encoded) * 8,
                "bitstream_sha256": digest(encoded)}

    def decode(self, result, frame):
        return yuv_to_rgb(self.decoded, frame["geometry"]["width"], frame["geometry"]["height"])


def synthetic_frame(index, width, height, seed):
    """Textured background and sharp foreground; no pretend COCO labels."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    y = rng.integers(50, 190, (height, width), dtype=np.uint8)
    y[height // 4:3 * height // 4, width // 3:2 * width // 3] = 210
    chroma = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    payload = y.tobytes() + chroma.tobytes() * 2
    weights = qg_area_weights(width, height)
    rows, cols = weights.shape
    sam = np.zeros((rows, cols), dtype=bool)
    sam[rows // 4:3 * rows // 4, cols // 3:2 * cols // 3] = True
    # Exercise the joint teacher proposal in mock, without claiming a trained M2.
    from tier1a.oracle_candidates import generate_candidates
    maps = generate_candidates(width, height, sam, np.where(sam, 1., .1),
        sample_id=str(index), observed=weights > 0, include_fusion=True)
    frame = {"image_id": index, "sample_id": str(index), "source_sha256": digest(payload),
             "geometry": {"width": width, "height": height}}
    return frame, payload, maps[-1].astype(np.float32)


def select_frames(args):
    """Read only frozen dev/pilot train2017 files and verify every source hash."""
    from PIL import Image
    from tier1a.yuv import letterbox_yuv
    ledger = SplitLedger.read(args.ledger)
    entries = sorted((e for e in ledger.entries if e.split == args.split), key=lambda e: e.sample_id)
    if len(entries) < args.limit:
        raise ValueError(f"Requested {args.limit} images, only {len(entries)} in {args.split}")
    selected = np.random.default_rng(args.seed).permutation(len(entries))[:args.limit]
    for i in selected:
        entry = ledger.require(entries[i].sample_id, args.split)
        if entry.source != "train2017":
            raise ValueError("Screening requires train2017 dev/pilot sources")
        path = args.images / f"{int(entry.sample_id):012d}.jpg"
        if sha256(path) != entry.sha256:
            raise ValueError(f"Source checksum mismatch: {path}")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        payload, geometry = letterbox_yuv(rgb, args.width, args.height)
        yield {"image_id": int(entry.sample_id), "sample_id": entry.sample_id,
               "source_sha256": entry.sha256, "ledger_sha256": ledger.digest,
               "geometry": geometry}, payload, None


class StudentMaps:
    def __init__(self, checkpoint_path, device):
        import torch
        from prism.models.student import MobileNetV4HybridController
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        backbone = {"mobilenetv4_conv_small": "timm", "small_conv_smoke_only": "small"}.get(
            checkpoint.get("backbone_name"))
        if backbone is None:
            raise ValueError("Unknown checkpoint backbone")
        self.model = MobileNetV4HybridController(backbone=backbone, allow_fallback=False)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(device).eval()
        self.device = device

    def __call__(self, payload, frame, crf):
        import torch
        from prism.models.student import upsample_qp_map
        from tier1a.run_tier1a_step3 import rgb_to_student_input
        width, height = frame["geometry"]["width"], frame["geometry"]["height"]
        x = rgb_to_student_input(yuv_to_rgb(payload, width, height))
        with torch.inference_mode():
            out = self.model(x.unsqueeze(0).to(self.device),
                torch.full((1, 1), crf / 51., device=self.device), for_encoding=False)
            raw = upsample_qp_map(out["delta_qp"], width=width, height=height)[0, 0].cpu().numpy()
        return raw, None


def saved_map(directory, frame, crf):
    """Replay unrounded controls bound to the exact encoded input bytes.

    <sample_id>.npz: raw_qg_map [C,Hq,Wq], crfs [C], input_yuv_sha256 scalar
    string; optional known_mask [Hq,Wq]. A 2D map requires exactly one CRF.
    """
    with np.load(directory / f"{frame['sample_id']}.npz", allow_pickle=False) as data:
        if str(data["input_yuv_sha256"].item()) != frame["input_yuv_sha256"]:
            raise ValueError("Saved map/input checksum mismatch")
        crfs = np.asarray(data["crfs"])
        if crfs.ndim != 1 or len(set(crfs.tolist())) != len(crfs):
            raise ValueError("Saved map CRFs must be a unique vector")
        indices = np.flatnonzero(crfs == crf)
        if len(indices) != 1:
            raise ValueError(f"No unique saved raw map for CRF {crf}")
        raw = data["raw_qg_map"]
        if raw.ndim == 2:
            raw = raw[None]
        if raw.ndim != 3 or raw.shape[0] != len(crfs):
            raise ValueError("Saved raw maps must have shape [CRFs,Hq,Wq]")
        return raw[indices[0]], data["known_mask"] if "known_mask" in data else None


def run_screen(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError("Use a fresh output directory to preserve screening provenance")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = {"schema_version": 1, "status": "RUNNING", "config": config,
        "backend": "mock_scalar_quantization_zlib" if args.mock else "pinned_x265",
        "rate_is_hevc": not args.mock, "bd_rate_claimed": False,
        "map_source": "synthetic_joint_teacher" if args.mock else ("student" if args.checkpoint else "saved_raw_maps"),
        "rate_offset_schedule": RATE_OFFSET_SCHEDULE, "frames": [], "summary": [],
        "versions": {"python": sys.version, "numpy": np.__version__},
        "detector": "unavailable (mock)" if args.mock else args.detector,
        "code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in (
            Path(__file__).resolve(), REPO / "prism/codec/qp_maps.py",
            REPO / "prism/preprocessing/background_filter.py", REPO / "tier1a/oracle_candidates.py",
            REPO / "tier1a/oracle_codec.py", REPO / "tier1a/oracle_detector.py",
            REPO / "tier1a/run_tier1a_step3.py", REPO / "tier1a/yuv.py",
            REPO / "prism/codec/x265_bridge.py", REPO / "prism/codec/ffmpeg_stream.py",
            REPO / "prism/models/student.py", REPO / "tier1a/step3_detectors.py")}}
    path = output / "report.json"
    write_json(path, report)
    started = time.monotonic()
    try:
        provider, scorer = None, None
        if args.mock:
            encoder = MockEncoder()
            frames = (synthetic_frame(i, args.width, args.height, args.seed) for i in range(args.limit))
        else:
            report["versions"]["Pillow"] = importlib.metadata.version("Pillow")
            encoder = Gate0Encoder(args.gate.resolve(), ffmpeg=args.ffmpeg,
                                   deadline=started + args.timeout_seconds)
            report["codec"] = {"stack": encoder.stack, "patched_pin": encoder.pin,
                               "stock_pin": encoder.stock_pin, "config": asdict(encoder.config)}
            frames = select_frames(args)
            if args.checkpoint:
                provider = StudentMaps(args.checkpoint, args.device)
                report["checkpoint_sha256"] = sha256(args.checkpoint)
            report["ledger_sha256"] = sha256(args.ledger)
            if args.detector == "yolo":
                from tier1a.oracle_detector import CocoAP50
                ledger = SplitLedger.read(args.ledger)
                ids = {int(e.sample_id) for e in ledger.entries if e.split == args.split}
                scorer = CocoAP50(args.annotations, args.weights, ids, device=args.device)
                report["detector"] = scorer.provenance
                report["annotations_sha256"] = sha256(args.annotations)
        predictions = {(crf, name): [] for crf in args.crfs for name in VARIANTS}
        for index, (frame, payload, mock_map) in enumerate(frames):
            if time.monotonic() - started >= args.timeout_seconds:
                raise TimeoutError("Screening deadline reached")
            frame["input_yuv_sha256"] = digest(payload)
            observation = {**frame, "observations": []}
            source_rgb = yuv_to_rgb(payload, args.width, args.height) if args.mock else None
            with tempfile.TemporaryDirectory(prefix="codec_", dir=output) as scratch:
                for crf in args.crfs:
                    if time.monotonic() - started >= args.timeout_seconds:
                        raise TimeoutError("Screening deadline reached")
                    if args.mock:
                        raw, known = mock_map, None
                    elif provider is not None:
                        raw, known = provider(payload, frame, crf)
                    else:
                        raw, known = saved_map(args.maps_dir, frame, crf)
                    for name, pixels, qmap in variants(payload, raw, args.width, args.height, crf,
                            high=args.high, strength=args.strength, delta_rate_offset=args.delta_rate_offset,
                            quantize_int8=args.int8, known=known):
                        result = encoder(pixels, qmap, frame, Path(scratch) / f"{crf}_{name}",
                                         stock=name == "M1", crf=crf)
                        if result["whole_bitstream_bytes"] <= 0 or result["bits"] != result["whole_bitstream_bytes"] * 8:
                            raise ValueError("Invalid whole-bitstream accounting")
                        rgb = encoder.decode(result, frame)
                        if rgb.shape != (args.height, args.width, 3) or rgb.dtype != np.uint8:
                            raise ValueError("Decoded frame shape/dtype mismatch")
                        record = {k: result[k] for k in ("whole_bitstream_bytes", "bits", "bitstream_sha256")}
                        record.update(variant=name, crf=crf, input_yuv_sha256=digest(pixels),
                            raw_map_sha256=digest(np.asarray(raw, np.float32).tobytes()),
                            known_mask_sha256=None if known is None else digest(np.asarray(known, bool).tobytes()))
                        if qmap is not None:
                            w = qg_area_weights(args.width, args.height)
                            record["map"] = {"dtype": str(qmap.dtype), "shape": list(qmap.shape),
                                "min": float(qmap.min()), "max": float(qmap.max()),
                                "area_mean": float((qmap * w).sum() / w.sum()),
                                "sha256": digest(qmap.tobytes())}
                        if scorer is not None:
                            rows = scorer.predict_rows(rgb, frame)
                            predictions[crf, name].extend(rows)
                            record["detections"] = rows
                        if args.mock:
                            record["mock_pixel_similarity"] = float(1 - np.abs(
                                rgb.astype(float) - source_rgb).mean() / 255)
                        observation["observations"].append(record)
                        # Compare patched zero with stock once per CRF before accepting measurements.
                        if index == 0 and name == "M1":
                            zero = encoder(payload, np.zeros(qg_shape(args.width, args.height), np.float32),
                                frame, Path(scratch) / f"{crf}_zero", crf=crf)
                            if zero["bitstream_sha256"] != result["bitstream_sha256"]:
                                raise ValueError(f"Stock/zero bit identity failed at CRF {crf}")
            report["frames"].append(observation)
            write_json(path, report)
            print(f"Screened {index + 1}/{args.limit}: {frame['sample_id']}", flush=True)
        image_ids = [f["image_id"] for f in report["frames"]]
        for crf in args.crfs:
            cells = [o for f in report["frames"] for o in f["observations"] if o["crf"] == crf]
            anchor = sum(o["whole_bitstream_bytes"] for o in cells if o["variant"] == "M1")
            for name in VARIANTS:
                group = [o for o in cells if o["variant"] == name]
                size = sum(o["whole_bitstream_bytes"] for o in group)
                metrics = None
                if scorer is not None:
                    from tier1a.step3_detectors import evaluate_pooled_coco_metrics
                    metrics = evaluate_pooled_coco_metrics(scorer.coco, image_ids, predictions[crf, name])
                report["summary"].append({"crf": crf, "variant": name, "images": len(group),
                    "total_bytes": size, "bitrate_delta_pct_vs_M1": 100 * (size / anchor - 1),
                    "map_50_95": None if metrics is None else metrics["ap_50_95"],
                    "coco_metrics": metrics})
        report.update(status="PASS", zero_stock_identity="PASS",
                      physical_encodes=args.limit * len(args.crfs) * 4 + len(args.crfs))
    except Exception as exc:
        report.update(status="FAIL", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        write_json(path, report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--images", type=Path, help="COCO train2017 JPEG directory")
    parser.add_argument("--ledger", type=Path, default=REPO / "prism/data/splits/coco500_tier1a_splits.json")
    parser.add_argument("--split", choices=("dev", "pilot"), default="dev")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--crfs", type=int, nargs="+", default=[20, 24, 28, 32, 36])
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--checkpoint", type=Path)
    controls.add_argument("--maps-dir", type=Path)
    parser.add_argument("--high", type=float, choices=(6., 8.), default=6.)
    parser.add_argument("--delta-rate-offset", type=float, help="Override the extra CRF margin, after convex compensation")
    parser.add_argument("--strength", type=float, default=1.)
    parser.add_argument("--int8", action="store_true", help="Quantize adaptive maps; default float32")
    parser.add_argument("--gate", type=Path, default=REPO / "gate0")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--detector", choices=("none", "yolo"), default="none")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=3600.)
    parser.add_argument("--output", type=Path, default=REPO / "outputs" / f"m2_screen_{time.time_ns()}")
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 50 or args.seed < 0:
        parser.error("--limit must be 1..50 (10..50 recommended); --seed must be nonnegative")
    if not args.crfs or len(set(args.crfs)) != len(args.crfs) or any(not 0 <= c <= 51 for c in args.crfs):
        parser.error("CRFs must be unique integers in [0,51]")
    frame_bytes(args.width, args.height)
    if not np.isfinite(args.strength) or args.strength < 0 or not np.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("Require finite nonnegative strength and positive timeout")
    if args.delta_rate_offset is not None and (not np.isfinite(args.delta_rate_offset) or args.delta_rate_offset < 0):
        parser.error("--delta-rate-offset must be finite and nonnegative")
    if args.mock and (args.detector != "none" or args.checkpoint or args.maps_dir or args.images):
        parser.error("Mock mode uses synthetic input/maps and no detector")
    if not args.mock and (args.images is None or (args.checkpoint is None and args.maps_dir is None)):
        parser.error("Real mode requires --images and --checkpoint or --maps-dir")
    if args.detector == "yolo" and (args.annotations is None or args.weights is None):
        parser.error("YOLO requires local --annotations and --weights")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = run_screen(args)
    print(f"\nBackend: {report['backend']}; bytes compared at equal CRF; BD-rate not established")
    print(f"{'CRF':>3} {'Variant':22} {'Total bytes':>12} {'vs M1':>9} {'mAP50:95':>10}")
    for row in report["summary"]:
        metric = "N/A" if row["map_50_95"] is None else f"{row['map_50_95']:.4f}"
        print(f"{row['crf']:3} {row['variant']:22} {row['total_bytes']:12,} "
              f"{row['bitrate_delta_pct_vs_M1']:+8.2f}% {metric:>10}")
    print(f"Report: {args.output.resolve() / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
