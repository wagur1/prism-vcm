"""Seeded QG16 proposals from the source-bound Step 1 teacher caches.

Teacher views cover only the center crop. Cell centers are inverse-letterboxed
into that crop, nearest-sampled, and unknown cells receive zero teacher offsets.
The partial bottom QG uses the last visible pixel center (CTU padding is not a
new observation). DINO Step 1 stores normalized tokens, NOT attention weights:
patch-to-CLS cosine similarity is explicitly named as a saliency proxy here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from prism.codec.x265_bridge import qg_shape
from prism.training.oracle_sweep import validate_candidate_maps
from tier1a.common import read_json, sha256

NAMES = ("zero", "uniform_plus3", "uniform_minus3", "sam_minus3_plus2",
         "dino_cosine_top25_minus3_plus2", "checkerboard_plusminus3",
         "seeded_saliency", "sam_minus4_plus1", "sam_dino_fusion_minus4_plus6")
RECIPE = {"version": 1, "seed": 0, "qg": 16, "bounds": [-6, 6],
          "sampling": "inverse-letterbox cell centers; nearest crop sampling",
          "unknown_teacher_region": "zero offset",
          "dino": "patch-to-CLS cosine proxy, NOT attention",
          "top25": "ceil(0.25 * observed QGs), stable row-major ties",
          "random": "PCG64(seed + stable sample hash); P(-3)=normalized saliency; else +3",
          "h": "foreground -4, background +1"}

# The first eight maps and v1 recipe remain byte-identical for existing labels.
# Opt in to K=9; never silently reinterpret a source-bound K=6/K=8 checksum.
FUSION_RECIPE = {**RECIPE, "version": 2,
                 "fusion": "SAM AND observed DINO top25: -4; other SAM: 0; observed non-SAM: +6"}


def candidate_recipe(count: int) -> dict:
    if count not in (6, 8, 9):
        raise ValueError("Expected legacy K=6/K=8 or fusion K=9")
    return FUSION_RECIPE if count == 9 else RECIPE


def candidates_hash(maps: np.ndarray) -> str:
    validate_candidate_maps(maps, distinct=False)
    digest = hashlib.sha256(json.dumps(candidate_recipe(len(maps)), sort_keys=True).encode())
    digest.update(str(maps.shape).encode())
    digest.update(np.asarray(maps, dtype=np.int8).tobytes(order="C"))
    return digest.hexdigest()


def generate_candidates(width: int, height: int, sam_foreground: np.ndarray,
                        dino_saliency: np.ndarray, *, sample_id: str,
                        seed: int = 0, observed: np.ndarray | None = None,
                        include_fusion: bool = False) -> np.ndarray:
    shape = qg_shape(width, height)
    sam = np.asarray(sam_foreground, dtype=bool)
    saliency = np.asarray(dino_saliency, dtype=np.float64)
    known = np.ones(shape, dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    if sam.shape != shape or saliency.shape != shape or known.shape != shape:
        raise ValueError("Teacher grids must match the padded QG16 domain")
    if not np.isfinite(saliency).all() or seed < 0:
        raise ValueError("Require finite saliency and nonnegative seed")
    indices = np.flatnonzero(known)
    if not len(indices):
        raise ValueError("No teacher-observed QGs")
    top = np.zeros(shape, dtype=bool)
    order = np.argsort(-saliency.ravel()[indices], kind="stable")
    top.ravel()[indices[order[:int(np.ceil(len(indices) / 4))]]] = True
    low, high = saliency[known].min(), saliency[known].max()
    probability = ((saliency - low) / (high - low) if high > low
                   else np.full(shape, .5))
    identity = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "little")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, identity])))
    maps = np.stack([
        np.zeros(shape), np.full(shape, 3), np.full(shape, -3),
        np.where(known, np.where(sam, -3, 2), 0),
        np.where(known, np.where(top, -3, 2), 0),
        np.where(np.indices(shape).sum(axis=0) % 2 == 0, 3, -3),
        np.where(known, np.where(rng.random(shape) < probability, -3, 3), 0),
        np.where(known, np.where(sam, -4, 1), 0),
    ]).astype(np.int8)
    if include_fusion:
        fusion = np.where(known, np.where(sam, np.where(top, -4, 0), 6), 0).astype(np.int8)
        maps = np.concatenate((maps, fusion[None]), axis=0)
    validate_candidate_maps(maps, distinct=False)
    return maps


def project_crop(values: np.ndarray, crop: np.ndarray, geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Project a 2D teacher field without treating unseen source pixels as BG."""
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite two-dimensional teacher field")
    left, top, right, bottom = map(float, crop)
    if not (0 <= left < right <= geometry["source_width"]
            and 0 <= top < bottom <= geometry["source_height"]):
        raise ValueError("Invalid teacher crop geometry")
    rows, cols = qg_shape(geometry["width"], geometry["height"])
    x = (np.minimum(np.arange(cols) * 16 + 8, geometry["width"] - .5)
         - geometry["left"]) / geometry["scale_x"]
    y = (np.minimum(np.arange(rows) * 16 + 8, geometry["height"] - .5)
         - geometry["top"]) / geometry["scale_y"]
    known = ((y[:, None] >= top) & (y[:, None] < bottom)
             & (x[None, :] >= left) & (x[None, :] < right))
    ix = np.clip(((x - left) / (right - left) * values.shape[1]).astype(int), 0, values.shape[1] - 1)
    iy = np.clip(((y - top) / (bottom - top) * values.shape[0]).astype(int), 0, values.shape[0] - 1)
    return np.where(known, values[iy[:, None], ix[None, :]], 0), known


def cached_candidates(root: Path, sample_id: str, geometry: dict, *,
                      source_sha: str, ledger_sha: str, selection_sha: str,
                      include_fusion: bool = False) -> tuple[np.ndarray, dict]:
    shards, hashes = {}, {}
    for teacher in ("sam21", "dinov2"):
        path = root / teacher / f"{sample_id}.npz"
        receipt = read_json(path.with_suffix(".json"))
        hashes[teacher] = sha256(path)
        binding = receipt["binding"]
        if (hashes[teacher] != receipt["sha256"] or binding["source_sha256"] != source_sha
                or binding["ledger_sha256"] != ledger_sha or binding["selection_sha256"] != selection_sha):
            raise ValueError(f"Teacher cache binding/checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as shard:
            shards[teacher] = {key: shard[key] for key in shard.files}
    sam, dino = shards["sam21"], shards["dinov2"]
    if not np.array_equal(sam["crop_box"], dino["crop_box"]):
        raise ValueError("Teacher crops disagree")
    masks = sam["masks"]
    if masks.ndim != 3 or masks.shape[1:] != (518, 518) or masks.dtype != bool:
        raise ValueError("Expected Step 1 boolean SAM [K,518,518]")
    foreground, known = project_crop(masks.any(axis=0), sam["crop_box"], geometry)
    patches, cls = dino["patch_tokens"].astype(np.float64), dino["cls_token"].astype(np.float64)
    if patches.shape != (1369, 768) or cls.shape != (768,):
        raise ValueError("Expected DINOv2-Base Step 1 tokens")
    norm = np.linalg.norm(patches, axis=1) * np.linalg.norm(cls)
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError("Invalid DINO token norms")
    saliency, _ = project_crop(((patches @ cls) / norm).reshape(37, 37), dino["crop_box"], geometry)
    maps = generate_candidates(geometry["width"], geometry["height"], foreground,
                               saliency, sample_id=sample_id, observed=known,
                               include_fusion=include_fusion)
    return maps, {"teacher_sha256": hashes, "recipe": candidate_recipe(len(maps)),
                  "observed_qgs": int(known.sum()), "total_qgs": int(known.size),
                  "unique_maps": len({m.tobytes() for m in maps})}
