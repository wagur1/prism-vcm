# Tier 1a Step 1 on Kaggle

Step 1 acquires 500 **train2017** images, freezes their SHA ledger, and caches
**DINOv2-Base + SAM 2.1-Base** targets. It runs no Student, oracle, calibration,
evaluation, or video job. A Step 1 `PASS` is artifact readiness, not the Tier 1a
pilot gate. CONFIG v1.4 amends D/E; its unchanged C/F constraints are inherited
from [CONFIG v1.3](../docs/CONFIG_V1_3.md).

Enable Internet and a T4 accelerator; Python 3.10+ is required. The chair must
publish this implementation before cloning. Reuse existing checkouts without
pulling/resetting them. One notebook cell:

```python
!bash -c 'set -e; cd /kaggle/working; if [ ! -d prism-vcm/.git ]; then git clone https://github.com/wagur1/prism-vcm.git prism-vcm; fi; cd prism-vcm; python3 -u tier1a/run_tier1a_step1.py --prior-gpu-hours 0'
```

Replace `0` with **all pilot GPU-hours outside this Step 1 account**. The persistent
Step 1 account adds its own previous jobs automatically; do not count those twice.
Keep this account across Kaggle sessions and carry its cumulative usage into
Step 2. With a provisioned checkout:

```bash
bash tier1a/run_step1_kaggle.sh --prior-gpu-hours 0
# Or:
python3 -u tier1a/run_tier1a_step1.py --prior-gpu-hours 0
# Verified download/source caches and preinstalled dependencies only:
python3 -u tier1a/run_tier1a_step1.py --offline --prior-gpu-hours 0
```

The Python entrypoint also works as a standalone bootstrap file: use
`python3 run_tier1a_step1.py --repo /kaggle/working/prism-vcm`. It clones only when
needed. `--skip-install` uses provisioned dependencies; `--offline` additionally
forbids data/weight/source downloads. The runner uses pip against Meta's official
`git+https://github.com/facebookresearch/sam2.git` (`SAM-2` distribution), with the
optional CUDA extension disabled. It records installed versions and the resolved
SAM source commit. This avoids ambiguity with similarly named PyPI packages.

Individual commands, after dependency provisioning, are:

```bash
python3 -u tier1a/prepare_coco500.py
python3 -u tier1a/cache_teachers.py --prior-gpu-hours 0
```

Prefer the combined entrypoint: its GPU account includes installation, acquisition,
extraction, and artifact verification. The standalone teacher command accounts
only its own job; separately include prior acquisition/setup GPU time when used.

## Frozen data and geometry

Selection sorts candidates by image ID, uses Python's private `Random(0)` seeded
shuffle, assigns disjoint strata in person/vehicle/animal order, then sorts the
500 selected rows by image ID. Quotas are 200/150/150. “Person-rich” is explicitly
defined as >=2 non-crowd people. Every image has >=3 total COCO annotations;
vehicle/animal membership requires >=1 non-crowd annotation in that group.
Vehicles are COCO IDs 2–9; animals are IDs 16–25. Images may contain multiple
groups but count toward one quota. Insufficient candidates fail; quotas are never
silently relaxed. Prompts use recorded target-category boxes, descending area then
annotation ID, clipped to the crop, with at most 32 intersecting boxes per image.
Images with no intersecting prompts have a valid empty mask cache, recorded below.

Only `annotations/instances_train2017.json` is extracted/read from the official
train/val ZIP. Its val member stays unopened; no val2017 images are requested.
Only selected train images are downloaded, with three retries after the initial
attempt. The official HTTP endpoints are used as requested:

- `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
- `http://images.cocodataset.org/train2017/<file_name>`

COCO supplies no per-image checksum through these endpoints. On first receipt,
streaming SHA-256 is independently checked against the saved bytes and recorded;
subsequent use checks that receipt and the frozen ledger. This is first-receipt
identity verification, **not upstream authenticity verification**. Existing
unverified or corrupt files fail rather than becoming trusted cache hits.

All 500 ledger entries use `split="train", source="train2017"`. No calibration,
dev, or holdout identities are allocated here. The existing `SplitLedger.freeze`
and `canonical_bytes` define the digest: sorted sample IDs, sorted JSON keys,
compact separators, including the `sequence_id: null` field. The ledger contains
both `frozen_sha256` (existing reader) and its identical `ledger_sha256` alias.
An existing frozen ledger is verified and never rewritten. No real ledger or
selection is fabricated by the local tests or committed in this package.

The existing `resize_inputs` now has an optional center-crop path; its previous
default behavior is unchanged. The shared teacher view is a centered square
resized to 518×518. DINO uses ImageNet normalization and caches normalized CLS
`[768]` and patch `[1369,768]` tokens. SAM's official predictor accepts the same
518 view and **internally resizes to 1024**; altering that model geometry would
invalidate its pretrained architecture. It emits one mask per box, with no
automatic-mask sampling. Masks are bool `[K,518,518]`, scores and floating target
arrays are fp16. SAM's official Base-family 2.1 checkpoint is **B+**, named
`sam2.1_hiera_base_plus.pt`, config `configs/sam2.1/sam2.1_hiera_b+.yaml`.

Official weight URLs (also included verbatim in download/load failures):

- `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth`
- `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt`

`torch.hub` loads `facebookresearch/dinov2`, `dinov2_vitb14`; its verified model
provenance records a hash of the downloaded Python source and checkpoint. Offline
resume uses `/kaggle/working/cache/weights/torch_hub/facebookresearch_dinov2_main`.
Failure messages also identify `https://github.com/facebookresearch/dinov2/zipball/main`.

CONFIG §F forbids raw YUV files in `/kaggle/working`. Consequently the requested
copies are **lossless `.yuv.gz`**, containing one planar 1920×1080 YUV420p8 frame:
BT.709, limited range, 2×2 box chroma averaging, black letterbox, even active
extents/offsets. Each decompressed frame is exactly **3,110,400 bytes**. Geometry
records source dimensions, active extent, offsets, scales, and raw/compressed SHA.
Step 2 must decompress in memory and use the existing codec memory/pipe interface.

## Artifacts

Defaults live under `/kaggle/working/tier1a/`; weight/annotation caches live under
`/kaggle/working/cache/`. Every final artifact has size/SHA in the runner manifest.

| Artifact | Contract |
| --- | --- |
| `coco500_selection.json` | Seed, quotas, annotation SHA, 500 IDs/names/counts/strata and target boxes |
| `coco500_ledger.json` | Portable frozen ledger copy |
| `<repo>/prism/data/splits/coco500_ledger.json` | Authoritative ledger, load with `SplitLedger.read` |
| `coco500/images/<12-digit-id>.jpg` | Verified source images and `.sha256.json` download receipts |
| `coco500/yuv/<id>.yuv.gz` and `.json` | Lossless YUV, source binding, crop-free letterbox geometry and hashes |
| `coco500/manifest.json`, `sources.json` | Per-image encode inputs/geometry and source mapping |
| `teachers/dinov2/<id>.npz`, `.json` | CLS/patch fp16 targets, crop box, SHA and provenance binding |
| `teachers/sam21/<id>.npz`, `.json` | <=32 bool masks, fp16 scores/boxes/RoIs, annotation IDs, source/crop coordinates |
| `teachers/cache/<source-sha>.npz`, `index.json` | Existing `CachedDataset` format: fp16 `x_low`, RoI-aligned `dino_features`, bool `sam_masks`, fp16 `rois` |
| `teachers/model_provenance.json` | Actual teacher checkpoint/source hashes and SAM configuration |
| `teachers/gpu_hours.json` | Persistent GPU usage, CPU usage, job reservations, measured/projection history |
| `dependencies.json`, `install.log` | Installed versions/source and exact pip commands/output |
| `prepare_report.json`, `teachers/cache_report.json` | Per-step status and created/skipped counts |
| `tier1a_step1_report.json` | Overall status, per-step failures, ledger SHA, stats, config/repo provenance, artifact manifest |

Completed artifacts are skipped only after their checksums and bindings verify.
Partial NPZ/JSON writes are regenerated. A changed source, selection, teacher
provenance, or ledger fails without replacing the frozen identity. Preserve the
whole artifact tree and the account between sessions. A crash leaves writer locks;
remove only `.prepare.lock` / `teachers/.writer.lock` after confirming the previous
process has stopped. The next run charges an unsettled job's full reservation.
Never delete/reset `gpu_hours.json` to obtain a fresh budget.

## Budget and Step 2

Initial **unmeasured planning projection** for one T4: 500 × 15 seconds/image ×
1.5 safety factor + 300 seconds setup = **3.21 GPU-hours**. This deliberately
conservative estimate covers teacher caching; acquisition/install are additional
measured wall time. Two allocated T4s are charged twice that wall time, even
though inference uses one device. Actual image timings raise the conservative
forecast if slower after warmup (the first image includes one-time model setup). CPU fallback has
zero GPU charge and remains subject to the same 6-hour job limit; slow measured
throughput can produce an honest budget failure.

Before execution, each job reserves its available allocation within the **60
GPU-hour cumulative pilot ceiling** and **6-hour job limit**, retaining a 60-second
shutdown margin. Forecasts are checked per image; a watchdog terminates before
the allowance expires. Failed jobs still consume budget; interrupted jobs are
charged their entire reservation. Disk checks include all of `/kaggle/working`,
with reserved headroom for acquisition/weights and every new image. Do not run
concurrent jobs in this working directory. No full-pilot throughput claim is made:
Step 2 still requires CONFIG's measured **200-image intra** forecast and a manifest
counting initial oracle, probes, refresh, and evaluation encodes.

Step 2 consumes the frozen ledger, selection, source/YUV manifest, source-bound
teacher index, crop geometry, and cumulative GPU account. Use
`tier1a.yuv.load_yuv(path)` to obtain validated raw bytes in memory. Map masks from
518 crop coordinates into the source, then apply the recorded letterbox transform;
pixels outside the center crop are unobserved, not verified background. Original
RoIs are clipped to the observed crop. `CachedDataset` reads the existing combined
target contract; the per-teacher NPZs retain dense features for other comparisons.

Step 2 must freeze D, lambda, R0 and its CRF grid/protocol, generate 8–16 feasible
maps per train image, and use actual reconstruction teacher evaluation. Original
image caches do **not** substitute for teachers on new reconstructions or for label
refresh when delta-x changes. This package does not open val2017, Tier 1b, K400,
detector evaluation, or a new Student training run.

## Offline validation

No network or teacher weights are used by the local tests:

```bash
python3 -m unittest discover -s tests -p test_tier1a_step1.py -v
python3 -m unittest discover -s tests -p test_stdlib_smoke.py
python3 -m unittest discover -s tests -p test_gate0_stack.py
python3 -m unittest discover -s tests -p test_telemetry.py
python3 -m compileall -q prism scripts tier1a tests
bash -n tier1a/run_step1_kaggle.sh
python3 tier1a/prepare_coco500.py --dry-run-annotations tests/fixtures/coco500_small.json --dry-run-quotas 2 2 2
```

The fixture dry run reports IDs `[1,2,3,4,5,6]` and creates no ledger or cache.
Tests expand its templates into 900 candidates to verify the real 200/150/150
selection quotas. Real COCO acquisition and GPU model inference remain Kaggle
validation work; offline tests cannot establish model throughput or a real ledger SHA.

Local validation (2026-11-01): **60 unittest checks passed** (14 Step 1, 18 GATE 0
stack, 24 telemetry, 4 stdlib smoke). Python compilation, Python 3.10 grammar,
Bash syntax and the six-image fixture dry run passed. NumPy/Pillow ran locally;
PyTorch, real teacher inference and the repository's separate pytest suite were
not available in this offline runtime. No network validation was performed.
