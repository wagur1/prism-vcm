# M2 enhancement screening

This experiment adds asymmetric QP bounds, content/CRF-dependent rate
compensation, background smoothing and a SAM/DINO joint oracle proposal.
BD-rate <= -15% is an experimental target, not a result established by this code.
Equal-CRF byte savings alone do not establish savings at equal detector mAP.

## QP projection

`prism.codec.qp_maps.project_adaptive_qp_map` is also exported by
`tier1a.run_tier1a_step3`. Supply the unrounded QG16 student output and
`qg_area_weights(width, height)`. Defaults are `low=-4`, `high=6`, float32;
`high=8` and `quantize_int8=True` are supported. Output is contiguous and keeps
the padded grid shape: **68 x 120 at 1920 x 1080**. Unobserved cells and cells
with zero pixel area receive zero. The partial bottom row at 1080p has half
the weight of a full row.

The projection first centers the spatial field with clipping, then solves for
a positive shift that makes the area-weighted exponential rate proxy
`mean_w(2**(-clip(q + shift)/6))` equal to one. This offsets the convexity
penalty of a nonuniform zero-mean map. It then adds a screening margin scaled
by the positive-QP area fraction. CRF 20/24/28/32/36 margins are
1.00/0.85/0.65/0.25/0.10 QP, with linear interpolation and endpoint clamping.
These defaults were not fitted on val2017. `delta_rate_offset` overrides only
this extra margin; zero still performs content-dependent convex compensation.
Uniform predictions without spatial evidence produce zero by default.

The proxy does not model actual x265 AQ or detector sensitivity. Int8 rounding
can perturb the rate constraint. The suite records dtype, bounds, area mean,
and hashes for review. `infer_student_qp_map(..., projection="adaptive")`
enables this path programmatically; default Step 3/Tier 1b inference retains
the historical zero-mean int8 baseline.

## Background filter

`apply_background_smoothing(image, qg_map, strength=1.0)` accepts an even-sized
luma array, RGB/YUV444 array, or `(Y, U, V)` YUV420 planes. It preserves dtype,
shape, input ownership, and original values at QP <= 0. Positive QP produces
`alpha = clip(strength * clip(QP/6, 0, 1), 0, 1)`, blending the original with
a separable Gaussian (`sigma=1` luma pixel). There is no channel mixing.

The mask expands by exact 16-pixel cells and crops padding; it is not resized
to stretch the final partial row. YUV420 chroma uses the minimum alpha of its
four luma pixels. `mask_kind="saliency"` accepts [0,1] saliency with 1 meaning
foreground. Packed byte buffers must first be unpacked into planes.

The screening prefilter additionally protects the original spatial foreground
if a positive DC offset pushes its final QP above zero. It applies smoothing
directly in YUV, preserving the baseline conversion and letterbox geometry.

## Joint oracle and old labels

`generate_candidates(..., include_fusion=True)` and the corresponding
`cached_candidates` option append index 8, `sam_dino_fusion_minus4_plus6`:

| Observed region | Offset |
| --- | ---: |
| SAM foreground AND top 25% DINO cosine saliency | -4 |
| Other SAM foreground | 0 |
| Outside SAM | +6 |
| Unobserved teacher region | 0 |

Top-quartile selection uses the existing stable row-major tie rule. DINO's
patch-to-CLS cosine is a saliency proxy, not attention. The first eight maps,
legacy K=6/K=8 recipe and hashes remain unchanged. K=9 uses recipe version 2.

Run the existing oracle CLI with `--include-fusion` and a fresh `--output`.
K=9 retains the fusion candidate under deadline accounting; it does not
silently reduce to K=6. Step 3 recognizes K=9 labels and allows the fusion map
to win spatial supervision, while preserving the old selection for old labels.
Teacher cache checksum/source binding checks still apply.

## Run screening

Fast offline wiring check (requires NumPy; no models or codec):

```sh
python scripts/screen_m2_enhancements.py --mock --limit 10 --width 1920 --height 1080 --output outputs/m2_mock
python -m unittest tests.test_m2_enhancements -v
```

Mock uses synthetic joint-teacher maps, scalar quantization and zlib. Its
reported bytes are mock payload bytes, **not HEVC bitrate**. It leaves mAP
null/N/A and records pixel similarity separately; no detector accuracy is
fabricated. It exercises all four variants at all five CRFs and zero-map/stock
identity, including shape and accounting checks.

Real dev screening, using an existing Step 3 checkpoint and provisioned builds:

```sh
python scripts/screen_m2_enhancements.py \
  --images /data/train2017 \
  --ledger prism/data/splits/coco500_tier1a_splits.json --split dev --limit 50 \
  --checkpoint /models/student_seed_0.pt --device cuda:0 \
  --gate /build/gate0 --ffmpeg ffmpeg \
  --high 8 --strength 1.0 \
  --detector yolo --annotations /data/annotations/instances_train2017.json \
  --weights /models/yolo11l.pt --output outputs/m2_dev_high8
```

`--split pilot` is also supported. Both use frozen train2017 ledger identities
and verify source SHA256 before encoding. Sample selection is deterministic
from `--seed`; all variants share the same images and CRFs. Use a fresh output
directory for each run. `--detector none` measures bytes/decode only. Real
execution never falls back silently to mock or downloads missing models.

For replay without PyTorch, replace `--checkpoint` with `--maps-dir`. Each
`<sample_id>.npz` must contain:

- `raw_qg_map`: unrounded maps of shape `[C, Hq, Wq]` (2D only for one CRF).
- `crfs`: unique vector of C CRFs, containing every requested CRF.
- `input_yuv_sha256`: scalar string hashing the exact letterboxed YUV420 bytes.
- Optional `known_mask`: `[Hq, Wq]` teacher-observed support.

The input hash can be obtained from `tier1a.yuv.letterbox_yuv` geometry's
`raw_sha256`. A shape or source/conversion mismatch fails rather than reusing
an unrelated map. Checkpoint mode uses raw student output independently at
each CRF. `--int8` quantizes both adaptive variants for an all-int8 comparison;
the baseline always reproduces the historical [-6,+6] int8 projection.

`report.json` contains input/map/code hashes, configuration, source IDs,
per-frame whole-stream bytes, paired per-CRF totals and percent versus M1.
Real YOLO runs score every decoded variant and compute pooled standard COCO
AP@[.50:.95], AP50 and size metrics. They do not average per-image AP as mAP.
The first image also checks patched zero-map bit identity against stock at
each CRF. Temporary codec files are removed after each image; raw YUV stays
in memory. Failures are saved with `status=FAIL`, not accepted as partial wins.

Select settings on dev, then evaluate a fixed configuration on the full
five-CRF grid before making a BD-rate claim. This suite does not automatically
run the sealed val2017 evaluation or modify its previous results.
