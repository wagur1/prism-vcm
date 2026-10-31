# Validation and release gates

GATE 0 and the real frozen source ledger are prerequisites for experiments.
`require_training_ready` rejects missing/failed evidence or a different build.
No data or pretrained weights are downloaded by repository code. The split ledger
template is intentionally unfrozen; real source images were not supplied.

The x265 residual-QP instrumentation/build and two stock HEVC decoders must be
provisioned locally. A stock library without the header-matched shim is not enough.
GATE 0 SKIP never means PASS. If the pinned build ignores offsets under CQP/AQ,
record FAIL and choose the council's written M2-only or delta-x-only fallback;
do not silently change the primary RC mode.

The breaker uses a 64x36 luma difference and phase-correlation motion estimate.
The 0.3 ms cap is a design budget, logged on every frame; this Python reference
implementation is not a measured T4 implementation. Scene-cut threshold 0.35 is
an implementation default (not fixed by CONFIG) to freeze during pilot. The
15 px threshold, 0.7 smoothing, three-frame hold, QP and residual limits are fixed.
Motion fields are backward sampling vectors from the previous encoder frame,
in source luma pixels. Missing motion is logged and uses identity warp.
The measured-rate projection callback must include the applicable frame costs;
ABR/VBV still require actual bitrate and VBV-buffer checks after encode.
`RateMonitor` records whole-stream bitrate and the encoder's actual buffer stats;
an absent buffer observation is reported as unknown, never as conformance.
Upsample the continuous CTU map to QG16 **before** `ControlContract.apply` so its
final projection and rounding are the exact map submitted to x265. Rounding the
coarse student output alone is not a substitute for the final QG-domain check.

BD overlap is pre-registered here as shared task-score span divided by the larger
of the two raw spans (equivalently, >=70% of each curve). CONFIG fixes the 70%
threshold but does not specify its denominator. No extrapolation or isotonic
repair is used for gate decisions. Invalid curves retain the per-QP table and
`[BD-INVALID-RAW]`; JSON contains null, never NaN/Infinity.

PR smoke includes synthetic CPU ST-01 bound/zero-map contracts and ST-10 immediate
scene-cut/motion override over one contiguous sequence. The test suite also checks
ST-02 causal lag. Full nightly Kaggle must run all ten council stress tests;
their complete ST-01..ST-10 definitions are not present in CONFIG v1.3, so no full
stress pass is claimed. ST-03/04 require a documented measurement method or an
explicit exclusion. Run the complete stress suite before release.

Latency must use T4 TensorRT FP16, batch 1, 200 warmup and 3x10,000 synchronized
host-YUV-to-ready-C-API measurements including allocation/copy/resize/forward/
flow/warp/clamp/pack/D2H. Report p50/p99, backend/driver; P100, encoder/lookahead
and FPS separately. TODO [CẦN XÁC MINH]: executable latency gates must be frozen
from the pilot; budget 5/10 ms, target 3.5/6 ms, stretch 3/4.5 ms are not results.

Budget forecasts require a measured 200-image intra run, separate CPU/GPU hours,
<=60 GPU-hours total, <=6 hours/job and <16 GB working storage. Record every
initial oracle, refresh, calibration and evaluation encode in the manifest,
including failed attempts. Freeze D/lambda/R0, splits, budgets and task floors
before accessing test. COCO pilot=500 train2017 images, K400 pilot=100 contiguous
clips; Phase 1b is sealed COCO val2017 5k plus all seven MOT17 validation sequences.

## Local validation, 2026-11-01

- Python compilation: passed for all package, baseline, script and test modules.
- Dependency-free `unittest` smoke: **4 passed** (memory pipe, split freeze and
  seal, training gate, encode manifest/budget).
- Direct NumPy checks: padded QG geometry and residual-block spatial direction
  passed on the local Python 3.12 runtime.
- `scripts/spike_gate0.py`: **SKIP**, no local pinned x265 library/shim.
- `pytest -q`: unavailable on PATH; `python -m pytest -q`: **not run**, the local
  runtimes report `No module named pytest`. Offline dependency resolution also
  confirms pytest is not cached. No network installation was attempted.
- PyTorch/timm are absent. The available Python 3.11 SciPy environment has a
  NumPy DLL access error; scientific/model tests are authored but not claimed as
  passed. Native C compilation is pending the matching x265 header/library.
- Incremental Git commit attempt failed with `.git/index.lock: Permission denied`;
  `.git` is read-only in this session. All implementation files remain reviewable
  in the working tree; no push was attempted.
