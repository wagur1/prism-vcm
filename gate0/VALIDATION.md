# Local validation (2026-11-01)

## esc#11: harness repair and offline QG16 recalibration

Council esc#11 (9/9), as directed by the owner, classifies the checkerboard
failure as a **harness bug**: the verification loop reused the last encoded
condition's `prefix`. It could read checkerboard telemetry while evaluating
plus6/minus6. Both condition and zero telemetry paths now resolve explicitly
from the verified `(qp, gop, condition)`. The regression test checks all 30
condition comparisons in the full matrix and requires the QG verifier for
each of the 10 checkerboard comparisons.

**Historical correction:** the esc#9-era minus6 FAILs were harness artifacts;
the raw data was healthy, per the council's retrospective ruling. Those
archived verdicts are not evidence of a codec sign failure. Their files are
preserved; this correction does not retroactively issue a full GATE 0 PASS.
Partition divergence remains a reason to replace CU-origin pairing, but it
does not explain the reproduced 37.4% harness result (CX-1/CX-3 esc#11).

`verify_net_effect_qg` implements the council's physical-cell comparison:

- Deduplicate by `(frame,qg_x,qg_y,qg_size)`. Every leaf in the effective QG
  must have the same `(final_qp,base_qp_at_cu)`, including inactive leaves;
  otherwise raise. An effective QG is active if any leaf is residual and
  non-skipped. Leaf multiplicity never adds weight.
- Validate that the deduplicated 16/32/64-pixel QGs tile each visible frame
  without gaps or overlaps. Project onto every intersected QG16 cell using
  intersection area. Dimensions are explicit verifier arguments so padding
  cannot silently become visible evidence. At 1920x1080 the visible grid is
  **120x68 QG16 cells**, with an 8-pixel final row; **30x17 is CTU64**, not QG16.
  Out-of-image QGs are excluded, never clamped. The actual checkerboard
  changes sign every **64 pixels** (four QG16 cells per side).
- Cell admission threshold is **1.0 active visible-area coverage in BOTH
  runs**. Partial active cells are excluded, never filled with zero. Geometric
  gaps/overlaps raise. Reports include both-active, zero-only, condition-only,
  neither, joint cell fraction and each run's active area fraction, per
  frame/sign. Per esc#11 ZC-3, less than **0.8 joint cell coverage** in any
  frame/sign is INVALID evidence; it cannot yield a GATE PASS or codec FAIL.
- For each admitted cell, compute `D_r = sum(area*d)/sum(active area)`, with
  `d = final_qp-base_qp_at_cu`, then `oriented = sign(input_map_cell) *
  (D_condition-D_zero)`. Effective-QG averaged `requested_offset` is not the
  sign source. Statistics include signed median, median absolute magnitude,
  type-7 p10/p90, count and fidelity (`oriented > 0`).
- Feedback classes are `first_frame` (frame 0) and `compensated` (frame >=1),
  including GOP1 all-I slices. `i_frames` labels syntax only. Frame 0 requires
  fidelity 1.0; compensated fidelity and aggregate floor are explicit caller
  arguments. Bands are required, with no fallback or auto-pass defaults:
  `{"first_frame": {sign: [lo,hi]}, "compensated": {sign: [lo,hi]}}`.
  Bands apply per frame/sign to median absolute magnitude; aggregate floors
  apply per direction. The legacy per-CU verifier remains available.

The harness accepts `--qg-criteria` (default path
`gate0/criteria_frozen_v3.json`) only with `status: "RATIFIED"`,
`metric: "net-effect-qg16"`, `aggregate_floor_magnitude`, and
`bands_magnitude.checkerboard` containing `first_frame`/`compensated` maps
keyed by `"-1"`/`"1"`, plus `fidelity`. Missing ratification stops a full
spike before encoding. Checkerboard has no per-CU/raw fallback. This work
does not create, edit or ratify any `criteria_frozen*.json`.

`recalibrate_offline.py` reads existing telemetry and job metadata only. It
checks the complete matrix, config/build/dimensions and frame/POC identity,
and uses the same projection/sample functions as the verifier. It writes
`band_calibration_v3_proposal.json`, marked PROPOSAL, council ratification
required, auto-apply false, gate verdict null. Per-group distributions retain
condition, direction, GOP class, nominal QP and feedback class; pooled bands
are separately identified. Compensated proposals are exactly
`[max(p10, frozen direction floor, frozen aggregate floor), p90]`, without
rounding, upper-bound widening or hidden floor relaxation. Conflicts are
reported. Frame 0 preserves AG-2's fixed **[5.5,6.5]** framework.

Recalibration completed on confirmatory #2, run
`20261101T090829Z-54126242`, from the supplied council evidence directory:
40 telemetry files (zero/plus6/minus6/checkerboard, GOP1/GOP10, QP labels
27/32/37/42/47), 100 distribution groups and 500 frame/sign records including
zero diagnostics. All signed comparisons have **100% joint active coverage
and 100% fidelity**, with no coverage problems or inconsistent QG groups.
The five QP labels have identical telemetry hashes within each GOP/condition
(eight distinct telemetry contents total); effective CRF is 20. These are
descriptive repeated CRF data, not independent replications or a CRF sweep.

| Condition | Direction | Frame 0 | Compensated GOP1 | Compensated GOP10 | Compensated pooled |
|---|---:|---|---|---|---|
| plus6 | +1 | [5.5,6.5] | [5,5] | [6,7] | [5,7] |
| minus6 | -1 | [5.5,6.5] | [6,6] | [5,6] | [5,6] |
| checkerboard | +1 | [5.5,6.5] | [5,6] | [5,7] | [5,7] |
| checkerboard | -1 | [5.5,6.5] | [6,7] | [5,7] | [5,7] |

All table values are **PROPOSED**, identical across the five nominal QP
labels. Degenerate empirical intervals are retained for council review.
The aggregate gross floor remains **4.9**. No independent-cell confidence
interval or power claim is made; inference would require cluster bootstrap.

**Freeze protocol v1.6:** council ratifies numbers and freezes the v3
criteria/SHA, then exactly **one OFFLINE confirmatory pass**, hard-stop
afterward. On failure, escalate to the owner; no retuning or reaggregation.
The owner's current offline-only instruction supersedes older council-seat
wording about a new encode. This recalibration is measurement, not that
confirmatory pass. No encoding, network access or git push occurred.

**Validation:** 46 unittest tests passed (24 telemetry/QG, 18 orchestration,
4 stdlib smoke), zero failures/errors. Includes different partitions with
identical physical-cell results, duplicate multiplicity, QP inconsistency,
clipped/padded edges, active/skip coverage, missing/overlapping support,
frame/POC mismatch, feedback class, fidelity, band/floor enforcement and
PROPOSAL rejection. Python syntax, `bash -n gate0/build_gate0_stack.sh` and
`git diff --check` passed. The ML/pytest suites are outside this unittest run.
The projection/recalibration code uses only the standard library; existing
per-CU numerical functions retain their NumPy dependency.

Frozen SHA-256 values remain:

- `criteria_frozen.json`: `e6d6516a174b788dbaf17fabb930c8ed8be48d7843c5b91b3411d04d19b3def4`
- `criteria_frozen_v2.json`: `965600649b83d26d8eae187cc6058fd57dafb6e25dfa35b794d3504a89c9ddd4`

The repository proposal is complete. Copying it to the requested
`Desktop/HOI_DONG_9/escalations/20261101_170803_band_calibration_v3_proposal.json`
was denied by the session's filesystem permissions; only the workspace is
writable and escalation is disabled. Council delivery remains outstanding.

Reproduce offline (no codec/runtime NumPy dependency):

```powershell
& 'C:\Users\Wagur1\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe' gate0/recalibrate_offline.py 'C:\Users\Wagur1\Desktop\HOI_DONG_9\kernels\gate0_out\20261101_165719\gate0_output\runs\20261101T090829Z-54126242\evidence'
```

## Historical deep telemetry v2 (owner-approved workstream)

- **Native build: PASS.** A fresh copy of the council's cached stock x265 3.5
  source received `x265_residual_patch.patch` followed by
  `x265_residual_telemetry.patch` using `git apply --check` / `git apply`.
  CMake configure and full build both exited 0; static `libx265.a` and shared
  `libx265.dll` linked successfully. V2 reverse-check also passed for rebuilds.
- Source archive: council run `20261101_071931`, x265 source HEAD and peeled
  tag 3.5 recorded as `f0c1022b6be121a753ff02853fbe33da71988656`; the archive's
  `x265Version.txt` is stale (`3.4+31-6722fce1f`), which CMake prints. No source
  version metadata was silently rewritten in the final validation tree.
- Toolchain: MATLAB-bundled CMake 3.19.7 / Ninja, MinGW GCC/G++ 6.3.0, Release,
  8-bit, `ENABLE_SHARED=ON`, `ENABLE_CLI=OFF`, `ENABLE_ASSEMBLY=OFF`,
  `ENABLE_LIBNUMA=OFF`, `WINXP_SUPPORT=ON`,
  `CMAKE_CXX_FLAGS=-D__USE_MINGW_ANSI_STDIO=1`. The existing XP option resolves
  this old MinGW's missing Windows 7 `GROUP_AFFINITY` type. These local flags do
  not modify Kaggle build settings. Old MinGW format-checker warnings for
  `%lld` and upstream `usleep` deprecation remain in the log; actual CSV parsing
  and nonconsecutive PTS were checked successfully.
- Final telemetry patch SHA-256:
  `4ad58dc315c7f77ae0294884351dd77718a12058743b420a0c2903d5f253f7ba`.
  Final local shared library SHA-256:
  `509eff1d4ac4e6f2691c77f5390f3a2d396dc7380b623419484f16634917c77f`.
- **Native probes: PASS.** Eight 128x128 ten-frame CRF20 encodes covered
  zero/+6/-6/checkerboard and GOP1/GOP10. All 9,227 residual leaf rows matched
  between v1 and v2 and their reconstructed post-clamp QPs matched final QPs.
  Nonconsecutive PTS 100,110,... matched POC through B-frame reordering. The
  offline analyzer produced 12 condition/direction/GOP-class groups from this
  one nominal-QP fixture; it is not the full production calibration matrix.
- A 256x256 probe confirmed **effective** frame-threads=2 / WPP=1 and 7,540
  matching unique v1/v2 rows. A CRF51/+6 probe flagged all 166 boundary-clipped
  leaves. Telemetry-on/off output was byte-identical. No decoder-QP agreement
  or full GATE 0 PASS is inferred from these checks.
- **Unittest: 28 passed, 0 failures, 0 errors.** 15 orchestration tests,
  9 telemetry tests and 4 existing stdlib smoke tests. Tests cover drift
  cancellation, signed/oriented effects, quantiles, asymmetry, frozen-floor
  enforcement and conflict reporting, proposal markers, duplicate/missing/
  malformed evidence, per-worker env, full collection after FAIL, QUICK skip,
  and preservation of the frozen verdict/exit when calibration fails.
  Orchestration map fixtures remain mocked and dependency-free; native probes
  above provide actual encoder validation. Pytest/ML/GPU suites are outside
  this unittest run.
- **Python syntax, `bash -n`, `git diff --check`: PASS.** Exact source-segment
  comparison confirmed the existing `run_spike()` body unchanged; the entire
  `prism/codec/gate0.py` file is unchanged. Frozen criteria SHA-256 remains
  `e6d6516a174b788dbaf17fabb930c8ed8be48d7843c5b91b3411d04d19b3def4`.
- Local artifacts are retained in ignored `outputs/telemetry-validation/`
  (clean source/patch build, configure/build logs, `checks.json`) and
  `outputs/telemetry-build/` (native probe source, executable, traces, proposal,
  `native_validation.json`). No network request, full Kaggle job, HM/FFmpeg
  decoder validation or push was performed.

Re-run the three unittest modules from the repository root:

```bash
python -m unittest discover -s tests -p test_gate0_stack.py
python -m unittest discover -s tests -p test_telemetry.py
python -m unittest discover -s tests -p test_stdlib_smoke.py
bash -n gate0/build_gate0_stack.sh
```

## Historical v1 first attempt (superseded by the build above)

- 12 dependency-free orchestration tests passed. Encoders, decoders and numeric
  effect evaluation are mocked; tests cover exact quick/full encode counts,
  separate worker environments, duplicate traces, identity/decoder/encode
  failures, missing Bash, exact failed build step/exit propagation, and rejecting
  a full PASS from quick mode.
- Four existing dependency-free smoke tests passed.
- Python compilation and `bash -n gate0/build_gate0_stack.sh` passed.
- `git apply --check` accepted the unified diff against a **synthetic context
  fixture**, not x265. This checks patch syntax and placement mechanics only.
- The actual notebook entrypoint, run locally with an output override, produced
  SKIP/exit 1 with the Bash sandbox diagnostic and log tail. This Windows
  workspace cannot execute the Ubuntu stack.
- Fetching x265 3.5 from Bitbucket failed with a port-443 connection error.
  No native x265/shim/HM build, actual residual-QP audit, decoder conformance,
  numeric codec test or GATE 0 PASS is claimed. The local Python used for these
  checks does not have NumPy/pytest; the orchestration suite uses stdlib unittest.

The remaining source/build verification is specified in [PATCH_AUDIT.md](PATCH_AUDIT.md).
