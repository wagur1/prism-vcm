# Local validation (2026-11-01)

## Deep telemetry v2 (owner-approved workstream)

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
