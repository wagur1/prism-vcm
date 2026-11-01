# GATE 0 on Kaggle

Enable notebook Internet. Use a public clone URL; never paste a token into the
cell. Ubuntu with Python 3.10+ and passwordless sudo apt is required. One cell:

```python
!bash -c 'set -e; cd /kaggle/working; if [ ! -d prism-vcm/.git ]; then git clone https://github.com/wagur1/prism-vcm.git prism-vcm; fi; cd prism-vcm; python3 -u gate0/run_gate0_kaggle.py'
```

The chair must publish this commit before cloning it there. Existing checkouts
are reused, not pulled or reset. For the two-encode smoke run, replace the final
command with `GATE0_QUICK=1 python3 -u gate0/run_gate0_kaggle.py` (or add `--quick`).
It tests map 0/+6 at QP 32, GOP 10, ten frames each and returns `SMOKE_PASS`, never
a full `PASS`. Full mode runs 60 ten-frame encodes over five QPs and intra/GOP.
Both modes build patched + stock x265 3.5, their shims and HM-16.2, and use
FFmpeg HEVC plus HM. CPU only; at most four build jobs (`GATE0_JOBS` can lower it).

The entrypoint runs `bash gate0/build_gate0_stack.sh`, then the equivalent of:

```bash
PRISM_X265_SHIM="$PWD/gate0/build/shim/libprism_x265.so" \
PRISM_X265_LIBRARY="$PWD/gate0/build/x265/install/lib/libx265.so" \
python3 scripts/spike_gate0.py \
  --output /kaggle/working/gate0_output/runs/<run-id>/evidence \
  --pin gate0/build-pin.json --stock-pin gate0/stock-pin.json \
  --stock-shim "$PWD/gate0/build/stock/shim/libprism_x265.so" \
  --stock-library "$PWD/gate0/build/stock/x265/install/lib/libx265.so" \
  --reference-command-json "[\"$PWD/gate0/build/hm/bin/TAppDecoderStatic\",\"-b\",\"{bitstream}\"]"
```

Each condition uses a fresh worker with its own `PRISM_X265_SHIM`,
`PRISM_X265_LIBRARY` and loader path, including the stock worker. Raw YUV is
generated/passed in memory. Only compressed HEVC is written; FFmpeg sends decoded
frames to its null muxer and HM receives no reconstruction-output option.

Artifacts under `/kaggle/working/gate0_output/`:

| Artifact | Contents |
| --- | --- |
| `gate0.json` | Latest report, status, reason, exit code, run directory and pending audit flag |
| `build_log.txt` | Latest complete builder output, including exact failing step/hunks |
| `build-pin.json`, `stock-pin.json` | Actual library/shim SHA256, version, ABI and picture size; only after successful build |
| `stack.json` | Resolved source commits, patch/decoder hashes and absolute build paths |
| `x265_residual_patch.patch`, `x265_residual_telemetry.patch`, `PATCH_AUDIT.md` | Ordered instrumentation patches and audit used for the run |
| `runs/<run-id>/` | Immutable run report, pins, logs, audit and `commands.json` with exact argv/environment selections |
| `runs/<run-id>/evidence/` | `gate0.json`, `encode_manifest.json`, per-condition `.hevc`, `.trace.csv`, `.telemetry.csv`, original x265 `.csv`, normalized `.bits.csv` (encode order), `.job.json`, `.encode.log`, `.decode.log` |
| `runs/<run-id>/calibration_status.json` | Separate measurement/proposal outcome; never a GATE verdict |

The stock build intentionally has no residual trace. Build/source caches remain
in the repo's ignored `gate0/src/` and `gate0/build/`; they are not copied into
each output run. Build logs and measured pins also exist at `gate0/` as requested.
Repeated builds are incremental; repeated runs use fresh evidence directories.
Do not run two builders against the same checkout simultaneously.

Build failure yields `SKIP` with the exact step and log tail and a nonzero exit.
Encode/effect/identity/decoder failures yield `FAIL`. Neither permits training.
Review [PATCH_AUDIT.md](PATCH_AUDIT.md) for the local native build and its
remaining Linux/decoder validation requirements. CQP disables AQ/offsets; a successful build can correctly FAIL
GATE 0. No PASS is claimed by these files.

Offline checks: `python3 -m unittest discover -s tests -p test_gate0_stack.py`
tests orchestration with mocked codecs/decoders; it is not a native codec test.
See [VALIDATION.md](VALIDATION.md) for the implementation workspace's results.

## Deep telemetry (owner approved 2026-11-01)

The runner defaults `GATE0_RC` to `CRF`, per CONFIG v1.4, while retaining explicit
environment overrides. Existing QP-group labels and CRF selection are preserved:
the current harness uses `GATE0_CRF` (default 20) across its nominal QP groups.
This is **not a five-point CRF sweep**. Proposals retain the actual job settings;
a separately registered grid/content study is needed before council ratification.

Each zero/plus6/minus6/checkerboard worker sets `PRISM_X265_RC_TELEMETRY` to its
own `<prefix>.telemetry.csv` alongside v1 `<prefix>.trace.csv`. Stock and no-map
controls have no v2 trace. The second CSV contains one `record_type=frame` row
per POC plus completed `record_type=cu` leaf rows; see the audit's field table.
The v1 schema is unchanged. Unset/empty telemetry env means no traversal or I/O.

FULL runs retain the frozen spike result, then collect missing map conditions
if its verifier stopped early. This measurement pass records its own
`telemetry_collection.json`; it does not perform or replace the frozen verdict.
Existing/partial artifacts are preserved. Calibration requires all four maps,
all five nominal QP groups, both GOPs, and all ten frames. Missing or inconsistent
evidence produces a separate calibration error. QUICK collects zero/+6 telemetry
but skips matrix completion and calibration.

Offline use, with no codec, NumPy or network required:

```bash
python gate0/calibrate_bands.py /path/to/run/evidence
```

The default output is `gate0/band_calibration_proposal.json` and its
`.net_effects.csv` sidecar. The runner archives these in the run's evidence
directory and maintains local aliases. These generated files are ignored by Git.
Every proposal is marked `PROPOSAL`, `auto_apply: false`, and requires council
ratification. `criteria_frozen.json` stays byte-identical, SHA-256
`e6d6516a174b788dbaf17fabb930c8ed8be48d7843c5b91b3411d04d19b3def4`.

For each shared residual leaf origin the analyzer computes
`(local_offset - base_offset) - (local_zero - base_zero)` using the **float RC
base captured before AQ**, then reports per-frame signed effects and clipping.
Band statistics use `direction * net_effect`, so negative-direction medians are
positive magnitudes. Asymmetry is `plus_median - minus_median`, not separation.
Uniform and checkerboard distributions stay separate. Groups use
`gop1-all-intra`, `gop10-first-frame`, and `gop10-inter` plus nominal QP label.

The proposed interval is `[max(gross_floor, floor(p10)), ceil(p90)]`, with linear
interpolated quantiles. Only lower endpoints and aggregate gross floors are
consumed from the frozen JSON. GOP1 pooling retains the strongest applicable
first-frame floor. If a floor exceeds the proposed upper endpoint, the proposal
records `floor_conflict`; it never silently widens the band. No existing upper
threshold or fidelity rule is read or changed by calibration. Per-CU samples are
not independent QGs/clips; coverage and QG coordinates make that limitation
visible. Neither a proposal nor native compile success opens training.

Additional offline tests: `python -m unittest discover -s tests -p test_telemetry.py`.
