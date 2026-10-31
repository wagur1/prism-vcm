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
| `x265_residual_patch.patch`, `PATCH_AUDIT.md` | Instrument and audit instructions used for the run |
| `runs/<run-id>/` | Immutable run report, pins, logs, audit and `commands.json` with exact argv/environment selections |
| `runs/<run-id>/evidence/` | `gate0.json`, `encode_manifest.json`, per-condition `.hevc`, `.trace.csv`, original x265 `.csv`, normalized `.bits.csv` (encode order), `.job.json`, `.encode.log`, `.decode.log` |

The stock build intentionally has no residual trace. Build/source caches remain
in the repo's ignored `gate0/src/` and `gate0/build/`; they are not copied into
each output run. Build logs and measured pins also exist at `gate0/` as requested.
Repeated builds are incremental; repeated runs use fresh evidence directories.
Do not run two builders against the same checkout simultaneously.

Build failure yields `SKIP` with the exact step and log tail and a nonzero exit.
Encode/effect/identity/decoder failures yield `FAIL`. Neither permits training.
Review [PATCH_AUDIT.md](PATCH_AUDIT.md) after the first real source build: patch
hunks/API names, HM make/GCC compatibility and Kaggle internet access remain
unverified. CQP may disable AQ/offsets; a successful build can correctly FAIL
GATE 0. No PASS is claimed by these files.

Offline checks: `python3 -m unittest discover -s tests -p test_gate0_stack.py`
tests orchestration with mocked codecs/decoders; it is not a native codec test.
See [VALIDATION.md](VALIDATION.md) for the implementation workspace's results.
