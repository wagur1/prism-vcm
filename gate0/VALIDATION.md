# Local validation (2026-11-01)

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
