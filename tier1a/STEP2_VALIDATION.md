# Step 2 local validation — 2026-11-01

Implementation validation passed offline on Windows, Python 3.12.14, NumPy
2.3.5, Pillow 12.3.0. No network calls, real COCO labels, or detector scores
were produced during validation.

| Check | Result |
| --- | --- |
| `test_oracle_sweep.py` | 21 passed; final run 0.291 seconds |
| `test_tier1a_step1.py` | 14 passed |
| `test_gate0_stack.py` | 18 passed |
| `test_telemetry.py` | 24 passed |
| `test_stdlib_smoke.py` | 4 passed |
| Total selected unittest checks | **81 passed** |
| Python compileall (`prism scripts tier1a tests`) | PASS |
| Python 3.10 grammar (Tier 1a, shared oracle, new tests; 14 files) | PASS |
| Bash syntax (Step 2 launcher, GATE-0 builder) | PASS |
| Entrypoint `--help`, `git diff --check` | PASS |
| Local 2-frame x 3-candidate real-codec smoke | **SKIP: no local GATE-0 binaries/pins** |

The smoke command was executed and wrote
`outputs/oracle_codec_dry_run/codec_dry_run_report.json` with status `SKIP`.
Archived GATE-0 stack reports point to Kaggle paths; they are not local codec
binaries. FFmpeg/real encoding and YOLO/pycocotools scoring were not exercised.
PyTorch, ultralytics, pycocotools, SciPy and pytest are absent in this local
runtime; the repository's separate pytest suite was not run.

The new tests use synthetic 64x64 inputs and a stub encoder. They verify seed
determinism, full 1080p map shape/bounds, teacher-cache hashes and bindings,
crop/box geometry, utility ordering and runner-up, durable success/failure
resume, interrupted/corrupt-output failure handling, train-only restrictions,
decode failure and nonfinite-score handling, no raw YUV writes, scope-reduction
math, and preservation of the existing minimized-distortion oracle.

**500 x 8 wall-clock projection: unmeasured locally.** The kernel measures its
first ten real frames and reports encoder-only and total wall projections with
a 1.25 safety factor, including setup in the scope decision. The K=8 workload
contains 4,000 candidate encodes plus one first-frame stock smoke. If reduced
after the ten-frame pilot, the workload is 10x8 + 490x6 = 3,020 candidates plus
one stock smoke. No stub or 64x64 timings are substituted for 1080p throughput.

Council-visible deviations are documented in `README.md` and emitted in the
kernel reports: AP50/inverse-rate maximization vs inherited CONFIG C; DINO cosine
saliency because attention is absent from Step 1 caches; candidate h = -4/+1;
YOLO reconstruction supervision; conditional K=6; the 10-frame planning pilot
vs CONFIG F's 200-image prerequisite. Winner PNGs satisfy the later no-raw-YUV
rule. No git push was performed.
