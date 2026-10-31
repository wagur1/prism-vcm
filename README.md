# PRISM-VCM

**Preprocessing for Video Coding for Machines** — Offline foundation-teacher distillation → single-shot edge student (ΔQ + bounded Δx) → standard-compliant H.265/VVC bitstream. Zero-trial, single forward, single encode.

> **Binding spec:** [`docs/CONFIG_V1_3.md`](docs/CONFIG_V1_3.md) — approved 9/9 by the tri-party council (Codex / Antigravity / ZCode), 2026-11-01. Council record: [`docs/HOI_DONG_9_KET_QUA.md`](docs/HOI_DONG_9_KET_QUA.md).

## Kiến trúc (tóm tắt CONFIG v1.3)

- **Student:** MobileNetV4-Conv-Small @256², dual-head — ΔQ map (CTU64 17×30 → QG16 68×120 @1080p) + bounded Δx (|Δx|≤3 reference frames / ≤8 non-reference B).
- **ΔQ training:** supervised-from-oracle-sweeps (K=8–16, J=D+λR/R₀); proxy codec chỉ fine-tune sau calibration (MAPE ≤5%, Spearman ≥0.8).
- **Codec:** x265 C-API `quantOffsets` per-QG (CQP [27–47] primary); VVC/VTM = extended Phase 3.
- **Control contract:** causal breaker detection; smoothing (MV trễ 1 frame) → simultaneous bounds/budget projection → round and rate recheck → breaker override (both heads zero, held ≥3 frames).

## Tiến độ

| Giai đoạn | Trạng thái | Ghi chú |
|---|---|---|
| CONFIG v1.3 (hội đồng 9/9) | ✅ 2026-11-01 | `docs/CONFIG_V1_3.md` |
| Repo skeleton + Student + tests | ✅ Implemented; validation limited by local dependencies | §A–F, H.2; see validation notes below |
| GATE 0 codec spike (x265 quantOffsets) | ⏸ Harness implemented; local run SKIP | Pinned libx265/shim and residual-QP trace unavailable; training locked |
| Frozen split SHA ledger | ⏸ Guard implemented; real data pending | No fabricated hashes; val2017 remains sealed |
| Tầng 1a pilot (COCO500 + K400-100, Kaggle pool) | ⏳ | §F |
| Tầng 1b (COCO5k + MOT17 đủ 7 chuỗi) | ⏳ | sau Gate 1a |

*Kaggle run logs sẽ được tải về `logs/kaggle/` và push sau mỗi mốc.*

## Implementation

Python 3.10+. Core install: `pip install -e '.[test]'`; add `.[model,cache]` on a
provisioned machine for PyTorch/timm/LMDB. Repository scripts do not download
datasets or teacher weights. Use `pytest -q` for the full suite. Torch and native
codec tests skip when their dependencies are absent.

| Area | Entry points |
|---|---|
| Student / proxy / losses | `prism/models/`, `prism/losses/` |
| Causal control contract | `prism/control/contract.py` |
| Header-matched x265 C API / VTM stub | `prism/codec/`, [native build notes](prism/codec/native/README.md) |
| Ten-frame CQP GATE 0 | `scripts/spike_gate0.py`, `tests/test_qp_map_effect.py` |
| Oracle / calibration / encode accounting | `prism/training/` |
| Frozen split ledger / NPZ and LMDB cache | `prism/data/`, `scripts/cache_teacher_targets.py` |
| Raw PCHIP / cluster bootstrap / Holm | `prism/evaluation/` |
| M1–M4 / experiment matrices | `baselines/`, `configs/`, `prism/codec/configs/` |
| CPU smoke CI | `.github/workflows/smoke.yml` |

The requested model skeleton is present, but **training is locked** until GATE 0
passes on the pinned build and real source SHA lists are frozen. This differs
from CONFIG's literal pre-model sequencing because the task requests the full
skeleton without local codec evidence or source data. No experiment is unlocked
by the skeleton. The split template cannot act as a frozen ledger.

See [validation details](docs/VALIDATION.md) for the remaining native trace hook,
teacher adapters, T4 timing TODOs, nightly stress definitions, and overlap
denominator. D/lambda/R0 and task floors remain unset until protocol freeze;
latency numbers are design budgets, not measurements. Statistical helpers do not
turn descriptive curves into an H1 claim without independent-seed/cluster evidence.

Local validation (2026-11-01): GATE 0 returned **SKIP** because the pinned x265
library/shim is absent. Python compilation and **4 dependency-free smoke tests
passed**; **pytest could not run because it is not installed or cached offline**.
Details are recorded in `docs/VALIDATION.md`. No network
operations or pushes were performed. Git commits could not be created because
this session mounts `.git` read-only (`index.lock: Permission denied`).
