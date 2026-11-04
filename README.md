# PRISM-VCM

**Preprocessing for Video Coding for Machines** — Offline foundation-teacher distillation → single-shot edge student (ΔQ + bounded Δx) → standard-compliant H.265/VVC bitstream. Zero-trial, single forward, single encode.

> **Binding spec:** [`docs/CONFIG_V1_4.md`](docs/CONFIG_V1_4.md) (amendment §D/§E: CRF primary) trên nền [`docs/CONFIG_V1_3.md`](docs/CONFIG_V1_3.md) — thông qua 9/9 bởi hội đồng 3 bên (Codex / Antigravity / ZCode), 2026-11-01. Council record: [`docs/HOI_DONG_9_KET_QUA.md`](docs/HOI_DONG_9_KET_QUA.md), escalation log: [`docs/EXPLORATION.md`](docs/EXPLORATION.md).

## Kiến trúc (tóm tắt CONFIG v1.3 / v1.4)

- **Student:** MobileNetV4-Conv-Small @256², dual-head — ΔQ map (CTU64 17×30 → QG16 68×120 @1080p) + bounded Δx (|Δx|≤3 reference frames / ≤8 non-reference B).
- **Teacher Distillation:** SAM2 định vị ranh giới hình học + DINOv2 gán độ ưu tiên ngữ nghĩa đối tượng → chưng cất sang MobileNetV4 Controller (208k tham số, 3.5ms trên edge GPU).
- **Codec:** x265 C-API `quantOffsets` per-QG (CRF primary, GOP 32, Low-Delay P / Random Access).
- **Control contract:** causal breaker detection; smoothing (MV trễ 1 frame) → simultaneous bounds/budget projection → round and rate recheck → breaker override.

---

## Tiến độ Thực nghiệm

| Giai đoạn | Trạng thái | Ghi chú & Báo cáo |
|---|---|---|
| CONFIG v1.3 (hội đồng 9/9) | ✅ 2026-11-01 | [`docs/CONFIG_V1_3.md`](docs/CONFIG_V1_3.md) |
| CONFIG v1.4 amendment (§D: CRF primary) | ✅ 2026-11-01 (9/9) | [`docs/CONFIG_V1_4.md`](docs/CONFIG_V1_4.md) |
| GATE 0 codec spike (x265 quantOffsets) | ✅ **PASS 6/6** | `logs/kaggle/gate0/confirmatory_final/` + `gate0/criteria_frozen_v3.json` |
| Tầng 1a Step 3 Pilot (M1–M4 across 5 CRFs & 3 seeds) | ✅ **PASS** | BD-rate $-15.90\%$ YOLO11-L, $-63.31\%$ Faster R-CNN ([Báo cáo Step 3](docs/reports/STEP3_M1_M4_EMPIRICAL_REPORT.md)) |
| Tầng 1b COCO val5k (5,000 ảnh, 19 Kaggle shards) | 🔄 **94.7% (18/19 shards)** | 4,737/5,000 ảnh ([Báo cáo 18 shards](docs/reports/tier1b_val5k_final_report.json)) |
| **Tầng 1b MOT17 Video Tracking (7 chuỗi val)** | ✅ **PASS (7/7 chuỗi)** | **BD-rate HOTA $-23.98\%$** ([Báo cáo MOT17](docs/reports/mot17/merged/mot17_final_report.json)) |

---

## Kết quả Thực nghiệm MOT17 Video Tracking (Journal Track)

Đánh giá trọn vẹn trên 7 chuỗi MOT17 validation (`MOT17-02`, `04`, `05`, `09`, `10`, `11`, `13`) sử dụng **YOLOv11-L + ByteTrack** qua bộ nén x265 Low-Delay P (GOP 32, $\text{CRF} \in \{22, 27, 32, 37\}$).

BD-rate tính theo chuẩn **PAVA-PCHIP (MPEG-VCM)** với khoảng tin cậy 95% qua 2,000 paired bootstrap resamples:

| Biến thể kiến trúc | Bitrate tb (kbps) | HOTA (%) | MOTA (%) | IDF1 (%) | BD-Rate HOTA [%] [95% CI] | BD-Rate MOTA [%] [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Anchor (Stock x265 LDP)** | 2960.5 | 36.82 | 38.09 | 42.02 | — | — |
| **M2: Background Filter Only** | **2948.3** | **37.22** | 38.06 | **42.60** | **-23.98% [-55.14, -7.70]** | **-1.84% [-11.06, +14.81]** |
| **M2: Asym + Rate-Comp** | 2971.0 | 36.95 | 38.07 | 42.18 | **-6.19% [-31.59, +11.38]** | +8.72% [-2.99, +33.50] |
| **M2: Asym + Prefilter (PRISM)** | 2960.0 | 36.87 | 38.04 | 42.18 | +9.64% [-14.24, +33.60] | +8.70% [+1.74, +23.34] |
| **M2: Zero-Mean Spatial QP** | 2973.4 | 36.86 | 37.95 | 41.91 | +11.98% [-30.63, +32.29] | +20.38% [+4.83, +51.87] |
| **M2: Asymmetric [-4, +6]** | 2973.4 | 36.86 | 37.95 | 41.92 | +12.51% [-28.48, +33.01] | +19.74% [+3.71, +51.01] |

### Artifacts Bài báo

- **Đồ thị Rate-Distortion Tracking**:
  - HOTA vs Bitrate: [`docs/reports/mot17/merged/mot17_rd_hota.png`](docs/reports/mot17/merged/mot17_rd_hota.png) ([PDF](docs/reports/mot17/merged/mot17_rd_hota.pdf))
  - MOTA vs Bitrate: [`docs/reports/mot17/merged/mot17_rd_mota.png`](docs/reports/mot17/merged/mot17_rd_mota.png) ([PDF](docs/reports/mot17/merged/mot17_rd_mota.pdf))
  - So sánh BD-Rate: [`docs/reports/mot17/merged/mot17_bdrate_bar.png`](docs/reports/mot17/merged/mot17_bdrate_bar.png) ([PDF](docs/reports/mot17/merged/mot17_bdrate_bar.pdf))
- **Bảng LaTeX cho Manuscript**:
  - Bảng tổng hợp Ablation: [`docs/reports/mot17/merged/table_mot17_ablation.tex`](docs/reports/mot17/merged/table_mot17_ablation.tex)
  - Bảng chi tiết từng chuỗi: [`docs/reports/mot17/merged/table_mot17_per_sequence.tex`](docs/reports/mot17/merged/table_mot17_per_sequence.tex)

---

## Cài đặt & Kiểm thử

Môi trường khuyến nghị Python 3.10+:

```bash
# Cài đặt môi trường phát triển
pip install -e '.[test]'

# Chạy toàn bộ test suite (99 unit tests)
python -m unittest discover -s tests -p "test_*.py"
```

| Thành phần | Đường dẫn thực thi |
|---|---|
| Video Codec Runner (x265 Inter-frame) | `prism/video/video_codec.py` |
| MOT17 Tracking Evaluator (ByteTrack + HOTA) | `prism/video/mot_eval.py` |
| Ablation Variant Generator | `prism/codec/ablation.py` |
| Kaggle Multi-Account Dispatcher | `scripts/dispatch_pool_mot17.py` |
| Tự động thu hoạch & Merge MOT17 | `scripts/auto_harvester_mot17.py`, `scripts/merge_mot17_results.py` |
| Thước đo BD-rate (PAVA-PCHIP & Bjøntegaard) | `prism/evaluation/metrics.py` |
