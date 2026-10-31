# BIÊN BẢN HỘI ĐỒNG 9 GHẾ — KẾT QUẢ CUỐI (PRISM-VCM CONFIG v1.3 THÔNG QUA 9/9)

* **Ngày:** 01/11/2026 • **Nơi lưu trữ:** `C:\Users\Wagur1\Desktop\HOI_DONG_9\`
* **Cơ cấu:** 3 bên × 3 ghế — Bên Codex (gpt-6-astra, effort high qua `codex exec`): CX-1 Codec/RC, CX-2 Proxy Math, CX-3 Systems. Bên Antigravity (gemini-3.8-flash-high qua `agy`): AG-1 Foundations, AG-2 Benchmark, AG-3 Novelty. Bên ZCode (GLM): ZC-1 Standards, ZC-2 Auditor (ARA Rigor), ZC-3 Stress Critic. Chủ tọa: ZCode.

## 1. QUY TRÌNH ĐÃ CHẠY

| Vòng | Nội dung | Kết quả |
|---|---|---|
| Chuẩn bị | INPUT PACK: 7 tài liệu nguồn (S1–S7) + 10 tranh chấp D1–D10 + ràng buộc hạ tầng Kaggle | `00_INPUT_PACK.md` |
| Vòng 1 | 9 hồ sơ lập trường độc lập — cấm đồng ý nguyên trạng, bắt buộc phản biện ≥2 điều khoản | `v1_positions/` (9 file) |
| Vòng 2 | Phản biện chéo: mỗi ghế bóc lỗi 8 ghế kia, nhượng bộ có lý do, tuyên bố make-or-break | `v2_cross/` (9 file) |
| Vòng 3 | CONFIG v1.0 → **v1.1** (6/9) → **v1.2** (7/9, CX-3 lỗi hạ tầng) → **v1.3** | **9/9 ĐỒNG Ý** |

## 2. HAI LẦN BỊ BẮT LỖI BỞI CHÍNH QUY TRÌNH (bằng chứng deliberation không hình thức)

1. **v1.1:** §J tuyên bố "chỉ bổ sung" nhưng thực tế nén mất 11 cụm điều khoản v1.0 (PCHIP, toàn-bộ-bytes, cùng GOP/công cụ, "trước mọi code model", lịch nightly, Gate 1a→1b, mAP metric, cap 0.3ms...). CX-1 + CX-3 đối chiếu nguyên văn và bỏ phiếu KHÔNG → v1.2 khôi phục.
2. **v1.2:** CX-2 bắt tiếp 4 lược bỏ ("≥1.000 resamples", seeds "độc lập", loss L_align/Huber-oracle, định nghĩa M1–M4) → v1.3 khôi phục + §J'' khai báo trung thực toàn bộ lịch sử.

## 3. NHỮNG ĐIỀM HỘI ĐỒNG ĐÃ ĐẢO NGƯỢC SO VỚI CÁC BIÊN BẢN TRƯỚC (S1–S7)

- **Bỏ STE làm phương án chính** cho nhánh ΔQ → supervised-from-oracle-sweeps (K=8–16, J=D+λR/R₀) + proxy chỉ fine-tune sau calibration (MAPE ≤5%, Spearman ≥0.8, n≥100/cell).
- **Bác `--dqpfile` VTM** (cờ không tồn tại) + **bác "Cách B" analysis-reuse** → chỉ x265 C-API `quantOffsets` per-QG (68×120 @ pad 1088); VVC = extended Phase 3.
- **Bác zero-sum ΣQ của S4** (Σ=0 vẫn +25% bits) → ΣΔbits≤0 một phía, chỉ ABR/VBV, tắt CQP.
- **Bác grid 16×9** → 17×30 CTU64 / 68×120 QG16.
- **Cách ly số liệu ảo** của S2/S7 (18.5%, +0.4pp, 1.4/2.8ms, 80–85%...) — hạ "design budget/kỳ vọng".
- **Hạ teacher** DINOv2-Giant → **Base + SAM 2.1-Base** (≤60 GPU-hour).
- **Claim mới:** "foundation-generalist" (sau ARPS), "no out-of-band side information", H1/H2/H3 pre-registered.
- **Mở mới:** 2 lỗi toán học trong S5 do hội đồng phát hiện (BCE logit bão hòa 0.269–0.731; ΣQ không bảo đảm bits); breaker hai tầng; nguồn motion tách warp-MV / breaker-frame-diff.

## 4. VĂN BẢN RÀNG BUỘC

**`CONFIG_V1_3.md`** là đặc tả kỹ thuật duy nhất có hiệu lực cho giai đoạn code, thay thế S5. Các file phục vụ tái lập: `00_INPUT_PACK.md`, `v1_positions/`, `v2_cross/`, `v3_vote*/`, `S6_PHIEN_2_HOIDONG_DOC_LAP.md`.

## 5. GIAI ĐOẠN TIẾP THEO

1. **Codex hiện thực + push** `github.com/wagur1/prism-vcm` theo CONFIG v1.3, GATE 0 spike trước mọi code model.
2. **Kaggle pool** (20 tài khoản, token ngoài repo): Tầng 1a pilot (COCO500 + K400-100) → Gate → 1b (COCO5k + MOT17 đủ 7 chuỗi).
3. **ZCode/agy cron:** theo dõi job, tải log push repo, cập nhật README theo mốc.

*Chủ tọa ký: ZCode — 9/9 phiếu lưu tại `v3_vote4/`.*
