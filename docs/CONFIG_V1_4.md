# CONFIG v1.4 — AMENDMENT §D/§E (CRF PRIMARY) — BẢN BIỂU QUYẾT TOÀN THỂ 9/9
*Căn cứ: esc#5 (H1 xác nhận — CQP ép aqMode=NONE, quantOffsets chết) + esc#6 (CRF discriminator thành công: 202,791/202,791 CU ΔQP>0; I-frame 6.00; bits −25.11% toàn bitstream; zero-map identity giữ). Mọi điều khoản khác của CONFIG v1.3 giữ nguyên hiệu lực. Đây là amendment có trọng — biểu quyết 9/9 bắt buộc.*

---

## §D (SỬA ĐỔI) — GIAO THỨC THỰC NGHIỆM

- **Cấu hình claim chính cho nhánh ΔQ: x265 CRF** (thay CQP — vốn bất khả thi cho quantOffsets, đã chứng minh ở E-01 `docs/EXPLORATION.md`).
  - Cờ đóng băng: `crf=<grid>`, `aq-mode=1`, `aq-strength=1.0`, `hevc-aq=0`, `cutree=0`, `sao` theo build pin, `qg-size=16`, bframes/GOP theo điều kiện test.
  - **CRF grid: 5 điểm, chốt trên dev-split trước khi chạm test** (đề xuất khởi tạo {20, 24, 28, 32, 36} — grid cuối được freeze trong `configs/` kèm rationale; không được chọn sau khi thấy kết quả test).
  - `cutree=1` là extended ablation riêng (đã ghi tradeoff: bỏ temporal allocation có thể giảm hiệu suất — kết luận giới hạn trong cấu hình test).
  - CQP chỉ trở lại như extended study nếu patch đường DQP/AQ (non-blocking, không cam kết).
- **BD-rate axis không đổi:** PCHIP log-rate theo task-score, toàn bytes, miền giao ≥70%, cấm ngoại suy; phi đơn điệu → per-QP + [BD-INVALID-RAW]; isotonic chỉ sensitivity. **Lưu ý CRF:** CRF chọn điểm chất lượng — cấm coi "giảm bits tại cùng CRF" là cải thiện RD; mọi claim BD-rate vẫn theo trục rate @ equal task performance.
- **Zero-sum ΣΔbits≤0:** giữ nguyên (chỉ ABR/VBV; tắt ở CRF/CQP).
- Baselines M1–M4 chạy cùng CRF grid, cùng build pin, cùng content/protocol.

## §E-F0 (BỔ SUNG) — GATE 0 CRF-verifier criteria (theo esc#6, AG-2 phê chuẩn)

1. **Sign-fidelity 100% = HARD GATE:** trên tập shared residual CU, mọi ΔQP > 0 cho vùng offset dương (0 CU ≤0 → FAIL ngay).
2. **I-frame:** median ΔQP ∈ [5.5, 6.0] (inject 1:1 nguyên vẹn).
3. **P-frame per-frame:** median ΔQP ∈ [2.5, 4.5] (CRF base-compensation band; sàn 2.5 bắt suy hao >50%, trần 4.5 xác nhận compensation bình thường).
4. **Aggregate:** median ΔQP ≥ 3.00.
5. **Bits:** whole-bitstream thay đổi ≥5% cho map ±6 toàn khung.
6. **Zero-map identity:** byte-identical 1:1 với stock — SHA-256 match.
7. Full GATE 0 vẫn yêu cầu: minus6/checkerboard conditions, GOP dài, hai decoder độc lập (FFmpeg + HM TAppDecoderStatic), pin đầy đủ.

## §E-F1 (BỔ SUNG) — PIN mở rộng cho CRF
Ghim và log sau `encoder_parameters`: crf value, aq-mode/aq-strength/hevc-aq hiệu lực, cutree, qCompress, psy, GOP/lookahead, threads/WPP, QG/SAO, VBV-off, commit/SHA/ABI/shim/patch, compiler/ASM, seed/content-sha. Metadata cũ ghi "CQP" phải được cập nhật theo cấu hình hiệu lực.

## TRACEABILITY
- AG-2 (esc#6): criterion amendment + PASS vote. CX-1 (esc#6): physics + YES vote với caveats (đã nhập vào §D/§E-F1).
- Lịch sử: v1.3 §D ghi "CQP [27,32,37,42,47] claim chính" — BỊ THAY THỊ bởi mục này; v1.3 §E-F0 gate vẫn giữ (chỉ điều kiện verify mở rộng như trên).
- Bằng chứng: `kernels/gate0_out/20261101_043917` (quick v6), `escalations/20261101_044524_*`, `docs/EXPLORATION.md` E-01.
