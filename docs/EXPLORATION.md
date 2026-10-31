# EXPLORATION RECORD — PRISM-VCM (dead-ends & negative results, per ARA D5)

Ghi nhận trung thực các kết quả âm — mỗi mục là một bài học không được phép quên.

## E-01: quantOffsets CHẾT dưới CQP (2026-11-01, kernel quick v5)
- **Phát hiện:** bitstream zero-map và +6-map **byte-identical** (8,935,397 bytes; mọi per-frame bits trùng khớp); ΔQP = 0.000 trên 240,000+ shared residual CU (10 frame, content iid-noise).
- **Root cause (xác minh bằng source x265 3.5, commit f0c1022b):** `Encoder::configure()` (encoder.cpp) ép `aqMode=NONE, cuTree=0, aqStrength=0` cho **mọi** encoder `X265_RC_CQP`; `x265_picture.quantOffsets` chỉ được tiêu thụ qua đường AQ (`calcAdaptiveQuantFrame` → `calculateQpforCuSize`, slicetype.cpp/analysis.cpp) → dưới CQP, offsets bị bỏ rơi hoàn toàn. Đổi riêng `--aq-mode` không cứu được CQP.
- **Bài học:** GATE 0 đã làm đúng nhiệm vụ — phát hiện cơ chế chết TRƯỚC khi bất kỳ con số BD-rate nào được tin. Nếu không có spike này, toàn bộ nhánh ΔQ sẽ chạy trên một kênh điều khiển vô hình.
- **Điều chỉnh:** chuyển cấu hình claim sang CRF (xem CONFIG amendment; esc#5 verdict + esc#6). CRF=20 + aq-mode=1 + cutree=0: offsets được cộng vào `qpAqOffset` sau AQ, không nhân strength.

## E-02: Verifier đòi per-frame residual vs P-frame skip-dominant (2026-11-01, quick v4)
- Trace 26,095 dòng hoạt động thật, nhưng 5/10 P-frame 0 khối residual vì `synthetic_frames` roll 1 texture 2px/frame → ME dự đoán gần hết.
- **Bài học:** phép thử và nội dung kiểm thử phải được thiết kế CÙNG NHAU; content phải bảo đảm điều kiện mà verifier đòi hỏi (iid noise per frame, seed cố định, content-sha provenance).

## E-03: Chuỗi hạ tầng (tổng hợp nhanh)
- `m_depth` vs `m_cuDepth` (x265 3.5 member name) — patch C++ phải đối chiếu source thật, không viết từ trí nhớ.
- HM-16.2 makefile dùng biến `CPP` (không phải `CXX`) — build system cũ cần audit biến trước khi wrapper có tác dụng.
- `-Werror` của HM + GCC hiện đại: `-Wno-error` đặt sau không override được các `-Werror=<type>` — phải lọc token khỏi argv.
- Kaggle worker có thể thiếu NASM — ASM decision phải probe + fallback OFF có ghi provenance (F0-scope only).

## Trạng thái hiện tại
- GATE 0 mechanism: **CHỨNG MINH được dưới CRF** (quick v6: 202,791/202,791 CU ΔQP>0; I-frame ΔQP = 6.00 chính xác; bits −22.2%; zero-map vẫn byte-identical với stock).
- Còn lại: verifier tier-2 threshold (esc#6) + biểu quyết CONFIG §D (9/9) → kernel FULL.
