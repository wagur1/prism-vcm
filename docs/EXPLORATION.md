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

## E-04: CRF compensation depth phụ thuộc GOP — frozen band không transfer (2026-11-01)
- Confirmatory run (frozen criteria e6d6516a): minus6@gop1 compensated median = 4.0 (PASS); minus6@gop10 frame-3 = 3.0 (FAIL floor 3.5). Cơ chế đúng (sign-fidelity 100%); độ sâu bù thay đổi theo cấu trúc GOP/feedback history.
- **Bài học:** band hiệu chuẩn từ MỘT cấu hình GOP không transfer được sang cấu hình khác — đúng cảnh báo transferability của ZC-2 (esc#9). Deep telemetry (per-POC base QP) là con đường duy nhất để band bất biến theo cấu hình.
- Kết quả FINAL theo hard-stop esc#9 — escalated to owner.

## E-05: Deep telemetry workstream — owner approved 2026-11-01

- **Authorization:** owner approved CX-1's telemetry-first workstream from esc#8/#9. Measurement and council proposals only; the criteria hard-lockout remains active. `gate0/criteria_frozen.json` SHA-256 stays `e6d6516a174b788dbaf17fabb930c8ed8be48d7843c5b91b3411d04d19b3def4`.
- **Implementation:** layer instrumentation v2 after v1. Capture float pre-AQ RC base QP, rounded base, weighted SATD/blurred complexity, effective RC/qCompress/VBV, and final leaf-QG requested offsets, slice/CTU base and clipping. Match the completed v1/v2 leaf records by input PTS and pixel origin, separately recording POC and encode order.
- **Analysis:** `gate0/calibrate_bands.py` subtracts each run's own pre-AQ base before comparing with zero, removing common RC-base drift. It emits explicit `PROPOSAL` intervals and direction asymmetry, separated by map condition, GOP class and nominal QP group, retaining existing gross floors. Nothing auto-applies or changes PASS/FAIL.
- **Coverage:** FULL mode completes missing telemetry conditions after the frozen verifier stops, preserving the original result. QUICK records telemetry but skips calibration. The analyzer rejects partial matrices and v1/v2 mismatches.
- **Validation:** CMake/GCC native static/shared build succeeded on the cached 3.5 source; native map/GOP, B-reordering/PTS, clipping, WPP and bit-identity probes passed. See `gate0/PATCH_AUDIT.md` and `gate0/VALIDATION.md` for exact scope.
- **Limits:** current harness QP labels share a single `GATE0_CRF` setting; they must not be presented as a CRF sweep. Full content/grid generalization, independent decoder QP reconstruction, and council ratification remain outstanding. No revised criteria or GATE 0 PASS is claimed.
