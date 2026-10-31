# CONFIG v1.3 — HỢP ĐỒNG KỸ THUẬT PRISM-VCM (BẢN BIỂU QUYẾT CUỐI, MỤC TIÊU 9/9)
*Lịch sử: v1.0 (6/9 YES) → v1.1 (bổ sung điều kiện CX; nhưng §J sai khi tuyên bố "chỉ bổ sung" — thực tế đã nén mất một số điều khoản v1.0; CX-1/CX-3 bắt đúng lỗi này) → **v1.2: KHÔI PHỤC nguyên văn mọi điều khoản v1.0 + bổ sung đủ điều kiện của CX-1/CX-2/CX-3 + quy tắc ưu tiên**. §J' ghi chính xác mọi khác biệt so với v1.0.*

---

## A. KIẾN TRÚC & TENSOR CONTRACT

- Student: MobileNetV4-Conv-Small, input `x_low [B,3,256,256]` + `qp_scalar` (chuẩn hóa QP/51), conditioning MLP 2 lớp vào bottleneck (theo S5, không đổi).
- Head 1 ΔQ: đầu ra lưới **CTU64 = 17×30 @1080p** (bác 16×9 của S2/S7), nội suy lên **QG16 = 68×120 trên miền pad 1920×1088**, ánh xạ theo tọa độ, xử lý mép `[MB-CX-1.1, MB-CX-3.1, MB-AG-3 (D3)]`.
- Head 2 Δx: residual luma bounded, |Δx| theo frame-type (mục B).
- Teacher: **DINOv2-Base (86M) + SAM 2.1-Base** (phân xử §I.5). Input teacher **bội của 14** (518×518) + RoI-Align khi sample target `[MB-AG-1.1]` — chỉ áp pipeline extraction offline, không đụng input 256² của Student.
- Codec claim chính: **x265 C-API `quantOffsets` (Cách A duy nhất; bác Cách B analysis-reuse của S5)**; VTM/VVC = extended results Phase 3 (mục H).

## B. HỢP ĐỒNG ĐIỀU KHIỂN CODEC

**B.0 Breaker hai tầng:**
- *Phát hiện* (zero-latency, nhân quả từ ảnh đầu vào): frame-difference 64×36 luma **(cap chi phí ≤0.3 ms — design budget `[ZC-2]`)** + scene-cut detector; V>15px/frame quy đổi 1080p hoặc scene-cut → kích hoạt.
- *Override:* **kích hoạt → đặt cả hai head ΔQ=Δx=0 NGAY, GIỮ =0 trong ≥3 frame (kể cả scene-cut áp cùng override), + flush M + reset lịch sử; đạt 0 trong ≤2 frame (thỏa ZC-3)** `[MB-CX-1.3, MB-CX-2.2, MB-CX-3.1, MB-ZC-3.1]`. ST-10 là trọng tài nếu phát hiện vệt chất lượng.

**B.1 Chuỗi áp dụng tín hiệu (nguyên văn CX-3, CÓ HIỆU LỰC THAY THẾ nội dung mâu thuẫn):** "Smoothing→chiếu đồng thời bounds/ngân sách→kiểm lại sau lượng tử hóa→breaker(override). |ΔQ|≤6, nền≤+4; **QP hợp lệ**; |Δx|≤3 với frame tham chiếu/chưa biết loại, ≤8 với B không tham chiếu. **Breaker đặt cả hai nhánh về 0 và reset lịch sử. Zero-sum không bảo đảm bits.** Nguồn motion phải sẵn trước encode, đo cả chi phí và độ trễ." Smoothing: M_t = α·Warp(M_{t−1}, MV encoder frame trước) + (1−α)·M_t_raw, α=0.7; MV trễ 1 frame, nhân quả, ghi ST-02 `[MB-ZC-3.2]`.

**B.2 Zero-sum (nguyên văn CX-1 + ZC-2):** "Zero-sum chỉ là ablation; công bố sai khác giữa anchor cùng binary/map-0 và stock. VBV phải kiểm rate và buffer thực." Công thức: **ΣΔbits≤0 một phía — CHỈ ABR/VBV, TẮT ở CQP**; projection = dịch hằng số toàn khung; kiểm **sau round() ở miền rate đo được**; vô nghiệm → co δ_fg trước, **KHÔNG được phá floor như phương án sau cùng trong B.1** (co δ_fg ưu tiên tuyệt đối; nếu vẫn vô nghiệm → breaker override về 0); log **tỷ lệ chạm biên/bypass** mỗi frame `[MB-CX-2.2]`. Khóa `--aq-mode`/`--cutree`/`--sao`/`--qg-size`/preset cho anchor và patched.

## C. HUẤN LUYỆN

- ΔQ: supervised-from-oracle-sweeps CHÍNH; loss rời rạc dùng **L_align/Huber-towards-QP-oracle** (SAM masks chỉ là đề xuất phụ `[CX-2/ZC-3]`). **"Oracle gồm K=8–16 map khả thi/frame, chọn theo J=D+λR/R₀; khóa D, λ, R₀. Nhãn chỉ từ train; calibration tách train/test. Làm mới nhãn khi Δx đổi; cache target không thay thế teacher trên reconstruction mới."** `[MB-CX-2.1]`
- **Định nghĩa mô hình ablation:** M1 = Anchor codec stock; M2 = ΔQ-only; M3 = Δx-only; M4 = Hybrid (ΔQ+Δx) `[khôi phục v1.0/S2]`.
- Δx: loss chưng cất ngữ nghĩa trên biến liên tục (x_raw+Δx) trước codec `[MB-AG-1.2]`; proxy_codec STE bị cấm định tuyến gradient rời rạc về backbone.
- **"Proxy chỉ fine-tune sau calibration rate và semantic. Hướng cập nhật Δx phải được thử trên codec thật; video đo cả frame phụ thuộc."** `[MB-CX-2-v1.1]`
- **"Oracle chỉ dùng train; calibration/dev/test tách biệt."** `[MB-CX-3.3]`
- Calibration mỗi 1.000 bước: từng QP/frame-type MAPE bits ≤5% + Spearman ΔJ ≥0.8 (probe ±1QP khả thi, cùng trạng thái codec; n≥100/cell + Fisher-z CI), kiểm cả semantic; **cận trên CI95 của ΔJ trung bình <0**. **"Hai lần trượt liên tiếp: khôi phục checkpoint giám sát, tắt fine-tune; Δx chưa chứng thực đặt 0."** `[MB-CX-1.4, MB-CX-2.2, MB-CX-3.4]`
- Định nghĩa "lệch hướng" (F5) = dấu(ΔJ thật) ≠ dấu(ΔJ proxy) trên ≥2 QP liên tiếp cùng frame-type.

## D. GIAO THỨC THỰC NGHIỆM

- RC: **CQP [27,32,37,42,47] là cấu hình claim chính, CHỈ dùng sau khi spike xác nhận offsets có hiệu lực dưới CQP/AQ của build đang khóa.** "Nếu cần patch, báo anchor cùng binary/map 0 và stock; công bố sai khác. CRF và ABR/VBV báo riêng, cùng preset/GOP/công cụ." `[MB-CX-1.2]` Zero-delta **bit-identity** (patch ΔQ≡0 → bitstream trùng stock) là precondition mọi benchmark trên encoder patched `[MB-ZC-1.1]`.
- **"BD-rate dùng PCHIP log-rate theo task-score trên miền chung, toàn bộ bytes bitstream, không ngoại suy; cùng GOP/công cụ; phi đơn điệu báo dữ liệu thô, không dùng isotonic để quyết định pass."** Overlap **≥70% (một ngưỡng duy nhất)**; phi đơn điệu → **bảng per-QP + cờ [BD-INVALID-RAW]** (không NaN/∞); isotonic-PAVA chỉ ở **Phụ lục Sensitivity**, không quyết định gate `[MB-CX-1.5, phân xử §I.1]`.
- **"M4 phải đạt BD≤−3% riêng so M2 và M3, cận trên CI95% dưới 0; ≥3 seeds ĐỘC LẬP, cluster bootstrap 95% ≥1.000 resamples, ghép cặp theo ảnh/chuỗi, đơn vị bootstrap ghi cạnh số liệu; Holm–Bonferroni trên họ M4-vs-{M2,M3}×5QP; sàn per-task cho H1."** `[MB-CX-1.5, MB-CX-2.5, MB-CX-3.4, MB-AG-1.4, MB-AG-2.2, MB-ZC-2.1]`
- **H1 (coupling)** như trên. **H2 (pre-registered, phân tích thứ cấp): "H2: tiết kiệm = −BD; −BD∈[1.0;2.9)% và AP_small≥+0.5pp" — công thức §D-H2 là văn bản chính thức, THAY công thức dấu cũ tại §I.8; H2 không thay thế luận điểm Title/Abstract; <1.0% → fallback M2** `[MB-CX-1.5-v1.2, §I.8]`. **H3 zero-shot ≥12%: giả thuyết, không gate.**
- Transfer R3D-18: ≤+2% pass; >+5% fail; (2,5]% chưa kết luận (K400-100 clips).

## E. BẢNG FALSIFICATION F0–F7 (MỘT BẢNG DUY NHẤT)

- **F0 GATE 0 Codec Spike — TRƯỚC MỌI CODE MODEL:** pin x265 version/binary SHA/ABI; shim C dùng đúng header, float32 row-major, buffer sống tới khi encoder sao chép `[MB-CX-3.1]`; spike map 0/±6/checkerboard qua intra+GOP **trong đúng RC mode CQP**; kiểm **trace QP thực tế tại khối CÓ residual** (loại CU skip/no-residual khỏi tiêu chí) + bytes + **decode độc lập đúng codec**; miền hợp lệ 30×17 loại hàng pad; **cấm CSV QP trung bình làm bằng chứng vị trí**; instrument đo realized-QP nêu tên cụ thể `[MB-ZC-1.3, MB-ZC-2.5, MB-CX-1.1, MB-CX-2.3]`. Δbits≥15% chỉ là chỉ báo, không phải bằng chứng duy nhất.
- **F1 Latency — nguyên văn CX-3:** "T4 TensorRT FP16, batch=1, YUV420p8 1080p, Student 256²; warm-up 200, 3×10.000 mẫu. Đo wall-clock đồng bộ CUDA từ host-YUV qua copy/resize/forward/flow/warp/clamp/pack/D2H đến YUV+quantOffsets sẵn cho C-API, gồm cấp phát phát sinh. Báo p50/p99; mục tiêu 5/10 ms [CẦN XÁC MINH]. P100, encode/lookahead và FPS đo riêng; lưu backend/driver." Ladder: **budget p50≤5.0/p99≤10.0; target p50≤3.5/p99≤6.0; stretch p50≤3.0/p99≤4.5 — gate thực thi chốt từ pilot đo trên T4, pre-registered**. **"Đóng băng splits, ngân sách CPU/GPU và gates trước test."** `[MB-CX-1.6]`
- F2 Hybrid synergy: theo §D. F3/F4 Transfer: R3D-18. F5 Proxy-direction: theo §C. F6 Thống kê: theo §D. F7 Zero-shot: H3.
- **Lịch test: "Smoke mỗi PR (CPU, 1 chuỗi, ST-01/ST-10) vs nightly-Kaggle (đủ 10 ST); stress đầy đủ trước phát hành; ST-03/04 ghi phương pháp đo hoặc loại."** `[MB-ZC-3.3, MB-CX-3.5]`

## F. SCOPE PHASE 1 KAGGLE

- **Tầng 1a (pilot):** COCO **500 ảnh carve-out từ train2017** (2 detectors: YOLOv11-L + Faster R-CNN) + K400 **100 clips**. **1b CHỈ MỞ sau khi 1a pass Gate** `[khôi phục v1.0]`. 1b: COCO 5k val2017 (niêm phong) + **MOT17 đủ 7 chuỗi val** (2 chuỗi bị bác — bootstrap per-sequence vô nghĩa). K400 full/MOT20/VTM: Phase 2–3. **"Dự toán bằng throughput đo, tách CPU/GPU-hour (đo 200-ảnh intra trước)"** `[MB-CX-3.6]`.
- **"Manifest đếm encode: oracle ban đầu + calibration probes + làm mới nhãn + eval; trần làm việc 60 GPU-hour, job ≤6h, /kaggle/working <16GB; vượt dự toán → sửa scope trước khi chạy."** `[MB-CX-3.6]`
- **"x265 bridge truyền YUV qua in-memory IPC (BytesIO/POSIX FIFO); CẤM ghi YUV chưa nén xuống disk (/kaggle/working) để tránh cạn disk và iowait."** `[MB-AG-1.3]`
- Provenance: mỗi run lưu SHA/config/build/seed/log; **kiểm repo hiện hữu trước khi sửa** (HEAD e8f4117); **token pool ngoài repo**.
- Video sampling khối liên tục, tách theo chuỗi; **sổ split: mọi calibration/holdout tách từ train2017, danh sách SHA đóng băng trước khi code Student; val2017 niêm phong tới protocol freeze** `[MB-ZC-2.2]`.

## G. CLAIMS

- C1 **"foundation-generalist chỉ sau probes text/QR/rare/tiny-face"** — **ARPS bắt buộc vào Phase 1: 50 ảnh micro-text/QR + 50 ảnh vật thể nhỏ <16×16px; metric: mAP downstream không suy giảm quá 1.0pp so với anchor x265 stock** `[MB-AG-3.2, MB-CX-3.6-v1.2]`.
- C2 **"no out-of-band side information — điều khiển chỉ qua in-band syntax chuẩn; bitstream giải mã không lỗi bởi ≥2 decoder stock THEO TỪNG CODEC (HEVC: FFmpeg + HEVC reference decoder; VVC: FFmpeg + vvdec)."** `[MB-ZC-1.2]`
- C3 **Cấm** Title/Abstract/Intro: "task-agnostic", "100% loss-free", "real-time guaranteed", và mọi số chưa đo (15–25%, 18.5%, +0.4pp, 1.4/2.8/2.5 ms). Ngưỡng latency = "design budget", không phải kết quả.
- C4 **"Pilot chỉ kết luận HEVC; VVC mở rộng sau spike. Giữ single-forward/single-encode; không suy kết quả HEVC thành bằng chứng xuyên codec."** `[MB-CX-3.7, MB-CX-2.7]`

## H. ROADMAP & PHÂN VAI

1. **GATE 0 codec spike (Codex, tuần 1) trước mọi code model** — theo E-F0. PASS → mở khóa training; FAIL → fallback bằng văn bản (M2-only hoặc Δx-only).
2. Codex implement + push `github.com/wagur1/prism-vcm`: repo S5 + `prism/training/oracle_sweep.py`, `prism/training/calibration.py`, `baselines/`, `prism/codec/configs/`, `tests/test_qp_map_effect.py` (encode A/B, trace residual-block, decode độc lập), CI smoke-PR.
3. ZCode/agy: cron trông job Kaggle, tải log push lên repo, cập nhật README theo mốc.
4. **"VVC chỉ mở rộng sau kiểm chứng bridge và conformance"** — audit `CuQpDeltaQp*` stock trước, patch timebox ≤1 người-tuần, extended intra subset `[MB-CX-2.4, ZC-1, ZC-2]`.

## I. PHÂN XỬ CHỦ TỌA — 8 mục giữ nguyên như CONFIG_V1.md §I, TRỪ: §I.8 được bổ sung quy tắc ưu tiên "công thức H2 tại §D là văn bản chính thức" (mục D).

## J''. NHẬT KÝ (chính xác, đầy đủ)
- **v1.0 → v1.1:** bổ sung nguyên văn điều kiện CX-1/CX-2/CX-3; breaker hai tầng; H2 công thức; "cận trên CI95% dưới 0". LỖI: §J v1.1 sai khi tuyên bố "chỉ bổ sung" — nén mất: PCHIP, toàn-bộ-bytes, cùng GOP/công cụ, đơn vị bootstrap, "trước mọi code model", nightly-Kaggle, Gate 1a→1b, đo 200 ảnh, "mAP downstream", cap 0.3ms, "QP hợp lệ".
- **v1.1 → v1.2:** khôi phục các mục trên; thêm "QP hợp lệ" (bổ sung mới, không phải khôi phục); cấm phá floor; ưu tiên §B.1; breaker zero-ngay + giữ ≥3 frame; §D-H2 thay công thức dấu §I.8; ARPS đủ text/QR/rare/tiny-face. LỖI BỊ BẮT TIẾP (CX-2): nén mất "≥1.000 resamples" và chữ "độc lập" của ≥3 seeds tại §D; mất "L_align/Huber-towards-QP-oracle" tại §C; mất định nghĩa M1–M4 tại §D; §J' không khai báo.
- **v1.2 → v1.3:** khôi phục: "≥3 seeds ĐỘC LẬP", "cluster bootstrap 95% ≥1.000 resamples", "L_align/Huber-towards-QP-oracle", định nghĩa M1–M4. §J'' này khai báo đầy đủ mọi khác biệt so với v1.0. Không thay đổi bất kỳ điều khoản make-or-break nào của 9 ghế.
