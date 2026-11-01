# GATE 0 — STRUCTURAL FAIL (confirmatory run, frozen criteria e6d6516a)
- Kernel: htran123456/prism-gate0-full (confirmatory per council esc#9 hard-stop)
- Kết quả: 12 encodes / 120 frames hoàn tất; group 1 (qp27-gop1) PASS toàn bộ **kể cả checkerboard** (sep 2.0 ≥ 0.5); group 2 (qp27-gop10): plus6 PASS; **FAIL: minus6 compensated frame 3 median 3.000 < floor 3.5**.
- Phân loại: FAIL theo frozen criteria — KHÔNG phải lỗi cơ chế (sign-fidelity 100%, mọi ΔQP dương, bits thay đổi đúng chiều). Là lỗi BAND CALIBRATION: độ sâu bù CRF phụ thuộc GOP (gop1: 4.0; gop10 frame 3: 3.0) — ZC-2 đã cảnh báo transferability risk ở esc#9.
- Hard-stop (esc#9, 4/4 ghế): confirmatory result là FINAL; cấm sửa criterion nếu không có deep telemetry của CX-1. → escalate lên chủ dự án.
