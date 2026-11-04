# PRISM bounded residual: kết quả feasibility và bước chẩn đoán

## Phạm vi bằng chứng

- Mã collector: commit `7f94434c737da2438290c8a03ec0f9dbb5a98fd4`.
- Checkpoint: seed 0, MobileNetV4-Conv-Small, 25 epochs; SHA256 `623b774ba216e7c0e39ce233f7e2cab78f5448acaa2f3ec01e47fe3ffd32c345`.
- Bảy nguồn MOT17: 02, 04, 05, 09, 10, 11, 13; lấy frame1–64.
- Can thiệp residual chỉ ở frame33–48; prefix32 và suffix16 frame giữ nguyên input.
- CRF32, cap1 luma, GOP32/B0, cùng x265 và detector/tracker giữa hai nhánh.
- 7/7 shard và 14/14 encode đã hoàn tất. Không huấn luyện mới.
- Đây là development feasibility, không calibration, không BD-rate và không test xác nhận độc lập.

## Kết quả từng chuỗi

HOTA/IDF1 tính trên 64 frame; chênh lệch là điểm phần trăm theo thang báo cáo, không phải phần trăm tương đối.

| Sequence | Anchor bytes | Residual bytes | Δ bytes (%) | Δ HOTA | Δ IDF1 |
|---|---:|---:|---:|---:|---:|
| MOT17-02 | 451958 | 451980 | +0.00487 | −0.09311 | −0.20115 |
| MOT17-04 | 258201 | 257576 | −0.24206 | +0.05851 | +1.41005 |
| MOT17-05 | 167255 | 167946 | +0.41314 | +0.84699 | +0.66132 |
| MOT17-09 | 394409 | 393245 | −0.29513 | −0.27566 | +0.09053 |
| MOT17-10 | 608554 | 610240 | +0.27705 | −1.52638 | −1.71683 |
| MOT17-11 | 639458 | 639841 | +0.05989 | −0.01819 | +0.22805 |
| MOT17-13 | 534469 | 534616 | +0.02750 | +0.12374 | +0.49689 |

Tổng byte: 3,054,304 thành 3,055,444, tăng 0.03732%. Trung bình chênh lệch HOTA theo chuỗi: −0.12630 điểm; IDF1: +0.13841 điểm. HOTA tăng ở 3/7 chuỗi.

Không có khoảng tin cậy xác nhận hay nhiều seed ở lượt này. Không dùng các trung bình mô tả trên để khẳng định superiority.

## Phát hiện chính

Mọi pixel luma trong đoạn can thiệp của cả bảy chuỗi đều có delta đúng `+1`. Tỷ lệ raw residual bị clip tại cap là 1.0; saturation vào biên `[0,255]` bằng 0 ở các báo cáo đã đọc.

Do đó, **trong điều kiện thử nghiệm này**, phép biến đổi thực thi tương đương tăng sáng đồng đều trong frame33–48. Kết quả này không chứng minh điều khiển theo nội dung hiệu quả. Nó cũng không tự chứng minh raw head là hằng số: các raw values khác nhau nhưng cùng vượt cap có thể bị biến thành cùng một giá trị.

Không cần thêm một lượt encode để chứng minh lại output đã biết chính xác là +1. Bước có giá trị hơn là đo raw logits và residual trước clipping.

## Kiểm tra checkpoint không chạy model

Checkpoint được tải từ output `htran123456/prism-tier1a-step3`, khớp hash của báo cáo MOT17. Đọc bằng `torch.load(..., weights_only=True, map_location="cpu")` cho thấy có `state_dict` đầy đủ và hai head.

History chỉ chứa `epoch`, `train_masked_huber`, `dev_masked_huber`. Epoch25: train masked Huber 0.59779045, dev 0.87027001. Không có loss/activation/statistics residual trong history này.

Bias cuối residual là khoảng −0.07214 và weights không bằng 0; chỉ nhìn bias không giải thích được output dương đồng đều. Chưa có căn cứ gọi đây là lỗi bias đơn giản.

## Phạm vi rà soát source

Source hiện tại dùng RGB normalized `[0,1]`, sau đó đổi luma về đơn vị 8-bit bằng `*255` trước khi cộng residual và đưa qua proxy. Inference cũng cộng residual theo đơn vị luma 8-bit. Không tìm thấy lỗi thiếu `*255` trong các đoạn được kiểm tra.

Những điểm cần đo thêm, không phải nguyên nhân đã xác nhận:

- Proxy khởi tạo mới rồi đóng băng; các hệ số ban đầu là prior chưa fitted với x265.
- Proxy chỉ mô phỏng block-DCT của ảnh, không reference-frame state/motion.
- Residual và QP chia sẻ backbone; gradient giữa các mục tiêu có thể tương tác.
- `tanh` bão hòa làm giảm gradient theo logits; residual regularization tồn tại không bảo đảm thoát bão hòa.
- Train luma full-range BT.601-like, inference dùng pipeline BT.709 limited-range/YUV-to-RGB; đây là khác biệt cần kiểm tra, không tự đủ để giải thích mọi pixel+1.
- Báo cáo calibration cũ có mô tả detached/Dirac không còn phản ánh proxy hiện tại. Không dùng nó làm bằng chứng nguyên nhân.

## Raw-forward probe có giới hạn

Thiết kế probe không encode và không train:

- Hai nguồn: MOT17-02-FRCNN và MOT17-04-FRCNN.
- Sáu frame cố định mỗi nguồn: 1,16,33,40,48,64.
- Ba CRF cố định: 22,32,37.
- Tổng 36 forward, checkpoint giữ nguyên, model `eval()`/inference mode.
- Đo input range, conditioned features, logits residual trước tanh, residual sau tanh/native resize/cap, và trường QP.
- So sánh các phép chiếu đối xứng `[-6,+6]`, `[-5,+5]` và bất đối xứng `[-4,+6]`, có/không compensation. Đây chỉ là chẩn đoán bản đồ, không phải hiệu quả codec.
- Không đổi cap hoặc lựa chọn frame theo kết quả đẹp; không sửa weights.

Kết quả probe sẽ được bổ sung riêng sau khi artifact đã thu và kiểm tra. Thiết kế ở đây không phải bằng chứng probe đã chạy.

## Tham chiếu artifacts đã thu

| Sequence | Kernel | SHA256 của feasibility_report.json |
|---|---|---|
| 02 | `shungg05/prism-residual-feasibility-02-1104-135216` | `3014882ac784d07562f1f17b5f049ed19c7fe33daf7645b9b17912455c517fb9` |
| 04 | `trnhlng/prism-residual-feasibility-04-1104-135216` | `e4915b6e895a6bd1fec789f804203cbfb68c7931d0cefb8cadac2e8ebc4c35d4` |
| 05 | `shungg05/prism-rdx-feas-05-7f94434-1104-135216` | `1bfa098f3cd731a7630ee916d5c5d6c1f47e493e52cb7bf2277e7795ee18ee82` |
| 09 | `trnhlng/prism-rdx-feas-09-7f94434-1104-135216` | `baae452d5c38c5b9ff8c2c25636b1d31368f3a830160cc72f32b52e15f638b32` |
| 10 | `shungg05/prism-rdx-feas-10-7f94434-1104-135216` | `f1a5e2975025f628b207c919d4c070a9d9eb2c0b99d7b5d5bc2473174c133bbc` |
| 11 | `trnhlng/prism-rdx-feas-11-7f94434-1104-135216` | `552ee7c68c7113bf3098a3e022c87d8cfa32ac2965ea10c84a2d2e70482d0cf9` |
| 13 | `shungg05/prism-rdx-feas-13-7f94434-1104-135216` | `bce6448a90c69443aeb471ff4e3a035c3acce42020787d059a98667956aa29c4` |

Các báo cáo collector luôn ghi `research_only=true`, `calibration_status=not_established`, `gate_authorization=false`.
