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

Probe đã hoàn tất 36/36 forward, không encode, không train. Kernel: [trnhlng/prism-raw-probe-497eff6-1104-173818](https://www.kaggle.com/code/trnhlng/prism-raw-probe-497eff6-1104-173818). Mã probe pin `497eff64aa5855fb8951be9ec935b108a9af40e2`; checkpoint khớp feasibility. Môi trường: torch `2.11.0+cu130`, timm `1.0.30`, numpy `2.1.3`, khớp các phiên bản tương ứng trong benchmark.

### Kết quả raw-output

| Nguồn | CRF | Logit nhỏ nhất | Residual native nhỏ nhất | Residual lớn nhất | Tỷ lệ abs(residual) >= 7.99 |
|---|---:|---:|---:|---:|---:|
| MOT17-02 | 22 | 4.81435 | 7.998947 | 8.0 | 100% |
| MOT17-02 | 32 | 4.78480 | 7.998883 | 8.0 | 100% |
| MOT17-02 | 37 | 4.76902 | 7.998847 | 8.0 | 100% |
| MOT17-04 | 22 | 4.41846 | 7.997676 | 8.0 | 100% |
| MOT17-04 | 32 | 4.39930 | 7.997585 | 8.0 | 100% |
| MOT17-04 | 37 | 4.38903 | 7.997535 | 8.0 | 100% |

Raw logit trên tất cả records nằm trong [4.38903,18.16965]. Dùng lại công thức `8*tanh(interpolate(logits))` khôi phục đúng đầu ra model: sai số lớn nhất bằng 0 ở cả 36 records.

**Kết luận được hỗ trợ:** residual bị bão hòa về +8 trước cap trên tập frame/CRF đã kiểm tra. Cap1 chỉ biến đầu ra đã bão hòa thành +1. Không chỉ là giả thuyết clipping một bản đồ còn nhiều biến thiên. Raw logits vẫn có biến thiên; không gọi mọi tầng là hằng số hoặc kết luận bão hòa trên mọi dữ liệu.

### Kết quả QP

- Bản đồ sau phép chiếu zero-mean nằm trong khoảng xấp xỉ [−1.483475,+1.162764].
- Cả ba khoảng `[-6,+6]`, `[-5,+5]`, `[-4,+6]` đều có **0% cap occupancy** trên 36 records.
- Sai khác lớn nhất giữa đối xứng và bất đối xứng chỉ `5.96e-8` offset, phù hợp sai số tính toán. Không khẳng định bitstream giống nhau từ phép forward-only này.
- Compensation làm mean map tăng khoảng 0.00462–0.00806 QP; không phải bitrate đo thật.

**Kết luận được hỗ trợ:** các giới hạn bất đối xứng hiện tại không tạo can thiệp đáng kể trên tập input của probe. Kết quả đó giải thích tính gần đồng nhất của requested maps, không chứng minh codec bỏ qua controls hoặc mọi phương pháp asymmetric allocation vô ích.

### Quyết định tiếp theo

1. Dừng mở rộng encode với checkpoint residual này. Tăng cap 2/3 trên các input probe vẫn tạo mức tăng sáng đồng đều lớn hơn.
2. Không huấn luyện lại hoặc reinitialize toàn model dựa trên phép đo này. Cần tách tác động normalization/train–eval, conditioning và từng thành phần gradient trên tập development/calibration có provenance.
3. Giữ QP bất đối xứng làm câu hỏi nghiên cứu, nhưng không lấy phép đổi khoảng chặn không hoạt động làm novelty đã xác nhận.
4. Chưa chạy ablation teacher hoặc RL; dữ liệu/thiết kế đó chưa đủ để phân biệt nguyên nhân.

### Raw artifact identities

| Artifact | SHA256 |
|---|---|
| MOT17-02-FRCNN/controller_diagnostics.json | `269a6107390a2deec0af83793ce4a4a829bc8a960a0f4935f706954104c26660` |
| MOT17-04-FRCNN/controller_diagnostics.json | `78b85b23f428497fd95c5ec563303d39ced46de65790bd258285af1ea5d39c20` |
| probe_state.json | `a0c6954e4eb9d3cc89865c97e49a2c272a11b8cf9da5f55ad7f089e4009e531a` |

## Probe normalization và conditioning: kết quả đã đo

Kernel [trnhlng/prism-bn-probe-a9334ce-1104-175119](https://www.kaggle.com/code/trnhlng/prism-bn-probe-a9334ce-1104-175119), mã pin `a9334ceeb53872b1c581c49efe7a03626dd5118d`.

Tám ảnh cố định: frame1/16/33/40 của MOT17-02 và MOT17-04. Sáu batch forward ở CRF32: ba chế độ, mỗi chế độ có batch trộn tám ảnh và batch lặp ảnh đầu tiên tám lần. Không optimizer, không encode, không ghi checkpoint.

Chế độ BN batch statistics giữ nguyên forward gốc của `BatchNormAct2d`, activation và eval flags. Chỉ tạm thay running_mean/running_var bằng None, track_running_stats=False, momentum=0 trong lúc gọi BN. Tham chiếu buffers được khôi phục trong finally; toàn bộ state_dict được đối chiếu trước/sau. Đây là can thiệp chẩn đoán, không phải đề xuất dùng batch statistics của test set khi triển khai.

| Chế độ | Thành phần batch | Residual nhỏ nhất | Residual trung bình | Tỷ lệ abs(residual)>=7.99 |
|---|---|---:|---:|---:|
| Eval gốc | Trộn 8 ảnh | 7.997585 | 7.999974 | 100% |
| Eval gốc | Lặp ảnh đầu | 7.998967 | 7.999979 | 100% |
| BN dùng batch statistics | Trộn 8 ảnh | 7.999285 | 7.999997 | 100% |
| BN dùng batch statistics | Lặp ảnh đầu | 7.999679 | 7.999996 | 100% |
| Bỏ QP conditioning | Trộn 8 ảnh | 7.999765 | 7.999999 | 100% |
| Bỏ QP conditioning | Lặp ảnh đầu | 7.999912 | 7.999999 | 100% |

Report xác nhận `state_unchanged=true`, `source_files_unchanged=true`, 45 BN layers và 6 batch forwards. Torch `2.11.0+cu130`; launcher yêu cầu timm `1.0.30`. Reconstructed tanh output khớp actual output, max error bằng 0.

### Kết luận có giới hạn

- Thay running statistics bằng batch statistics **không giải quyết** bão hòa trên tập probe.
- Bỏ QP conditioning **không giải quyết** bão hòa; output còn gần +8 hơn trong phép thử này.
- Không đủ căn cứ sửa BN hoặc bỏ conditioning để tuyên bố đã sửa lỗi.
- Bão hòa tồn tại trong mapping residual của checkpoint hiện tại dưới nhiều điều kiện forward; chưa xác định thời điểm hoặc cơ chế lịch sử gây ra nó.
- Không được nói đã loại trừ mọi vai trò của BN hoặc conditioning trong quá trình học: probe chỉ kiểm tra can thiệp ở inference.

### Khuyến nghị sau khi hoàn tất hai probe

1. Không mở rộng RD benchmark với residual checkpoint này và không tăng cap để tìm kết quả đẹp.
2. Giữ artifacts/checkpoint cũ nguyên vẹn làm negative result.
3. Trước khi sửa training: đo từng thành phần loss/gradient trên dữ liệu train/calibration có provenance; đối chiếu objective, proxy đã fitted hay chưa, và raw-QP versus projected-QP trong train/deploy.
4. Nếu nghiên cứu huấn luyện lại residual head, thiết kế riêng với initialization gần identity, giữ QP/backbone cố định trong đối chứng đầu tiên, và giám sát rate–task hợp lệ. Đây chỉ là kế hoạch tiếp theo; lượt này không huấn luyện.
5. Không làm E3 teacher ablation hoặc thêm temporal/RL khi hai nhánh điều khiển chưa được chứng minh có giá trị.

Raw artifacts:
- `normalization/controller_normalization.json`: SHA256 `22a8fc4e00c9a2dcc293992592a437c3a8b89c4ae69ddf2cc3a905940a1a4332`.
- `normalization_probe_state.json`: SHA256 `acce17d469d9a93bf35a16dabffae9f8fd0ace1de25b0abb636625d9b0cc58e9`.

## Phạm vi hoàn tất và việc chưa làm

Đã hoàn tất thống kê bảy feasibility shards, probe raw-output, probe normalization/conditioning, các tests và tổng hợp sáu bài trong `PREPROCESSING_LITERATURE_APPLICATION.md`.

Chưa có training mới, teacher-value ablation, proxy calibration thật, kiểm chứng đa seed hoặc benchmark đầy đủ. Không có tuyên bố novelty hoặc journal readiness từ những phép thử này. Các lỗi dispatcher cũ không được sửa trong lượt chẩn đoán.

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
