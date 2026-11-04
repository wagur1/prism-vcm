# Bounded Residual: Hợp đồng và hướng nghiên cứu

## 1. Mục tiêu và phạm vi

Tài liệu này định nghĩa nền tảng để nghiên cứu tiền xử lý video bằng residual có giới hạn. Đây là hướng thay thế để đánh giá bên cạnh QP bất đối xứng, không phải quyết định loại bỏ hướng QP.

Phần bổ sung gồm helper residual, parser evidence, runner đánh giá video riêng, collector nghiên cứu codec, CPU tests và tài liệu. Runner QP lịch sử, backbone, QP head, checkpoint, calibration gate và artifacts cũ không đổi. Không bổ sung Gaussian, DSU-Net, GRU, temporal hay kết hợp residual + QP.

**Chưa có kết quả chứng minh residual cải thiện bitrate, detection hoặc tracking.** Các tests chỉ kiểm tra hợp đồng số học và điều kiện cho phép áp dụng residual.

## 2. Kiến trúc mục tiêu

### Offline: mục tiêu huấn luyện trong tương lai

1. Student nhỏ nhận ảnh/video và mức nén mục tiêu.
2. Student dự đoán residual luma có giới hạn.
3. Video đã điều chỉnh đi qua proxy codec có bằng chứng calibration phù hợp.
4. Học theo chi phí rate, distortion và bảo toàn thông tin tác vụ.
5. DINOv2/SAM2 có thể cung cấp hướng dẫn offline; không cần chạy khi triển khai.

Các bước này là thiết kế nghiên cứu, chưa được triển khai thành một trainer mới trong thay đổi này. Proxy DCT/STE hiện có không tự động là mô phỏng inter-frame x265. Trọng số được đóng băng không có nghĩa phải ngắt gradient truyền đến input: khi phát triển trainer, cần phân biệt hai việc đó. Rounding pixel cho huấn luyện cũng cần một cơ chế gradient được khai báo và kiểm chứng; helper dưới đây chỉ dành cho suy luận số nguyên.

### Online: hướng dự kiến

Student dự đoán residual từ frame gốc và CRF. Caller đưa residual về kích thước native, kiểm tra điều kiện calibration, rồi cộng residual đã giới hạn vào luma. Chroma giữ nguyên. Codec tiêu chuẩn nén frame đã điều chỉnh và decoder tiêu chuẩn khôi phục video cho tác vụ máy.

Nhánh residual-only không dùng external QP offsets. Không có thử nhiều candidate hay teacher inference trong đường online. Helper đã được nối vào runner riêng `tier1b/run_bounded_residual_mot17.py`; cap lớn hơn 0 chỉ được chạy khi raw evidence phù hợp vượt gate. Chưa có calibration hay benchmark thực mới từ phần bổ sung này.

## 3. Những thành phần được tái sử dụng

- `prism/models/student.py`: output có tên `residual`, kích thước `[B,1,256,256]`, giới hạn bởi `8*tanh(...)`. Đây là đơn vị cường độ pixel 8-bit, không phải vector chuyển động.
- `tier1a/run_tier1a_step3.py`: đã có logic bilinear resize residual về `[H,W]` và cộng vào luma với nearest-even rounding, clamp `[0,255]`. Helper mới giữ cách diễn giải đơn vị và thứ tự xử lý, nhưng không sửa helper lịch sử.
- `prism/control/contract.py`: giới hạn residual ±3 cho reference/unknown frames. Helper mới dùng cap tối đa 3 cho mọi frame; không hỗ trợ nới lên ±8 cho non-reference B trong giai đoạn này.
- `prism/training/calibration.py`: giữ nguyên `CalibrationGate`; chưa đạt calibration thì không áp dụng residual.

Checkpoint cũ từng có đường huấn luyện cả hai head, nhưng điều đó chưa chứng minh head residual hữu ích. Không coi head tồn tại hoặc load thành công là bằng chứng về hiệu quả.

## 4. API mới

```python
from prism.preprocessing.residual import apply_bounded_residual

output, stats = apply_bounded_residual(
    payload,             # packed YUV420p8: Y rồi U rồi V
    residual_native,     # real numeric ndarray [H,W], sau resize
    width,
    height,
    cap=0,               # chỉ nhận số nguyên 0, 1, 2, 3; mặc định tắt
    gate=None,           # đối tượng CalibrationGate; không nhận dict/boolean
    step=0,              # training step mà gate kiểm tra; không phải frame index
    qp=None,             # định danh mức lượng tử trong calibration
    frame_type="unknown",
)
```

### Đầu vào

- `payload` phải là immutable `bytes`, đúng `H*W*3/2` byte.
- `width`, `height` là số nguyên dương chẵn. Helper không suy đoán stride hoặc planar layout.
- `residual_native` phải có kích thước chính xác `[H,W]`, kiểu số thực hoặc số nguyên; không nhận boolean, complex, object hoặc chuỗi.
- Helper không resize. Caller phải resize trước; cap được áp dụng sau resize.
- Residual đã ở đơn vị mức luma. **Không chia thêm cho 255** trước khi cộng vào luma 8-bit.
- Cap nhận số nguyên để bảo đảm giới hạn thay đổi sau làm tròn uint8 không vượt cap. `cap=0.5` hoặc `cap=3.0` bị từ chối; không tự ép kiểu thông số sai.

### Điều kiện áp dụng

Residual chỉ được áp dụng khi tất cả điều kiện đúng:

1. Cap lớn hơn 0.
2. Có `CalibrationGate`.
3. `gate.allow_delta_x(step)` còn hợp lệ.
4. Cặp `(qp, frame_type)` thuộc `gate.expected_cells`.
5. Residual hợp lệ và hữu hạn trên đường áp dụng.

Thiếu hoặc hết hạn calibration, sai cell, hoặc cap bằng 0 đều trả lại payload không thay đổi. Helper cũng gọi `CalibrationGate.apply_delta_x()` trước khi giới hạn và áp dụng.

**Không tạo một gate giả có `verified=True` để chạy thí nghiệm.** Không dùng báo cáo dummy/lỗi thời hoặc tests tổng hợp làm bằng chứng calibration.

Gate hiện tại không chứa toàn bộ checkpoint hash, codec build, cap, geometry hoặc danh tính dữ liệu. Caller tương lai phải ràng buộc các thông tin đó trong manifest và bằng chứng calibration. Việc một cell có tên `(32, "P")` không tự chứng minh QP32, CRF32 và các trạng thái codec khác nhau là tương đương. Trường `qp` ở helper là định danh cell, không phải phép suy ra effective block QP.

Không mặc định gán frame type khi chưa biết. Có thể dùng cell `unknown` chỉ khi chính điều kiện đó đã được calibration thực sự. `step` không phải thời hạn theo đồng hồ hoặc số frame video.

### Dữ liệu không hữu hạn và fail-closed

- Khi đường residual bị tắt hoặc gate không cho phép, NaN/Inf trong prediction không được lan vào output; payload giữ nguyên.
- Khi gate cho phép áp dụng, NaN/Inf gây `ValueError`.
- Sai geometry, payload, shape hoặc dtype vẫn bị từ chối ngay cả khi cap bằng 0. Fail-closed không có nghĩa bỏ qua lỗi cấu trúc.

### Phạm vi giá trị

Luma được clip trong `[0,255]` như helper lịch sử. Đây **không phải** phép legalization BT.709 limited-range `[16,235]`; cần khai báo color/range convention rõ khi thiết kế thí nghiệm sau này. Chroma được giữ nguyên từng byte. Cap nhỏ không bảo đảm bảo toàn ngữ nghĩa, perceptual quality hoặc hiệu quả codec.

## 5. Thống kê thực tế sau lượng tử hóa pixel

`stats` chỉ chứa scalar có thể serialize JSON, không chứa tensor hoặc bản đồ frame:

| Trường | Ý nghĩa |
|---|---|
| `cap`, `step`, `calibration_qp`, `frame_type` | Cấu hình và cell yêu cầu |
| `applied` | Đường residual được cho phép; không đồng nghĩa có pixel thay đổi |
| `bypass_reason` | `cap_zero`, `missing_calibration`, `unverified_calibration`, `uncalibrated_cell`, hoặc `None` |
| `num_luma_pixels` | Mẫu số H×W cho các tỷ lệ |
| `changed_pixel_fraction` | Tỷ lệ luma uint8 đầu ra khác đầu vào |
| `min_pixel_delta`, `max_pixel_delta`, `max_abs_pixel_delta` | Chênh lệch thực tế tính bằng phép trừ có dấu |
| `mean_pixel_delta`, `mean_abs_pixel_delta` | Trung bình chênh lệch có dấu và trị tuyệt đối |
| `residual_clipped_fraction` | Tỷ lệ residual được cho phép vượt cap trước khi giới hạn |
| `saturation_fraction` | Tỷ lệ luma sau cộng/làm tròn nằm ngoài `[0,255]`, trước final clamp |

Khi bypass, mọi thống kê thay đổi và clipping bằng 0; helper không báo số đo clipping giả cho prediction chưa áp dụng. Trường `applied` vẫn có thể là `True` khi mọi residual nhỏ đều biến mất sau rounding.

Helper không ghi ra đĩa. Runner riêng tổng hợp statistics theo số pixel và lưu cùng source/checkpoint/build/config identities. Không lưu toàn bộ video hoặc bản đồ residual trong RAM. Các fixtures kiểm thử không tạo ra bằng chứng benchmark.

## 6. Quy trình nghiên cứu tiếp theo, cần phê duyệt riêng

### Bước A: kiểm chứng calibration

- Dùng tập calibration độc lập với train/development/test.
- So sánh proxy và codec thật với trạng thái codec tương đương.
- Đánh giá ảnh hưởng lên frame phụ thuộc khi dùng inter-frame coding.
- Gắn bằng chứng với checkpoint, codec build/config, input geometry, preprocessing contract và phạm vi cap.
- Không hạ tiêu chí hoặc đổi từ QP sang CRF mà không khai báo protocol mới.

Helper có gate không cung cấp cờ bypass calibration. Collector nghiên cứu tại mục 9 cho phép thu phép đo có residual chưa calibration trong một đường tách biệt, yêu cầu xác nhận research-only và không tạo quyền triển khai. Việc chạy codec thật vẫn cần phê duyệt tài nguyên/ngân sách riêng; kết quả không được trộn với các kết quả đã xác nhận.

### Bước B: so sánh phương pháp

Giữ các nhánh riêng biệt:

1. Anchor không tiền xử lý.
2. QP bất đối xứng hiện có, không đổi định nghĩa.
3. Residual-only có giới hạn.
4. Gaussian/filter-only hiện có làm baseline.

Chưa ghép residual + QP. Cap 0/1/2/3 là phạm vi khảo sát dự kiến; lựa chọn trên development, không chọn bằng kết quả test. Đếm toàn bộ bitstream, kiểm tra số frame và dùng tác vụ sau giải mã. Không suy luận tốt hơn tại cùng bitrate chỉ từ cùng CRF.

Báo cáo mọi kết quả bất lợi, pixel thay đổi thực tế, clipping, đầu ra gate và latency đo được. Không coi giảm proxy loss là thành công trên codec thật.

### Bước C: nghiên cứu khả năng chuyển giao

Mục tiêu tiềm năng là một checkpoint nhỏ có hiệu quả trên detector/tracker không dùng khi huấn luyện. Đây là giả thuyết cần kiểm chứng, không phải novelty đã xác nhận.

- Tách video nguồn giữa train, calibration, development và test.
- Không coi bảy chuỗi MOT17 đã ảnh hưởng đến lựa chọn phương pháp là holdout mới.
- Giữ task network/teacher evaluation độc lập theo protocol; báo rõ supervision nào được dùng khi train.
- Nếu nghiên cứu temporal sau này, so sánh spatial/temporal với cùng dữ liệu huấn luyện để không nhầm lợi ích thêm dữ liệu với lợi ích kiến trúc.

## 7. Quan hệ với công trình trước

Bài **A Preprocessing Framework for Video Machine Vision under Compression** của Zhao và cộng sự đã nghiên cứu neural pixel preprocessing, nhánh spatial/temporal, virtual codec và rate–distortion–task loss. Vì vậy, chỉ kích hoạt head residual không đủ làm đóng góp mới.

Khả năng chuyển giao sang mạng tác vụ chưa thấy, chi phí triển khai đo được và một cơ chế học thực sự khác biệt là các câu hỏi có thể nghiên cứu. Chúng cần so sánh công bằng và rà soát tài liệu rộng hơn.

Bài DSU-Net minh họa kết hợp đặc trưng DINOv2/SAM2 cho phân đoạn, không chứng minh lợi ích nén video. Không đưa DSU-Net vào runtime của thay đổi này.

## 8. Runner video và bằng chứng calibration

### 8.1. Phạm vi runner

`tier1b/run_bounded_residual_mot17.py` đánh giá một nguồn video, với các nhánh `anchor` và `residual`. Mặc định cap 0, CRF 22/27/32/37, GOP32, B=0, device CPU. Giữ `SimpleByteTracker`, detector YOLOv11-L và đường chuyển màu của runner MOT17 cũ.

- Anchor/cap0 không tải student. Nhánh residual có cap dương chỉ suy luận một lần mỗi frame, sau đó resize và giới hạn residual bằng helper.
- Cả hai nhánh gửi `offset_generator=None` cho x265. Không phối hợp residual với external QP.
- `--max-frames` chỉ lấy prefix từ frame 1. Danh tính nguồn vẫn bao phủ toàn video, không tạo một source identity mới cho từng prefix.
- Đầu ra phải là thư mục mới, không chứa hoặc nằm trong nguồn/evidence/weights/build files. Không resume hoặc ghi đè run cũ.
- `--validate-only` không tạo output, không `torch.load`, không load native library và không encode. Nó vẫn đọc toàn bộ frame files để tính hash, nên có chi phí I/O. Nó không chứng minh checkpoint load được hoặc binary chạy được.
- Khi cap > 0 thiếu evidence, mismatch hoặc gate không đạt, CLI trả JSON `status=blocked`, exit code 2. Không biến residual bị tắt thành kết quả thành công.

### 8.2. Evidence protocol: CRF-setting và unknown frame type

Parser `prism/evaluation/residual_evidence.py` chỉ nhận protocol mới **`bounded_residual_crf_unknown_v1`**, schema version 1. Nó tái sử dụng ngưỡng thống kê của `CalibrationGate`, nhưng khai báo rõ cell đại diện cho **điều kiện CRF**, không phải effective block QP. Không chấp nhận evidence QP/I/P lịch sử rồi đổi nhãn thành CRF/unknown.

Frame type là `unknown` cho cấu hình inter-frame hỗn hợp đã calibration. Runner không suy đoán actual I/P từ số thứ tự frame. B=0 và toàn bộ encoder config phải khớp context. Nếu chưa có phép đo phù hợp, runner phải bị chặn.

Top-level JSON chỉ có các trường:

| Trường | Nội dung bắt buộc |
|---|---|
| `schema_version` | Số nguyên 1 |
| `protocol` | `bounded_residual_crf_unknown_v1` |
| `context` | Danh tính/config triển khai chính xác |
| `calibration_sources` | Danh sách nguồn calibration không rỗng, không trùng đánh giá |
| `history` | Các lần calibration liên tiếp, bắt đầu step 0 và cách nhau 1000 bước |

`context` chứa đúng:

- `checkpoint_sha256`, `seed`, `checkpoint_step`. Runner hiện hỗ trợ checkpoint seed 0. Step do người chạy khai báo rõ; checkpoint cũ không chứa step thì không được tự đoán.
- `encoder_config`: toàn bộ `dataclasses.asdict(EncoderConfig(...))`, không bỏ các field có giá trị `null`.
- `build_pin`: toàn bộ `BuildPin`, gồm library/shim SHA256, version, ABI, picture size và assembly flag.
- `width`, `height`, `fps`, `cap` tương ứng run.
- `color_contract`: `bt709_limited_yuv420p8_luma_clip_0_255`.
- `rate_control_condition`: đúng các field `mode="CRF"`, `value=<CRF nguyên>`, `cell_qp=<cùng giá trị>`, `frame_type="unknown"`, `interpretation="crf_setting_not_effective_qp"`.

Mỗi `calibration_sources` entry có `source_group`, `sha256`, `split="calibration"`. Các tên `MOT17-02`, `MOT17-02-DPM`, `MOT17-02-SDP`, `MOT17-02-FRCNN` cùng một nhóm nguồn. Nhóm và hash trùng nhau hoặc trùng nguồn đánh giá bị từ chối.

Danh tính nguồn đánh giá được tạo bởi `source_identity()`:

1. SHA256 bắt đầu bằng JSON gọn của `[width,height,fps,seq_length]`.
2. Theo thứ tự frame 1..N, thêm ASCII `\n<index>:<SHA256 của file frame>` vào hash.
3. Không đưa detector suffix hay tên thư mục vào hash.

Khi tạo evidence thật, sử dụng cùng quy ước hash cho nguồn calibration. Kiểm tra này không nhận biết chắc chắn các bản sao bị rename/re-encode hoặc data leakage từ pretraining; nguồn gốc dữ liệu vẫn phải được kiểm toán riêng.

Mỗi entry `history` có `step` và `cells`. Phiên bản này yêu cầu đúng một cell cho CRF tương ứng. Cell chứa đủ field của `CalibrationCell`:

- `qp`, `frame_type`, `split`;
- `actual_bits`, `predicted_bits`, `actual_delta_j`, `predicted_delta_j`, `direction_delta_j`: các danh sách số hữu hạn, ghép cặp và cùng độ dài;
- `semantic_passed`, `same_codec_state`, `feasible_plus_minus_one`, `dependent_frames_measured`: boolean thực, không dùng chuỗi `"true"` hay số 1.

Parser gọi lại `gate.assess()` cho từng bước. Nó không load cờ `verified=True`, không đổi weights khi replay và không sửa ngưỡng gate. Gate phải còn cho phép ở `checkpoint_step`; hai lần fail liên tiếp dẫn đến disabled không được phục hồi bằng một record `passed` về sau.

**Hashes và mẫu số không chứng minh phép đo thực sự được thực hiện.** Đây là kiểm tra cấu trúc, danh tính và thống kê; không phải chứng nhận độ trung thực của người tạo evidence. Không có file evidence giả kèm theo source để chạy thật. Dữ liệu tổng hợp chỉ nằm trong tests.

### 8.3. Lệnh sử dụng

Các đường dẫn dưới đây là placeholder và phải được thay bằng dữ liệu/build thực đã có. Các lệnh không tải model hoặc cài môi trường.

Kiểm tra CLI:

```bash
python tier1b/run_bounded_residual_mot17.py --help
```

Validate identity control, không tải checkpoint/native:

```bash
python tier1b/run_bounded_residual_mot17.py \
  --sequence-dir /path/to/MOT17-02-FRCNN \
  --weights /path/to/yolo11l.pt \
  --library /path/to/libx265 \
  --shim /path/to/prism_x265_shim \
  --build-pin /path/to/build_pin.json \
  --output /path/to/new_identity_run \
  --crfs 32 --cap 0 --validate-only
```

Validate residual có evidence thực cho một CRF:

```bash
python tier1b/run_bounded_residual_mot17.py \
  --sequence-dir /path/to/MOT17-02-FRCNN \
  --weights /path/to/yolo11l.pt \
  --library /path/to/libx265 \
  --shim /path/to/prism_x265_shim \
  --build-pin /path/to/build_pin.json \
  --output /path/to/new_residual_run \
  --checkpoint /path/to/student_seed_0.pt \
  --checkpoint-step 1000 \
  --evidence 32=/path/to/measured_crf32_evidence.json \
  --crfs 32 --cap 3 --validate-only
```

`1000` là ví dụ tham số, không phải step đã được xác minh của checkpoint hiện tại. Thay bằng step có provenance thật. Mỗi CRF được yêu cầu phải có một `--evidence CRF=PATH` riêng, ràng buộc đúng cap/config. Sau khi được duyệt chạy thật và preflight hợp lệ, bỏ `--validate-only` để chạy encode/decode/tracking cục bộ. Không có tích hợp Kaggle trong runner này.

Build hashes được đối chiếu trước native load; khi chạy thật, `X265Library(pin=...)` kiểm tra runtime hash/version/ABI/picture size. Assembly flag vẫn là thông tin khai báo của pin, không phải benchmark hay phép đo hiệu năng.

### 8.4. Artifacts và trạng thái

Thư mục run chứa:

- `residual_seq_report.json`: provenance, cấu hình, observations và trạng thái `running/failed/complete`.
- `attempts.jsonl`: từng điểm vận hành, kể cả thất bại, cùng thời gian và lỗi. Đây là nhật ký thực thi, không phải số lần training.
- `bitstreams/`: toàn bộ HEVC bitstream, cùng telemetry CSV khi codec thật cung cấp.
- `tracks/`: tracks chỉ tính từ frame giải mã.
- GT prefix khi cần đánh giá `--max-frames`.

Frame count hoặc byte accounting không khớp gây fail. Gate mất quyền áp dụng giữa chuỗi cũng gây fail; không tiếp tục dưới dạng identity. Điểm đã hoàn thành trước lỗi vẫn được giữ trong report. Thiếu GT được ghi `evaluated=false` và score null.

`student_seconds` bao gồm chuyển YUV-to-RGB, inference và đưa kết quả về CPU. `preprocessing_seconds` đo helper áp dụng residual. `encode_pipeline_seconds` bao gồm đọc/chuyển frame nguồn, student, preprocessing và encode; không cộng lại các thành phần vào tổng này. `decode_track_seconds` bao gồm decode, detection, tracking và ghi tracks. Đây là đo wall-clock cho nghiên cứu, không phải chứng nhận latency edge.

Không sinh BD-rate hoặc PASS nghiên cứu. Tính HOTA từ một pipeline chạy được không chứng minh neural residual hữu ích hoặc metric đã đạt parity với TrackEval chính thức.

## 9. Collector chẩn đoán codec: không phải chứng nhận calibration

### 9.1. Mục đích và ranh giới

`scripts/collect_residual_feasibility.py` thu **hai phép encode nghiên cứu** trên cùng video, CRF, cap và cửa sổ can thiệp. Nó không dùng proxy, không giả lịch sử huấn luyện, không tạo `CalibrationCell`, không mở gate và không tạo artifact hợp lệ cho `load_residual_evidence()`.

Hai nhánh cố định:

1. `anchor`: không thay đổi pixel.
2. `student_residual`: chỉ thay đổi luma trong đoạn can thiệp, dùng checkpoint MobileNetV4 đã có.

Không có fallback random weights. Checkpoint có backbone `small_conv_smoke_only` bị từ chối khi load; `validate-only` chưa load checkpoint nên không chứng minh backbone hoặc trạng thái huấn luyện. Metadata và hashes vẫn không tự chứng minh checkpoint đã được huấn luyện đúng.

Collector gọi helper nội bộ `_apply_residual_pixels()` sau khi người chạy khai báo research-only. Helper này chỉ làm phép biến đổi số học có kiểm tra, không có trường `applied`, `verified` hay quyền gate. `apply_bounded_residual()` dùng cùng phần số học sau kiểm tra gate; API fail-closed cũ giữ nguyên, bao gồm việc không cho NaN lan ra output khi bypass.

Pure helper yêu cầu residual hữu hạn ngay cả cap0. Gated cap0 vẫn giữ hành vi cũ: bypass byte-identical và thống kê bằng 0. Collector chỉ nhận cap1/2/3 và dùng residual zero hữu hạn ngoài đoạn can thiệp.

### 9.2. Cửa sổ và ngân sách cố định

Đánh số frame từ 1. Bắt buộc khai báo:

- `--intervention-start >= 2`: phải có prefix gốc không rỗng.
- `--intervention-end >= intervention-start`.
- `--suffix-frames >= 1`: giữ đoạn sau can thiệp để đo ảnh hưởng phụ thuộc.
- `--max-frames`: ngân sách frame cho **mỗi encode**, tính cả prefix từ frame 1.
- `--max-encodes 2`: đúng hai nhánh, không sweep hoặc tự retry.

Số frame thực tế mỗi nhánh là `intervention_end + suffix_frames`, không vượt ngân sách hoặc chiều dài nguồn. Mỗi nhánh mở encoder mới và nhận cùng prefix/suffix gốc. Chỉ input luma trong intervention thay đổi; chroma giữ nguyên.

**Cùng prefix input không chứng minh trạng thái encoder trước intervention giống tuyệt đối.** Lookahead/rate-control có thể nhìn các frame can thiệp tương lai. Sai khác decoded prefix cũng có thể xảy ra. Collector chỉ báo phép so sánh end-to-end, không tuyên bố đã cô lập motion-reference state hoặc đo gradient-direction của codec.

### 9.3. Nguồn dữ liệu và xác nhận nghiên cứu

`--source-manifest` là JSON có đúng các field:

```json
{
  "schema_version": 1,
  "source": {
    "source_group": "MOT17-02",
    "sha256": "REPLACE_WITH_MEASURED_FULL_SOURCE_SHA256"
  },
  "role": "development"
}
```

Đây là schema minh họa, không phải manifest hợp lệ hay tập đã phê duyệt. Tính hash bằng `source_identity(MOTSequence(path))` theo quy ước ở mục 8.2. Vai trò chỉ nhận `development` hoặc `calibration`; `test` bị chặn. Tên detector suffix được chuẩn hóa cùng nhóm nguồn. Người chạy vẫn chịu trách nhiệm provenance; đổi nhãn manifest không làm dữ liệu đã dùng trở nên độc lập.

Collector đọc hash toàn bộ source và kiểm tra lại giữa hai nhánh. Không tự tìm checkpoint/data trên ổ đĩa, không tải model, không sửa split. Đường output phải mới và không chồng nguồn/checkpoint/weights/build/manifest.

`--validate-only` không cần cờ execution research-only, không load model/native và không ghi output. **Muốn chạy encode phải thêm `--research-only`.** Cờ này là xác nhận loại phép thử, không biến nó thành calibration hoặc cho phép chạy ngoài ngân sách đã duyệt.

### 9.4. Ví dụ chuẩn bị lệnh, chưa chạy thật

Các path và số frame dưới đây chỉ là ví dụ. Phải chọn nguồn, cap, CRF và ngân sách trước khi xem kết quả.

```bash
python scripts/collect_residual_feasibility.py \
  --sequence-dir /path/to/MOT17-02-FRCNN \
  --checkpoint /path/to/student_seed_0.pt \
  --library /path/to/libx265 \
  --shim /path/to/prism_x265_shim \
  --build-pin /path/to/build_pin.json \
  --source-manifest /path/to/source_manifest.json \
  --output /path/to/new_feasibility_run \
  --crf 32 --cap 1 --gop 32 \
  --intervention-start 33 --intervention-end 48 \
  --suffix-frames 16 --max-frames 64 --max-encodes 2 \
  --device cpu --validate-only
```

Sau khi kiểm tra tài nguyên và được duyệt chạy thật, thay `--validate-only` bằng `--research-only`. Nếu muốn đánh giá tracking, thêm `--weights /path/to/yolo11l.pt`; cần cả GT của chuỗi. Không có detector hoặc GT thì task metrics là unavailable, không bịa score. Nếu weights được chỉ định nhưng file không tồn tại, preflight bị chặn.

### 9.5. Artifact và số đo

- `feasibility_report.json`: report khác schema calibration, có `kind=residual_codec_feasibility`.
- `attempts.jsonl`: đếm hai điểm vận hành và ghi cả thất bại; không retry tự động.
- `bitstreams/`: HEVC và telemetry CSV khi codec cung cấp.
- `tracks/`: chỉ tạo khi có detector/GT để chấm từ video giải mã.

Report/observations/attempts đều ghi:

- `research_only=true`;
- `calibration_status=not_established`;
- `gate_authorization=false`.

Báo cáo chứa source/checkpoint/build/config/code hashes, seed khai báo trong adapter, khoảng can thiệp và các giới hạn frame/encode. Danh sách `code_hashes` ghi các thành phần trực tiếp tham gia; không thay thế một snapshot đầy đủ hoặc lockfile môi trường.

Mỗi nhánh ghi số byte toàn bitstream, frame counts, pixel changes toàn clip và riêng intervention. Distortion dùng MSE RGB giữa frame giải mã và **ảnh RGB nguồn gốc**, tách `prefix`, `intervention`, `suffix`, và `all`. Nó bao gồm ảnh hưởng chuyển RGB/YUV và codec, không phải loss riêng do residual và không phải chất lượng tác vụ.

Đo riêng student/preprocessing; tổng encode pipeline bao gồm các thành phần đó. `decode_measure_seconds` bao gồm decode, đọc reference, tính distortion và tracking nếu bật; không coi đây là latency detector riêng.

Không sinh BD-rate từ một CRF, không tính proxy loss giả, không tuyên bố `semantic_passed` hoặc `same_codec_state`. Một clip có nhiều frame phụ thuộc không trở thành nhiều mẫu độc lập cho calibration chỉ vì collector ghi nhiều pixel/frame.

Giảm bitrate ở cùng CRF không chứng minh cải thiện rate–task. Nếu task metrics thiếu, kết luận chỉ là codec/pixel feasibility. Nếu residual sau rounding không làm đổi pixel, báo đúng zero-change thay vì gọi đó là tác động của CNN.

### 9.6. Kiểm thử và bước thực nghiệm còn thiếu

```bash
python -m pytest -q tests/test_residual.py tests/test_residual_feasibility.py tests/test_residual_evidence.py tests/test_run_bounded_residual_mot17.py tests/test_training_guards.py tests/test_video_codec.py
```

Tests dùng checkpoint/model/codec tổng hợp để kiểm tra phân luồng, không tạo bằng chứng thực nghiệm. Gate mặc định và runner có gate phải giữ nguyên các kiểm thử trước.

Sau phần code này vẫn cần checkpoint thật, video có provenance và bộ codec tương thích. Không có encoder/GPU/model thật được chạy trong lần bổ sung collector. Không tạo lại lịch sử calibration bị mất. Một protocol chứng nhận mới, proxy calibration hoặc training là công việc riêng cần thiết kế và phê duyệt.

## 10. Kiểm thử và giới hạn xác minh

Chạy tests mới cùng hồi quy tập trung từ repo:

```bash
python -m pytest -q tests/test_residual.py tests/test_residual_evidence.py tests/test_run_bounded_residual_mot17.py tests/test_training_guards.py tests/test_control_contract.py tests/test_ablation.py tests/test_m2_enhancements.py tests/test_video_codec.py
```

Có thể chạy riêng `tests/test_run_tier1b_mot17.py` để kiểm tra runner lịch sử. Module đó cũng chứa tests dispatcher không thuộc phạm vi phần bổ sung này; ghi nhận lỗi thật nếu chúng thất bại, không gọi toàn bộ suite xanh khi chỉ nhóm mới qua.

Các calibration cells và model/codec thay thế trong tests chỉ kiểm tra hành vi API và luồng điều khiển; **không phải calibration thực nghiệm của một checkpoint**.

Tests bao phủ cap-zero identity, đúng đơn vị luma, signed differences, nearest-even rounding, saturation, unchanged chroma/input, dữ liệu sai định dạng, prediction không hữu hạn, context/source mismatch, gate hết hạn/không hợp lệ, inference count, decoded-only scoring, validate-only không load runtime, cùng bảo toàn output.

Không có training, tải model/dataset, native codec integration, GPU benchmark hoặc Kaggle dispatch trong lần triển khai này. Pipeline QP và báo cáo lịch sử giữ nguyên. Các hạn chế calibration/evaluation cũ không tự được sửa chỉ vì helper và runner mới tồn tại.
