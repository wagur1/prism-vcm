# PRISM: áp dụng có kiểm chứng từ sáu bài preprocessing

## Phạm vi

Đây là tổng hợp nghiên cứu để định hướng thí nghiệm, không phải chứng nhận ARA Level 2 hoặc tuyên bố novelty. Các cơ chế bài báo được mô tả theo nguồn đã đọc; RAPNet, P4VC và Task-Switchable mới được kiểm tra ở mức abstract/metadata, không đủ để tái tạo mọi chi tiết thuật toán.

Danh sách dùng SCImago/SJR 2025 để lọc tạp chí, không phải JCR hoặc khẳng định quartile tại năm xuất bản. Q1 của tạp chí không chứng minh một cơ chế sẽ hiệu quả trên PRISM.

## Sáu nguồn và phần có thể áp dụng

| Nguồn | Cơ chế đã xác minh | Áp dụng có điều kiện vào PRISM | Không được suy ra |
|---|---|---|---|
| [Preprocessing Enhanced Image Compression for Machine Vision, TCSVT 2024](https://doi.org/10.1109/TCSVT.2024.3441049) | Neural preprocessor theo mức nén/tác vụ; codec truyền thống ở forward và proxy ở backward. [Bản đọc](https://arxiv.org/abs/2206.05650). | Giữ kiểm tra rate–task trên codec thật; kiểm định proxy bằng khả năng dự đoán thứ hạng và hướng thay đổi của các phương án. | Proxy ảnh DCT hiện có đã mô phỏng inter-frame hoặc luôn cung cấp gradient hữu ích. |
| [Task-Switchable Pre-Processor for Image Compression for Multiple Machine Vision Tasks, TCSVT 2024](https://doi.org/10.1109/TCSVT.2023.3348995) | Đặc trưng ngữ nghĩa điều biến pixel mapping; chuyển điều biến để hỗ trợ nhiều tác vụ. | Có thể nghiên cứu adapter/task modulation sau khi chứng minh lợi ích một task. So với model riêng trên cùng dữ liệu và ngân sách. | Một model nhiều task là ý tưởng hoàn toàn mới, hoặc task switching chứng minh transfer sang task chưa thấy. |
| [P4VC, TPAMI 2026](https://doi.org/10.1109/TPAMI.2026.3727894) | Tối ưu perturbation từng mẫu trên codec, rồi học mạng tổng quát; abstract mô tả distribution-aware adversarial learning. | Teacher đề xuất candidate; đo thực để tạo giám sát. Chỉ học từ lợi ích đã được kiểm chứng trên development. | “Positive perturbation” nghĩa là mọi pixel phải tăng; +1 luma đồng đều là cơ chế bài này. |
| [RAPNet, TCSVT 2026](https://doi.org/10.1109/TCSVT.2026.3700718) | One-step reinforcement learning cho video preprocessing trước codec không khả vi, mục tiêu rate–distortion/perception. | Cân nhắc tối ưu hộp đen trong không gian điều khiển nhỏ nếu proxy không đáng tin; so với tìm kiếm có cùng ngân sách encode. | RL chắc chắn tốt hơn tìm kiếm đơn giản hoặc reward VMAF bảo đảm tracking tốt. |
| [Better Compression With Deep Pre-Editing, TIP 2021](https://doi.org/10.1109/TIP.2021.3096085) | CNN sửa nội dung trước JPEG; loss kết hợp proximity, bit budget và no-reference quality. [Abstract tác giả](https://research.google/pubs/better-compression-with-deep-pre-editing/). | Giới hạn/đo thay đổi pixel, identity control, bảo toàn nội dung; tách raw prediction và output sau rounding/clipping. | Neural residual trước codec tự nó là novelty, hoặc tăng cap sẽ cải thiện rate–task. |
| [Utility-Driven Adaptive Preprocessing for Screen Content Video Compression, TMM 2017](https://doi.org/10.1109/TMM.2016.2625276) | Đánh giá utility của vùng, temporal masking và Gaussian chọn lọc để giảm nội dung ít hữu ích. | Gaussian và edge/texture guidance là đối chứng mạnh; teacher mask không thay thế đo task utility. | Foreground luôn quan trọng như nhau, background luôn không quan trọng, hoặc Gaussian chọn lọc là ý tưởng mới. |

[A Preprocessing Framework for Video Machine Vision under Compression](https://doi.org/10.1109/DCC58796.2024.00039) của Zhao và cộng sự là công trình hội nghị DCC 2024, không đưa vào danh sách journal Q1. Bài này đã có neural spatial–temporal preprocessing, virtual codec và rate–distortion–task optimization.

## Những gì PRISM đã đo, và chưa đo

Lượt feasibility được thực hiện trên checkpoint seed 0 có SHA256 `623b774ba216e7c0e39ce233f7e2cab78f5448acaa2f3ec01e47fe3ffd32c345`, bảy chuỗi MOT17, 64 frame/chuỗi, CRF32, cap1, can thiệp frame33–48. Mỗi chuỗi có hai encode.

- Cả bảy chuỗi cho pixel delta bằng +1 trên toàn bộ vùng can thiệp.
- Prediction vượt cap tại toàn bộ pixel được báo cáo; điều đó không tự chứng minh raw logits/feature map là hằng số.
- Tổng byte tăng khoảng 0.0373%; trung bình chênh lệch HOTA theo chuỗi là −0.1263 điểm.
- Đây là so sánh cùng CRF, không phải BD-rate hoặc bằng chứng tổng quát về residual preprocessing.
- Các chuỗi đã được sử dụng để lựa chọn hướng nghiên cứu. Không được gọi chúng là một test set mới chưa xem.

## Thứ tự quyết định

### E1 — Raw-output diagnosis, trước khi train lại

Giữ checkpoint cố định, chỉ forward một tập frame/CRF đã khai báo:

- Logits trước `tanh`, đầu ra residual sau `tanh`, sau resize và sau cap.
- Min/max, mean/std, phân vị, tỷ lệ gần ±8 và tỷ lệ vượt ±cap.
- QP raw field và mức khác biệt của các phép chiếu `[-6,+6]`, `[-5,+5]`, `[-4,+6]`, compensation ON/OFF.
- Chuẩn hóa input, dtype, phiên bản thư viện và checkpoint/source hashes.

Diễn giải:

1. Logits/residual gần hằng cực đại ở nhiều input: có bằng chứng về bão hòa đầu ra trong điều kiện đo, chưa xác định nguyên nhân huấn luyện.
2. Raw residual biến thiên nhưng đều lớn hơn cap: clipping làm mất biến thiên; không được gọi raw head là hằng số.
3. QP projections trùng nhau: ablation chưa kích hoạt khác biệt trên những input đó, không phải bằng chứng encoder bỏ qua quantOffsets.

Không chọn lại cap hoặc sửa weights sau khi thấy vài frame đẹp. Không chạy thêm encode để xác nhận lại một phép biến đổi đã biết chính xác là +1.

### E2 — Tách lợi ích bất đối xứng

Giữ QP bất đối xứng là hướng nghiên cứu chính mà người dùng đã chọn. Residual là nhánh thay thế, chưa ghép hai cơ chế.

Nếu raw-output diagnosis cho thấy bản đồ điều khiển có biến thiên sử dụng được, so sánh giới hạn đối xứng lịch sử, đối xứng cùng độ rộng và bất đối xứng; tách compensation. Dùng precision chung, không prefilter, cùng nguồn/config và vùng rate–quality hỗ trợ.

Mục tiêu là phân biệt bất đối xứng với thay đổi độ rộng, mean shift và saturation. Chỉ đổi khoảng chặn không đủ làm đóng góp khoa học.

### E3 — Teacher-value ablation

Chỉ sau khi xác định controller có tác động hữu ích, so sánh:

- Heuristic không teacher nền tảng.
- SAM2-only.
- DINOv2-only.
- Hướng dẫn kết hợp.

Giữ tương đương candidate count, oracle encode budget, dữ liệu train và số seed. Mask đẹp hoặc oracle tốt trên train chưa chứng minh student có ích trên dữ liệu độc lập.

## Những phần chưa triển khai trong lượt chẩn đoán

- Không train thêm GRU, adapter, residual head hoặc backbone.
- Không chuyển hướng dẫn task sang “task-agnostic” chỉ bằng cách bỏ detector khỏi runtime.
- Không giả lịch sử calibration hoặc đổi nhãn CRF thành effective QP.
- Không gán kết quả inference-only thành cải thiện nén/tracking.
- Không gọi tổng hợp các thành phần đã biết là novelty trước khi rà soát tài liệu và có đối chứng.

## Câu hỏi nghiên cứu còn mở

“Hướng dẫn foundation model, khi được đánh giá bằng chi phí codec thật và task utility, có giúp học phân bổ QP bất đối xứng hiệu quả hơn heuristic và đối xứng không?”

Đây là giả thuyết, không phải kết quả. Nếu bất đối xứng không vượt đối chứng, phải giữ kết quả âm và điều chỉnh hướng nghiên cứu, không điều chỉnh protocol để ép thắng.
