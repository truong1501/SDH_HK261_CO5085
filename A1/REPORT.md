# A1.2 — CNN và Transformer trên Food-101

> **Bản nháp đang chờ các thí nghiệm còn lại.** Hiện có 8 run chính đã có metrics; không lấy kết quả smoke test làm kết quả báo cáo.

Nguyễn Xuân Trường (2570526), Trần Lê Hoàng Long (2570443). CO5085, HK261.

## 1. Mục tiêu

So sánh ResNet-50 (CNN) và DeiT-Small (Transformer thuần) theo bốn chế độ: from-scratch, freeze toàn bộ backbone, freeze một phần và full finetune. Các câu hỏi gồm mức lợi của pretrain, mức freeze phù hợp và sự đánh đổi độ chính xác–chi phí. Cặp ConvNeXt/Swin chỉ là mở rộng tùy chọn, không mặc định đã thực hiện.

## 2. Dữ liệu và protocol

Theo protocol A1.1, Food-101 có 101 lớp; tập train chính thức được tách stratified seed 42 thành train/validation, giữ test chính thức cho đánh giá cuối từng run. File chỉ số đã sync có **68,175 train** và **7,575 validation**, không giao nhau. Không dùng subset trong các run chính hiện đã hoàn tất.

Ảnh 224×224, chuẩn hóa ImageNet; validation/test Resize 256 + CenterCrop 224. Run scratch dùng augmentation mạnh và Mixup/CutMix; partial/full dùng RandAugment và RandomErasing. Freeze-all trích feature một lần bằng CenterCrop để tiết kiệm tính toán, nên không dùng augmentation ngẫu nhiên như các chế độ còn lại. Đây là lựa chọn đã nêu trong hướng dẫn triển khai, cần ghi rõ khi diễn giải so sánh.

## 3. Thiết lập

| Run | Mô hình | Chế độ | Epoch theo protocol | Seeds |
|---|---|---|---:|---|
| R1 | ResNet-50 | Scratch | 100 | 42 |
| R2 | ResNet-50 | Freeze all | 30 | 42,43,44 |
| R3 | ResNet-50 | Partial | 15 | 42,43,44 |
| R4 | ResNet-50 | Full | 15 | 42,43,44 |
| R5 | DeiT-S | Scratch | 100 | 42 |
| R6 | DeiT-S | Freeze all | 30 | 42,43,44 |
| R7 | DeiT-S | Partial | 15 | 42,43,44 |
| R8 | DeiT-S | Full | 15 | 42,43,44 |

Các run hiện có xác nhận thiết bị NVIDIA A100-SXM4-80GB, BF16, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130 và timm 1.0.31. Model dùng pretrained ImageNet-1k; head mới 101 lớp. Checkpoint chọn theo validation top-1. Không chọn biến thể hoặc seed bằng điểm test.

## 4. Kết quả hiện có

Bảng này chỉ liệt kê run có `metrics.json` đã đồng bộ. Top-1, top-5, macro-F1 được nhân 100. Thời gian gồm train/validation; freeze-all cộng thời gian trích feature.

| Run | Seed | Val top-1 (%) | Test top-1 (%) | Test top-5 (%) | Macro-F1 (%) | Train (s) | Trainable |
|---|---:|---:|---:|---:|---:|---:|---:|
| R2 | 42 | 66.046 | 71.081 | 91.568 | 70.874 | 110.71 | 206,949 |
| R2 | 43 | 65.980 | 71.006 | 91.537 | 70.808 | 110.20 | 206,949 |
| R2 | 44 | 66.059 | 71.101 | 91.584 | 70.892 | 110.38 | 206,949 |
| R6 | 42 | 69.030 | 72.689 | 92.309 | 72.527 | 100.94 | 38,885 |
| R6 | 43 | 68.977 | 72.808 | 92.368 | 72.674 | 101.08 | 38,885 |
| R6 | 44 | 69.083 | 72.863 | 92.376 | 72.734 | 101.14 | 38,885 |

| R4 | 42 | 77.914 | 82.170 | 96.028 | 82.046 | 1045.39 | 23,714,981 |

| R8 | 42 | 84.634 | 88.610 | 98.036 | 88.617 | 1024.93 | 21,704,549 |

## 5. Phân tích và giới hạn

Lần kiểm tra gần nhất: R3_s42 mới có history tới epoch 4/15 và chưa có metrics cuối. Chưa có kết quả scratch R1/R5 hoặc các seed pretrained còn lại; cần tiếp tục trên môi trường có GPU. Không khẳng định hoàn thành A1.2 từ tám run hiện có.

Chưa đủ bốn chế độ của hai mô hình để kết luận về lợi ích pretrain hoặc kiến trúc tốt hơn. Bản cuối sẽ tổng hợp mean ± sample standard deviation qua ba seed cho các run pretrained, loss curves, confusion matrix và các lớp dễ nhầm.

Thời gian freeze-all ghi cả chi phí trích feature được cache dùng chung cho nhiều seed; cộng trực tiếp train_time_s của các seed sẽ tính lặp phần trích feature. Cần tách chi phí một lần và chi phí train head. Throughput là forward trên input tổng hợp batch 256, không bao gồm đọc ảnh. Không đồng nhất throughput với tốc độ huấn luyện end-to-end.

Validation lấy từ train Food-101 nên có thể có nhiễu nhãn. Một seed cho scratch chưa cho biết độ biến thiên qua lần chạy. Các augmentation và số epoch khác nhau giữa chế độ là một phần protocol; so sánh không cô lập riêng ảnh hưởng pretrain. Trọng số pretrained cùng ImageNet-1k nhưng recipe huấn luyện có thể khác, do đó không quy mọi chênh lệch cho CNN/attention.

## 6. Tái lập và minh chứng

- [Protocol A1.1](../reports/A1.1_Dataset-Protocol.pdf).
- [Notebook Colab](notebooks/A1_Colab.ipynb) và [mã nguồn](src/).
- EDA đã có: [phân bố kích thước](figures/eda_sizes.png), [ảnh mẫu](figures/eda_samples.png).
- Cấu hình, history, metrics và confusion/test predictions lưu theo `runs/<RUN>_s<seed>/`.
- Chưa có PDF hoặc video demo mới trong báo cáo Markdown này.

## 7. Hỗ trợ AI

Code ban đầu có hỗ trợ Antigravity. OpenAI Codex hỗ trợ rà soát triển khai, kiểm tra kỹ thuật và tổng hợp kết quả từ file đã đồng bộ. Nhóm chịu trách nhiệm xác minh số liệu và nội dung nộp. Kết quả smoke test không đưa vào bảng thí nghiệm chính.
