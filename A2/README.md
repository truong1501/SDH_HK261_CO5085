# A2.1: Chủ đề, paper, dữ liệu và kế hoạch

## Chủ đề

Phân đoạn ngữ nghĩa (semantic segmentation): gán nhãn lớp cho từng pixel của ảnh.

## Paper

Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. *Rethinking Atrous Convolution for Semantic Image Segmentation* (DeepLabv3), 2017. [arXiv:1706.05587](https://arxiv.org/abs/1706.05587).

## Dữ liệu

Pascal VOC 2012, phần segmentation: 20 lớp đối tượng và lớp nền, gồm 1.464 ảnh train và 1.449 ảnh validation. Nhóm dùng tập train để huấn luyện và tập validation để đánh giá.

## Metric

- mIoU trên validation (metric chính).
- Dice để đối chiếu thêm.

## Kế hoạch pipeline

1. Xem dữ liệu: phân bố lớp theo pixel, kích thước ảnh, ảnh và mask mẫu.
2. Viết Dataset và DataLoader, với resize, crop và lật ngang áp dụng đồng thời cho ảnh và mask.
3. So sánh hai phương pháp:
   - U-Net huấn luyện từ đầu.
   - DeepLabv3 với backbone ResNet-50 pretrained.
4. Đánh giá cả hai trên cùng tập validation bằng mIoU, Dice, số tham số và thời gian train, kèm vài ảnh dự đoán đúng và sai.
5. Phân tích paper: phương pháp, thí nghiệm, hạn chế, và đối chiếu với kết quả của nhóm.

## Hỗ trợ AI

Phương án chủ đề, paper và kế hoạch trong file này được soạn với hỗ trợ của Claude Code.
