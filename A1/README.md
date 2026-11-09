# A1: CNN vs Transformer trên Food-101

So sánh **ResNet-50 (CNN)** và **DeiT-Small (ViT)** trên Food-101 (101 lớp) theo 4 chế độ: from-scratch, freeze toàn bộ backbone, freeze một phần, full finetune. Protocol chi tiết: [A1.1_Dataset-Protocol.pdf](../reports/A1.1_Dataset-Protocol.pdf).

## Cấu trúc

```
A1/
├── notebooks/A1_Colab.ipynb   # chạy toàn bộ trên Colab A100
├── src/
│   ├── config.py              # mô hình, chế độ, siêu tham số, R1-R8 / X1-X8
│   ├── data.py                # split stratified, augmentation, DataLoader
│   ├── models.py              # timm, freeze, BatchNorm eval, nhóm tham số AdamW
│   ├── engine.py              # vòng lặp train/eval tự viết, warmup + cosine, metric
│   ├── train.py               # chạy run, resume, ghi log/metric
│   └── summarize.py           # bảng tổng hợp + biểu đồ + phân tích lỗi
├── splits/                    # chỉ số train/val (seed 42) + tên lớp
├── runs/<RUN>_s<seed>/        # config.yaml, history.csv, metrics.json, confusion.npy, test_preds.npz
├── figures/                   # biểu đồ cho báo cáo
└── requirements.txt
```

## Chạy

**Google Colab (cách chính):** mở [A1_Colab.ipynb](notebooks/A1_Colab.ipynb) trên runtime A100, chạy lần lượt cell 1 → 6. Notebook mount Drive, copy Food-101 về `/content/data`, in trạng thái từng run rồi train hàng đợi còn thiếu **trong kernel** (log từng epoch). Mất kết nối hoặc Interrupt: chạy lại cell 1, 2, 3, 5, 6 để resume. Cell 8 sinh bảng và toàn bộ hình báo cáo, cell 9 flush Drive.

Thứ tự chạy nằm trong `src/plan.py`. Theo protocol mục 7, notebook dò LR backbone {5e-5, 1e-4, 3e-4} với full finetune seed 42, chọn theo validation cho từng mô hình (`runs/lr_selection.json`), rồi dùng LR đó cho R3/R7 và seed 43/44 của R4/R8. Run ở LR khác 1e-4 có hậu tố `_lr` trong tên thư mục; `summarize` chỉ đưa run ở LR đã chọn vào bảng chính và vẽ thêm `figures/lr_sweep.png`.

JupyterHub ([A1_JupyterHub.ipynb](notebooks/A1_JupyterHub.ipynb), [SERVER.md](SERVER.md)) là phương án tuỳ chọn.

Dòng lệnh tương đương, trong thư mục `A1`:

```bash
python -m src.train --run R2 R6 --data_root /content/data      # freeze toàn bộ, 3 seed
python -m src.train --run R4 R8 R3 R7 --seeds 42                # finetune
python -m src.train --run R1 R5                                 # from-scratch 100 epoch
python -m src.summarize --out_root runs --fig_dir figures --data_root /content/data
```

Run đã có `metrics.json` sẽ được bỏ qua; run dở sẽ resume từ `last.pt`.

## Ma trận thí nghiệm

| ID | Mô hình | Chế độ | Phần được train | Epoch | LR backbone / head | Augmentation |
|---|---|---|---|---|---|---|
| R1 / R5 | ResNet-50 / DeiT-S | From-scratch | toàn bộ | 100 | 5e-4 / 5e-4 | Crop, Flip, RandAugment, RandomErasing, Mixup/CutMix, label smoothing |
| R2 / R6 | ResNet-50 / DeiT-S | Freeze toàn bộ | head | 30 | — / 1e-3 | feature cache (xem dưới) |
| R3 / R7 | ResNet-50 / DeiT-S | Freeze một phần | layer4 / block 6–11 + head | 15 | 1e-4 / 1e-3 | Crop, Flip, RandAugment, RandomErasing |
| R4 / R8 | ResNet-50 / DeiT-S | Full finetune | toàn bộ | 15 | 1e-4 / 1e-3 | Crop, Flip, RandAugment, RandomErasing |

Chung: AdamW, warmup + cosine, batch 256, bf16 (fp16 nếu GPU không hỗ trợ bf16), trọng số ImageNet-1k từ timm. Checkpoint chọn theo val top-1; test chỉ đánh giá một lần trên checkpoint đó.

## Hình và bảng cho báo cáo

`python -m src.summarize --out_root runs --splits_dir splits --fig_dir figures --data_root <DATA_ROOT>` (cell 6 notebook JupyterHub) sinh:

| File | Dùng cho phần báo cáo |
|---|---|
| `figures/mode_compare.png` | Hình chính: top-1/macro-F1 CNN vs ViT theo 4 chế độ, std + từng seed |
| `figures/curves_by_mode.png` | Train vs val loss/acc từng chế độ, epoch được chọn theo val (overfit/underfit) |
| `figures/pretrain_gap.png` | Lợi ích pretrain: from-scratch vs pretrained tốt nhất |
| `figures/tradeoff.png`, `figures/efficiency.png` | Accuracy vs tham số/thời gian; tham số trainable, s/epoch, VRAM, throughput |
| `figures/agreement.png` + bảng McNemar trong `summary.md` | CNN và ViT đúng/sai trên cùng ảnh; khác biệt có ý nghĩa thống kê không |
| `figures/per_class.png`, `per_class.csv` | Accuracy từng lớp, lớp CNN/ViT hơn nhau nhiều nhất |
| `figures/confusion_<model>.png` + top cặp nhầm trong `summary.md` | Phân tích lỗi |
| `figures/misclassified_<model>.png` | Ảnh test bị đoán sai (cần `--data_root`) |
| `summary.md`, `summary.tex`, `summary.csv` | Bảng tổng hợp (Markdown / LaTeX booktabs / CSV) |

Run tốt nhất mỗi mô hình được chọn theo validation, không theo test. `figures/eda_*.png` là hình EDA dữ liệu đã có.

## Ghi chú cài đặt

- **Freeze toàn bộ (R2, R6):** trích feature một lần rồi train head trên feature (protocol mục 9). Vì vậy ảnh train ở chế độ này dùng Resize 256 + CenterCrop 224, không có Crop/Flip ngẫu nhiên như bảng augmentation ở mục 4 của protocol. Cần nêu điểm này trong báo cáo.
- **BatchNorm:** các lớp BatchNorm bị đóng băng giữ ở `eval()`.
- **Thời gian train** gồm cả đánh giá validation mỗi epoch; với R2/R6 gồm thời gian trích feature.
- **Throughput** đo bằng forward trên batch tổng hợp 256 ảnh, không tính thời gian đọc ảnh.
- Mọi thay đổi so với protocol (ví dụ giảm epoch bằng `--epochs`) được ghi vào trường `notes` trong `metrics.json`.

## Hỗ trợ AI

Code pipeline A1 được viết với hỗ trợ của trợ lý AI (Antigravity, OpenAI Codex), dựa trên protocol A1.1 của nhóm.

## Chuẩn bị trước khi chạy chính thức

Notebook cài đầy đủ requirements và hiện log pip; lệnh train dừng khi có lỗi. Chạy tests trên ảnh tổng hợp trước khi tải/train Food-101. Các tests không tải pretrained weights và không chứng minh độ chính xác trên dữ liệu thật.

Cache feature phân biệt model, resolution, đường dẫn dữ liệu và nội dung split. Khi thay đổi dữ liệu ngay trong cùng thư mục, dùng out_root mới. Run đang có config khác sẽ bị từ chối để tránh trộn kết quả. Muốn train lại từ đầu, chọn out_root khác; --force không xóa checkpoint cũ.

Resume standard khôi phục model, optimizer, scaler và RNG CPU/Python/NumPy/CUDA tại cuối epoch; không bảo đảm bit-for-bit với augmentation trong worker. Freeze-all train lại head nếu chưa hoàn tất, tái sử dụng cache feature. Các biểu đồ chọn biến thể bằng validation, không chọn bằng điểm test.

## Kiểm chứng ngày 07/11/2026

3 tests đạt trên CPU: ResNet-50/DeiT-S thật chạy forward/backward và freeze đúng (không tải pretrained), validation split/config, train/checkpoint/resume cùng nhánh feature-cache với model nhỏ và dữ liệu tổng hợp. Cú pháp toàn bộ source và code cells notebook đạt. Chưa tải/train Food-101, chưa benchmark A100 hoặc kiểm chứng accuracy; số liệu test tổng hợp không dùng làm kết quả A1.2.
