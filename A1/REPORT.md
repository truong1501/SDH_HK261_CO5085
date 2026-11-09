# A1.2: CNN và Transformer trên Food-101

Nguyễn Xuân Trường (2570526), Trần Lê Hoàng Long (2570443). CO5085, HK261.

Báo cáo đầy đủ: [reports/A1.2_Report.pdf](../reports/A1.2_Report.pdf) (nguồn [A1.2_Report.tex](../reports/A1.2_Report.tex), biên dịch bằng XeLaTeX). File này chỉ tóm tắt kết quả.

## Kết quả

ResNet-50 (CNN) và DeiT-Small (ViT thuần), mỗi mô hình bốn chế độ, theo [protocol A1.1](../reports/A1.1_Dataset-Protocol.pdf). Bảng ghi top-1 trên test (%), dạng mean ± std qua 3 seed; from-scratch chạy 1 seed.

| Chế độ | ResNet-50 | DeiT-S | LR backbone (ResNet / DeiT) |
|---|---|---|---|
| From-scratch | **82,48** | 67,13 | 5e-4 / 5e-4 |
| Freeze toàn bộ | 71,06 ± 0,05 | **72,79 ± 0,09** | — |
| Freeze một phần | 81,77 ± 0,12 | **87,25 ± 0,14** | 3e-4 / 1e-4 |
| Full finetune | 85,65 ± 0,01 | **88,62 ± 0,10** | 3e-4 / 1e-4 |

- **Pretrain:** giúp DeiT-S thêm 21,5 điểm, ResNet-50 thêm 3,2 điểm. Train từ đầu trên 68 nghìn ảnh, ResNet-50 hơn DeiT-S 15,4 điểm. Có pretrain thì DeiT-S tốt hơn ở cả ba chế độ (McNemar, p < 1e-8).
- **Mức freeze:**
  - Freeze một phần của DeiT-S giữ 98,5% accuracy của full finetune với một nửa số tham số trainable và gần một nửa VRAM.
  - Freeze một phần không nhanh hơn full finetune. Mọi chế độ có backbone train được đều mất khoảng 66 giây/epoch, vì thời gian bị chặn bởi khâu đọc ảnh và augmentation trên CPU.
- **Dò LR:** ResNet-50 chọn 3e-4, DeiT-S chọn 1e-4 (theo validation). Nếu ResNet-50 dùng 1e-4 thì test top-1 thấp hơn khoảng 3,5 điểm.
- **Lỗi:**
  - Hai mô hình nhầm gần như cùng các cặp lớp: steak/filet mignon, chocolate cake/mousse, beef/tuna tartare.
  - Trong số ảnh cả hai cùng sai, hơn một nửa bị đoán thành cùng một lớp.

## Tệp

- **Tổng hợp:**
  - Bảng: [summary.md](summary.md), [summary.tex](summary.tex).
  - Số liệu từng run: [runs_all.csv](runs_all.csv). Accuracy từng lớp: [per_class.csv](per_class.csv).
  - Hình: thư mục [figures/](figures/).
- **Từng run:** `runs/<run>_s<seed>/` chứa `config.yaml`, `history.csv`, `metrics.json`, `confusion.npy` và `test_preds.npz`. Kết quả chọn LR: `runs/lr_selection.json`.
- **Chạy lại:** [notebooks/A1_Colab.ipynb](notebooks/A1_Colab.ipynb) và mã nguồn trong [src/](src/).
