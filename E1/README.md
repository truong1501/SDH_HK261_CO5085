# E1 — Phân loại ảnh: softmax, MLP, CNN

Dataset: **Fashion-MNIST** (28×28 xám, 10 lớp), dùng chung cho E1–E3.

## Môi trường

PyTorch hiện chưa hỗ trợ Python 3.14. Hai cách chạy:

- **Colab (khuyến nghị):** mở `notebooks/E1_Colab.ipynb`, bật GPU (Runtime > Change runtime
  type > GPU). Torch/torchvision đã có sẵn trên Colab. Cần commit + push nhánh chứa `E1/`
  lên GitHub trước, vì notebook `git clone` repo từ GitHub chứ không đọc máy local.
- **Local venv** (Python 3.10–3.12):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cấu trúc

```
E1/
├── requirements.txt
├── configs/          # 1 file config cho mỗi model (softmax / mlp / cnn)
├── splits/           # index train/validation (sinh tự động, cố định seed 42)
├── src/
│   ├── data.py        # Dataset/DataLoader, split, transform
│   ├── models.py       # SoftmaxClassifier, MLPClassifier, CNNClassifier
│   ├── train.py        # training loop tự viết (forward/loss/backward/step)
│   └── evaluate.py     # accuracy, confusion matrix, so sánh 3 model
├── notebooks/        # notebook chạy thử / EDA
└── results/          # log, checkpoint, biểu đồ (không commit checkpoint lớn)
```

## Chạy

```bash
# Sinh split (chỉ cần chạy 1 lần, tự tải Fashion-MNIST về E1/data/)
python -m src.data

# Huấn luyện từng model
python -m src.train --model softmax
python -m src.train --model mlp
python -m src.train --model cnn

# So sánh 3 model đã train
python -m src.evaluate
```

## Kết quả sinh ra (`results/`)

- `results/<model>/history.csv`, `summary.json`, `best.pt` — log, cấu hình và checkpoint tốt
  nhất (theo validation accuracy) của từng model.
- `results/comparison_table.{csv,json}` — bảng so sánh accuracy, macro-F1, số tham số, thời
  gian train của cả 3 model (dùng trực tiếp cho báo cáo).
- `results/loss_accuracy_curves.png` — đường cong loss/accuracy train & val của 3 model.
- `results/confusion_matrix_<model>.png`, `results/misclassified_<model>.png` — phân tích lỗi.

## Dataset & split

- Train chính thức (60.000 ảnh, cân bằng 6.000/lớp) → tách 500 ảnh/lớp (5.000 ảnh) làm
  validation, stratified, seed 42; còn lại 55.000 ảnh để train.
- Test giữ nguyên 10.000 ảnh chính thức, chỉ đánh giá một lần.
- Chuẩn hóa: `mean=0.2860, std=0.3530`. Train dùng thêm `RandomCrop(28, padding=2)`;
  **không** dùng flip ngang vì nhiều lớp (giày, dép, túi) có hướng cố định.
- Index split lưu tại `splits/train_idx.npy`, `splits/val_idx.npy` để tái lập giữa các
  thành viên và giữa E1/E2/E3.
