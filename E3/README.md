# E3 — LSTM / GRU trên chuỗi ảnh

Dùng lại đúng dataset, split và transform của E1 (Fashion-MNIST, 28×28 xám, 10
lớp). Không tự tải dữ liệu hay chia split riêng; `src/data.py` import trực
tiếp từ `../E1/src/data.py`.

## Hai cách biến ảnh thành chuỗi

- **row**: mỗi hàng pixel là một bước thời gian, 28 bước, mỗi bước 28 chiều.
- **patch**: chia ảnh thành lưới patch 4×4 không chồng lấp, quét theo hàng,
  49 bước, mỗi bước 16 chiều.

Xem `src/sequences.py`.

## Mô hình

`LSTMClassifier` và `GRUClassifier`: 1 tầng recurrent (`nn.LSTM`/`nn.GRU`),
hidden size 128, lấy hidden state ở bước cuối cùng, qua `Linear(128, 10)`.

## Baseline

`src/baseline.py` đánh giá lại MLP và CNN đã train ở E1 trên cùng test set,
**không train lại**, dùng trực tiếp checkpoint `../E1/results/{mlp,cnn}/best.pt`.

## Môi trường

Giống E1: Python 3.10–3.12 hoặc chạy trên Colab (xem
`notebooks/E3_Colab.ipynb`). Cần đã có sẵn checkpoint của E1 (chạy E1 trước
khi chạy baseline của E3).

```bash
pip install -r requirements.txt
```

## Chạy

```bash
# Kiểm tra nhanh shape của sequences và models
python -m src.sequences
python -m src.models

# Huấn luyện 4 run: lstm/gru x row/patch
python -m src.train --model lstm --seq row
python -m src.train --model gru  --seq row
python -m src.train --model lstm --seq patch
python -m src.train --model gru  --seq patch

# Đánh giá lại baseline MLP/CNN của E1 trên cùng test set
python -m src.baseline

# Tổng hợp bảng so sánh (4 run + 2 baseline) và biểu đồ
python -m src.evaluate
```

## Kết quả sinh ra (`results/`)

- `results/<model>_<seq>/history.csv`, `summary.json`, `best.pt` — log, cấu
  hình và checkpoint tốt nhất của từng run recurrent.
- `results/baseline_table.json` — kết quả đánh giá lại MLP/CNN từ E1.
- `results/comparison_table.{csv,json}` — bảng tổng hợp cả 6 dòng (4 run +
  2 baseline): số tham số, val accuracy, test accuracy, thời gian train.
- `results/loss_accuracy_curves.png` — đường cong loss/accuracy của 4 run
  recurrent.
