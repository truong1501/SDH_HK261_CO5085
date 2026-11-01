# E2 — Transformer: Multi-Head Self-Attention (MSA) và tokenization

Dataset: **Fashion-MNIST**, dùng **đúng split của E1** (seed 42; train 55k / validation 5k / test 10k).
`src/data.py` gọi lại `E1/src/data.py`, nên E1, E2, E3 luôn dùng cùng một split.

## Ý tưởng

Một mô hình Transformer phân loại ảnh gồm ba phần:

1. **Tokenizer**: biến ảnh thành một chuỗi N vector (token), mỗi vector có D chiều.
2. **Encoder**: L khối Transformer. Mỗi khối có **MSA** (*Multi-Head Self-Attention*, mỗi token
   "nhìn" mọi token khác và tổng hợp thông tin) và **FFN** (*Feed-Forward Network*, một MLP nhỏ áp lên
   từng token). Cả hai có **LayerNorm** phía trước và kết nối tắt (residual).
3. **Head**: thêm một token đặc biệt **CLS** (*classification token*) vào đầu chuỗi. Sau encoder,
   vector của CLS đi qua một lớp Linear để ra 10 lớp. Vì attention không biết thứ tự token, mô hình cộng
   thêm **positional embedding** (vector vị trí học được) cho mỗi token.

Đề yêu cầu hai bản MSA để đối chiếu:

- **MSA tự viết** (`ScratchMSA`): chỉ dùng `nn.Linear`, `torch.einsum`, `softmax` và phép ghép head,
  không dùng `nn.MultiheadAttention`.
- **MSA PyTorch** (`TorchMSA`): bọc `nn.MultiheadAttention`.

Hai bản có cùng giao diện, nên mô hình chỉ khác nhau ở đúng một module này.
`src/check_msa.py` chứng minh bản tự viết đúng: nó copy trọng số của bản PyTorch sang bản tự viết, cho
cùng input, rồi so output, attention và gradient. Sai lệch chỉ ở mức làm tròn float32.

Bốn cách lấy token được cài đặt như sau:

| Tên | Cách làm | Số token N |
|---|---|---|
| `patch4` | Chia lưới ô 4×4 không chồng lấn (kiểu ViT), flatten mỗi ô rồi qua Linear | 49 |
| `patch7s3` | Biến thể: ô 7×7 chồng lấn, bước (stride) 3 | 64 |
| `row` | Mỗi hàng pixel là một token | 28 |
| `cnnstem` | CNN 2 tầng (giống CNN của E1) rút feature map 7×7, mỗi vị trí là một token | 49 |

Không dùng token theo từng pixel (N = 784), vì chi phí attention tăng theo N²: khoảng 615 nghìn cặp
cho mỗi ảnh và mỗi head.

Encoder dùng chung cho mọi thí nghiệm: D = 64, 4 khối, 4 head, FFN rộng 2×, dropout 0.1.
Huấn luyện: AdamW (lr 1e-3, weight decay 0.05), 2 epoch warmup rồi giảm lr theo cosine, 30 epoch,
batch 128, seed 42. Tất cả nằm trong `src/experiments.py`.

## Thí nghiệm và so sánh

Mỗi thí nghiệm là một cặp (tokenizer, bản MSA). Hai phép so sánh theo đề:

- **(i) MSA tự viết vs PyTorch**: `patch4_scratch` vs `patch4_torch`.
- **(ii) các cách tokenize**: `patch4_scratch`, `patch7s3_scratch`, `row_scratch`, `cnnstem_scratch`.

## Cấu trúc

```
E2/
├── requirements.txt
├── src/
│   ├── attention.py    # ScratchMSA (tự viết) + TorchMSA (nn.MultiheadAttention)
│   ├── tokenizers.py   # patch / patch chồng lấn / hàng / CNN-stem
│   ├── models.py       # khối Transformer (pre-norm) + bộ phân loại có token CLS
│   ├── experiments.py  # config cố định của 5 thí nghiệm
│   ├── data.py         # dùng lại split E1; chọn device cuda/mps/cpu
│   ├── check_msa.py    # kiểm chứng MSA tự viết == nn.MultiheadAttention
│   ├── train.py        # training loop tự viết (forward/loss/backward/step), có --resume
│   └── evaluate.py     # test accuracy, F1, thời gian, bảng + biểu đồ + attention map
├── notebooks/E2_Colab.ipynb
└── results/            # log, best.pt, bảng, biểu đồ (last.pt không commit)
```

## Môi trường

- **Colab (khuyến nghị):** mở `notebooks/E2_Colab.ipynb` và bật GPU. Có hai cách đưa code lên:
  `SOURCE = "zip"`: nén `E1/src` và `E2/` rồi upload, không cần push.
  ```bash
  # chạy ở thư mục gốc repo
  zip -r ../code_E1_E2.zip E1/src E2 -x "*/results/*" "*__pycache__*"
  ```
  `SOURCE = "github"`: notebook `git clone` repo, cần push code trước.
- **Máy cá nhân** (Python 3.10–3.12; Mac tự dùng GPU Apple qua MPS):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r E2/requirements.txt
```

## Chạy (từ thư mục `E2/`)

```bash
python -m src.check_msa                      # 1. kiểm tra MSA tự viết
python -m src.models                         # 2. in số token + số tham số mỗi cấu hình
python -m src.train --exp patch4_scratch --subset 2000 --epochs 1   # 3. chạy thử nhanh

# 4. huấn luyện đủ 5 thí nghiệm
for e in patch4_scratch patch4_torch patch7s3_scratch row_scratch cnnstem_scratch; do
  python -m src.train --exp $e
done

python -m src.evaluate                       # 5. bảng + biểu đồ vào results/
```

Các tùy chọn hữu ích: `--device cpu|cuda|mps`, `--data-dir <thư mục dữ liệu>`,
`--ckpt-dir <vd Google Drive>` (nơi lưu `last.pt`), `--resume` (train tiếp nếu Colab bị ngắt).

## Kết quả sinh ra (`results/`)

- `<exp>/history.csv`, `summary.json`, `best.pt`: log từng epoch (có thời gian mỗi epoch), cấu hình,
  checkpoint tốt nhất theo validation accuracy.
- `msa_check.json`: sai lệch giữa MSA tự viết và PyTorch.
- `comparison_table.{csv,json}`, `table_attention.csv`, `table_tokenizers.csv`: số liệu cho báo cáo.
- `curves_attention.png`, `curves_tokenizers.png`: đường cong loss và accuracy.
- `confusion_matrix_<exp>.png`, `attention_maps.png`: phân tích lỗi và vùng ảnh mà token CLS chú ý.
