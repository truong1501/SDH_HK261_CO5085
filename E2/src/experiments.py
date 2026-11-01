"""Danh sách thí nghiệm E2 (config cố định để tái lập).

Mỗi thí nghiệm = (cách tokenize, bản MSA). Kiến trúc encoder và siêu tham số huấn luyện
giống nhau cho mọi thí nghiệm, nên khác biệt kết quả chỉ đến từ tokenizer hoặc MSA.

So sánh theo đề:
  (i)  MSA tự viết vs MSA PyTorch, cùng tokenize  -> COMPARE_ATTENTION
  (ii) các cách tokenize với nhau (cùng MSA tự viết) -> COMPARE_TOKENIZERS
"""

from __future__ import annotations

# Kien truc chung (encoder nong): D=64, 4 khoi, 4 head (d_h = 16), FFN 2x
MODEL_CFG = {"dim": 64, "depth": 4, "num_heads": 4, "mlp_ratio": 2, "dropout": 0.1}

# Huan luyen chung
TRAIN_CFG = {
    "epochs": 30,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 0.05,
    "warmup_epochs": 2,
    "seed": 42,
}

EXPERIMENTS = {
    "patch4_scratch": {"tokenizer": "patch4", "attn_impl": "scratch"},
    "patch4_torch": {"tokenizer": "patch4", "attn_impl": "torch"},
    "patch7s3_scratch": {"tokenizer": "patch7s3", "attn_impl": "scratch"},
    "row_scratch": {"tokenizer": "row", "attn_impl": "scratch"},
    "cnnstem_scratch": {"tokenizer": "cnnstem", "attn_impl": "scratch"},
}

COMPARE_ATTENTION = ["patch4_scratch", "patch4_torch"]
COMPARE_TOKENIZERS = ["patch4_scratch", "patch7s3_scratch", "row_scratch", "cnnstem_scratch"]

# Mo ta ngan, dung trong bang/bieu do
LABELS = {
    "patch4": "Patch 4x4 (49 token)",
    "patch7s3": "Patch 7x7 overlap, stride 3 (64 token)",
    "row": "Moi hang 1 token (28 token)",
    "cnnstem": "CNN-stem 7x7 (49 token)",
}
