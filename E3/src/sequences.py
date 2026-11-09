"""Bien anh Fashion-MNIST (1x28x28) thanh chuoi cho LSTM/GRU.

Hai cach bieu dien:
- row: moi hang pixel la mot buoc thoi gian (28 buoc, 28 chieu/buoc).
- patch: chia anh thanh luoi patch khong chong lap, quet theo hang
  (voi patch 4x4: 49 buoc, 16 chieu/buoc).
"""

from __future__ import annotations

import torch

ROW_INPUT_SIZE = 28
ROW_SEQ_LEN = 28

PATCH_SIZE = 4
PATCH_INPUT_SIZE = PATCH_SIZE * PATCH_SIZE  # 16
PATCH_SEQ_LEN = (28 // PATCH_SIZE) ** 2  # 49

SEQUENCE_CONFIG = {
    "row": {"input_size": ROW_INPUT_SIZE, "seq_len": ROW_SEQ_LEN},
    "patch": {"input_size": PATCH_INPUT_SIZE, "seq_len": PATCH_SEQ_LEN},
}


def to_row_sequence(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, 28, 28) -> (B, 28, 28): moi buoc thoi gian la mot hang pixel."""
    return x.squeeze(1)


def to_patch_sequence(x: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """(B, 1, 28, 28) -> (B, so_patch, patch_size*patch_size), quet patch theo hang."""
    b, c, h, w = x.shape
    nh, nw = h // patch_size, w // patch_size
    x = x.view(b, c, nh, patch_size, nw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(b, nh * nw, c * patch_size * patch_size)


SEQUENCE_FUNCS = {
    "row": to_row_sequence,
    "patch": to_patch_sequence,
}


def to_sequence(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind not in SEQUENCE_FUNCS:
        raise ValueError(f"Unknown sequence kind '{kind}', available: {list(SEQUENCE_FUNCS)}")
    return SEQUENCE_FUNCS[kind](x)


if __name__ == "__main__":
    dummy = torch.randn(4, 1, 28, 28)
    for kind, cfg in SEQUENCE_CONFIG.items():
        seq = to_sequence(dummy, kind)
        ok = tuple(seq.shape) == (4, cfg["seq_len"], cfg["input_size"])
        print(f"{kind:6s} -> shape={tuple(seq.shape)} expected=(4, {cfg['seq_len']}, {cfg['input_size']}) ok={ok}")
