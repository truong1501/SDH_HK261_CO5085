"""Các cách biến ảnh (B, C, H, W) thành chuỗi token (B, N, D) cho E2.

- PatchTokenizer : chia ảnh thành ô lưới p x p (stride s). s = p -> patch không chồng lấn
                   (kiểu ViT); s < p -> patch chồng lấn (overlap). Mỗi patch flatten rồi
                   chiếu tuyến tính lên D chiều.
- RowTokenizer   : mỗi hàng pixel là một token (N = H = 28).
- CNNStemTokenizer: CNN nhỏ rút feature map (D, 7, 7); mỗi vị trí không gian là một token.

Mỗi tokenizer có thuộc tính `num_tokens` (để tạo positional embedding) và `grid`
(hình dạng lưới (gh, gw) của token, dùng khi vẽ attention map; None nếu không có lưới 2D).

Ghi chú: token theo từng pixel (N = 784) không dùng vì attention có chi phí O(N^2):
784^2 ≈ 615k cặp/ảnh/head, quá nặng so với lợi ích trên ảnh 28x28.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IMG_SIZE = 28
IN_CHANNELS = 1


class PatchTokenizer(nn.Module):
    def __init__(
        self,
        dim: int,
        patch_size: int = 4,
        stride: int | None = None,
        in_channels: int = IN_CHANNELS,
        img_size: int = IMG_SIZE,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride or patch_size
        g = (img_size - patch_size) // self.stride + 1  # so patch moi chieu
        self.grid = (g, g)
        self.num_tokens = g * g
        # Moi patch co C*p*p gia tri -> Linear len D chieu (patch embedding)
        self.proj = nn.Linear(in_channels * patch_size * patch_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.unfold cat anh thanh cac patch: (B, C*p*p, N), moi cot la 1 patch da flatten
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        patches = patches.transpose(1, 2)  # (B, N, C*p*p)
        return self.proj(patches)          # (B, N, D)


class RowTokenizer(nn.Module):
    def __init__(self, dim: int, in_channels: int = IN_CHANNELS, img_size: int = IMG_SIZE):
        super().__init__()
        self.num_tokens = img_size
        self.grid = None  # chuoi 1D theo chieu doc
        self.proj = nn.Linear(in_channels * img_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        rows = x.permute(0, 2, 1, 3).reshape(B, H, C * W)  # (B, H, C*W): hang h la token h
        return self.proj(rows)                              # (B, H, D)


class CNNStemTokenizer(nn.Module):
    """2 khoi Conv-BN-ReLU-MaxPool (giong CNN cua E1): 28x28 -> 7x7, so kenh = D."""

    def __init__(self, dim: int, in_channels: int = IN_CHANNELS, img_size: int = IMG_SIZE):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        g = img_size // 4
        self.grid = (g, g)
        self.num_tokens = g * g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.stem(x)                     # (B, D, 7, 7)
        return fmap.flatten(2).transpose(1, 2)  # (B, 49, D): moi vi tri (i, j) la 1 token


# ten -> ham tao tokenizer voi so chieu D
TOKENIZER_REGISTRY = {
    "patch4": lambda dim: PatchTokenizer(dim, patch_size=4),             # 7x7 = 49 token
    "patch7s3": lambda dim: PatchTokenizer(dim, patch_size=7, stride=3),  # overlap, 8x8 = 64 token
    "row": lambda dim: RowTokenizer(dim),                                 # 28 token
    "cnnstem": lambda dim: CNNStemTokenizer(dim),                         # 7x7 = 49 token
}


def build_tokenizer(name: str, dim: int) -> nn.Module:
    if name not in TOKENIZER_REGISTRY:
        raise ValueError(f"Unknown tokenizer '{name}', available: {list(TOKENIZER_REGISTRY)}")
    return TOKENIZER_REGISTRY[name](dim)
