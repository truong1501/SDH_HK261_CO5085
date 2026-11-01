"""Hai bản Multi-Head Self-Attention (MSA) có cùng giao diện cho E2.

- ScratchMSA: tự viết từ phép toán cơ bản (Linear, einsum, softmax, ghép head).
  KHÔNG dùng nn.MultiheadAttention / nn.TransformerEncoder*.
- TorchMSA:   bọc nn.MultiheadAttention của PyTorch, dùng làm bản đối chiếu.

Cả hai nhận x có shape (B, N, D) và trả về (out, attn):
    out  : (B, N, D)
    attn : (B, h, N, N) nếu return_attn=True, ngược lại None.

Công thức (Vaswani et al., 2017), với mỗi head i:
    Q_i = X W_q^i,  K_i = X W_k^i,  V_i = X W_v^i        (d_h = D / h)
    head_i = softmax(Q_i K_i^T / sqrt(d_h)) V_i
    MSA(X) = concat(head_1, ..., head_h) W_o
"""

from __future__ import annotations

import torch
from torch import nn


class ScratchMSA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} phai chia het cho num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(d_h)

        # Gop 3 phep chieu Q, K, V vao 1 Linear (D -> 3D) cho nhanh.
        # Thu tu trong trong so: [W_q; W_k; W_v], moi khoi D hang, moi khoi lai chia
        # thanh h head lien tiep -> giong het cach nn.MultiheadAttention luu in_proj_weight,
        # nen co the copy trong so qua lai de kiem tra (xem check_msa.py).
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)  # W_o
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, N, D = x.shape
        h, dh = self.num_heads, self.head_dim

        # 1) Chieu tuyen tinh -> Q, K, V, roi tach head
        qkv = self.qkv(x)                              # (B, N, 3D)
        qkv = qkv.view(B, N, 3, h, dh)                 # tach 3 khoi, moi khoi h head
        qkv = qkv.permute(2, 0, 3, 1, 4)               # (3, B, h, N, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]               # moi cai (B, h, N, dh)

        # 2) Diem tuong dong giua moi cap token i, j: q_i . k_j / sqrt(dh)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale  # (B, h, N, N)

        # 3) Softmax theo chieu j (cac token duoc "nhin"): moi hang tong = 1
        attn = scores.softmax(dim=-1)
        attn_dropped = self.attn_drop(attn)

        # 4) Tong co trong so cac vector value
        out = torch.einsum("bhij,bhjd->bhid", attn_dropped, v)  # (B, h, N, dh)

        # 5) Ghep head: (B, h, N, dh) -> (B, N, h*dh) = (B, N, D), roi chieu W_o
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj_drop(self.out_proj(out))
        return out, (attn if return_attn else None)


class TorchMSA(nn.Module):
    """Ban doi chieu: dung nn.MultiheadAttention (self-attention: query = key = value = x)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # need_weights=False cho phep PyTorch dung kernel fused (scaled_dot_product_attention)
        # -> nhanh hon; day la mot ly do khi so sanh thoi gian voi ban tu viet.
        out, attn = self.mha(
            x, x, x, need_weights=return_attn, average_attn_weights=False
        )
        return self.proj_drop(out), attn


ATTN_REGISTRY = {"scratch": ScratchMSA, "torch": TorchMSA}


def build_msa(impl: str, dim: int, num_heads: int, **kwargs) -> nn.Module:
    if impl not in ATTN_REGISTRY:
        raise ValueError(f"Unknown attention '{impl}', available: {list(ATTN_REGISTRY)}")
    return ATTN_REGISTRY[impl](dim, num_heads, **kwargs)
