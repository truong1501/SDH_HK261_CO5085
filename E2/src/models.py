"""Bộ phân loại Transformer nông cho E2: tokenizer -> [CLS] + positional embedding
-> L khối Transformer encoder (pre-norm) -> LayerNorm -> head trên token CLS.

Khối encoder (pre-norm, như ViT):
    x = x + MSA(LN(x))
    x = x + FFN(LN(x))       FFN = Linear(D, r*D) -> GELU -> Linear(r*D, D)
MSA có thể là bản tự viết hoặc bản PyTorch (tham số attn_impl), phần còn lại giống hệt,
nên khác biệt giữa hai bản chỉ nằm ở MSA.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import build_msa
from .tokenizers import build_tokenizer

NUM_CLASSES = 10


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float, attn_impl: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = build_msa(attn_impl, dim, num_heads, attn_dropout=dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        attn_out, attn = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + attn_out              # residual 1
        x = x + self.ffn(self.norm2(x))  # residual 2
        return x, attn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        tokenizer: str = "patch4",
        attn_impl: str = "scratch",
        dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.tokenizer = build_tokenizer(tokenizer, dim)
        n = self.tokenizer.num_tokens

        # Token CLS hoc duoc: dau ra cua no sau encoder dai dien cho ca anh
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # Positional embedding hoc duoc cho N token + 1 CLS (attention ban than khong biet thu tu)
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_ratio, dropout, attn_impl) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        tokens = self.tokenizer(x)                         # (B, N, D)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        z = torch.cat([cls, tokens], dim=1) + self.pos_embed  # (B, N+1, D)
        z = self.pos_drop(z)

        attns = []
        for blk in self.blocks:
            z, attn = blk(z, return_attn=return_attn)
            attns.append(attn)

        logits = self.head(self.norm(z[:, 0]))  # chi dung token CLS de phan loai
        return (logits, attns) if return_attn else logits


def build_model(tokenizer: str, attn_impl: str, **kwargs) -> TransformerClassifier:
    return TransformerClassifier(tokenizer=tokenizer, attn_impl=attn_impl, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from .tokenizers import TOKENIZER_REGISTRY

    dummy = torch.randn(4, 1, 28, 28)
    for tok in TOKENIZER_REGISTRY:
        for impl in ("scratch", "torch"):
            m = build_model(tok, impl)
            out = m(dummy)
            print(
                f"{tok:9s} {impl:7s} tokens={m.tokenizer.num_tokens:3d} "
                f"out={tuple(out.shape)} params={count_parameters(m):,}"
            )
