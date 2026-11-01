"""Kiểm tra MSA tự viết ĐÚNG bằng cách so với nn.MultiheadAttention.

Ý tưởng: copy trọng số của bản PyTorch sang bản tự viết, cho cùng một input, rồi so:
  1. output (B, N, D)
  2. attention weights từng head (B, h, N, N)
  3. gradient theo input (đảm bảo backward cũng giống)
Nếu sai lệch chỉ ở mức sai số làm tròn float32 (~1e-6) thì hai bản tương đương về toán học.

Chạy:  python -m src.check_msa
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .attention import ScratchMSA, TorchMSA

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@torch.no_grad()
def copy_weights(src: TorchMSA, dst: ScratchMSA) -> None:
    # in_proj_weight cua PyTorch = [W_q; W_k; W_v] xep doc, shape (3D, D) -> trung voi dst.qkv
    dst.qkv.weight.copy_(src.mha.in_proj_weight)
    dst.qkv.bias.copy_(src.mha.in_proj_bias)
    dst.out_proj.weight.copy_(src.mha.out_proj.weight)
    dst.out_proj.bias.copy_(src.mha.out_proj.bias)


def check(dim: int = 64, num_heads: int = 4, batch: int = 8, n_tokens: int = 50, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    ref = TorchMSA(dim, num_heads).eval()
    mine = ScratchMSA(dim, num_heads).eval()
    copy_weights(ref, mine)

    x1 = torch.randn(batch, n_tokens, dim, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    out_ref, attn_ref = ref(x1, return_attn=True)
    out_mine, attn_mine = mine(x2, return_attn=True)

    out_ref.sum().backward()
    out_mine.sum().backward()

    res = {
        "dim": dim,
        "num_heads": num_heads,
        "input_shape": [batch, n_tokens, dim],
        "max_abs_diff_output": (out_ref - out_mine).abs().max().item(),
        "max_abs_diff_attention": (attn_ref - attn_mine).abs().max().item(),
        "max_abs_diff_grad_input": (x1.grad - x2.grad).abs().max().item(),
        "attention_rows_sum_to_1": bool(torch.allclose(attn_mine.sum(-1), torch.ones(1))),
    }
    res["passed"] = bool(
        res["max_abs_diff_output"] < 1e-5
        and res["max_abs_diff_attention"] < 1e-5
        and res["max_abs_diff_grad_input"] < 1e-4
    )
    return res


if __name__ == "__main__":
    results = [check(), check(dim=32, num_heads=8, n_tokens=29, seed=1)]
    for r in results:
        print(json.dumps(r, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "msa_check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    assert all(r["passed"] for r in results), "MSA tu viet KHONG khop voi nn.MultiheadAttention!"
    print("OK: MSA tu viet khop voi nn.MultiheadAttention (output, attention, gradient).")
