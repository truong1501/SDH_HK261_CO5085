"""Đánh giá + so sánh các thí nghiệm E2 đã train.

Sinh ra trong results/:
  comparison_table.{csv,json}  : mọi thí nghiệm (test acc, macro-F1, params, số token, thời gian)
  table_attention.csv          : (i)  MSA tự viết vs MSA PyTorch (cùng patch 4x4)
  table_tokenizers.csv         : (ii) các cách tokenize (cùng MSA tự viết)
  curves_attention.png, curves_tokenizers.png : đường cong loss/accuracy
  confusion_matrix_<exp>.png   : cho thí nghiệm tốt nhất
  attention_maps.png           : token CLS "nhìn" vào đâu (khối cuối, trung bình các head)

Chạy:  python -m src.evaluate
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from .data import CLASS_NAMES, MEAN, STD, get_dataloaders, get_device
from .experiments import COMPARE_ATTENTION, COMPARE_TOKENIZERS, EXPERIMENTS, LABELS, MODEL_CFG
from .models import build_model, count_parameters
from .train import RESULTS_DIR, sync


def load_history(run_dir: Path) -> list[dict]:
    with (run_dir / "history.csv").open(encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load_model(exp_name: str, results_dir: Path, device: torch.device):
    exp = EXPERIMENTS[exp_name]
    model = build_model(exp["tokenizer"], exp["attn_impl"], **MODEL_CFG).to(device)
    ckpt = torch.load(results_dir / exp_name / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval(), ckpt


@torch.no_grad()
def evaluate_on_test(exp_name: str, results_dir: Path, test_loader, device) -> dict:
    model, ckpt = load_model(exp_name, results_dir, device)

    # Do toc do suy luan: chay vai batch "khoi dong" truoc, roi bam gio ca tap test
    for i, (images, _) in enumerate(test_loader):
        model(images.to(device))
        if i == 2:
            break
    preds, labels = [], []
    sync(device)
    t0 = time.time()
    for images, y in test_loader:
        preds.append(model(images.to(device)).argmax(1).cpu())
        labels.append(y)
    sync(device)
    infer_time = time.time() - t0

    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    summary = json.loads((results_dir / exp_name / "summary.json").read_text(encoding="utf-8"))
    return {
        "exp": exp_name,
        "tokenizer": EXPERIMENTS[exp_name]["tokenizer"],
        "attn_impl": EXPERIMENTS[exp_name]["attn_impl"],
        "num_tokens": model.tokenizer.num_tokens,
        "num_params": count_parameters(model),
        "best_epoch": int(ckpt["epoch"]),
        "best_val_acc": float(ckpt["val_acc"]),
        "test_acc": float((preds == labels).mean()),
        "test_macro_f1": float(f1_score(labels, preds, average="macro")),
        "train_time_sec": summary["train_time_sec"],
        "avg_epoch_sec": summary["avg_epoch_sec"],
        "test_infer_imgs_per_sec": len(labels) / infer_time,
        "_preds": preds,
        "_labels": labels,
    }


def write_table(rows: list[dict], path: Path) -> None:
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_curves(names: list[str], results_dir: Path, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in names:
        hist = load_history(results_dir / name)
        ep = [h["epoch"] for h in hist]
        (line,) = axes[0].plot(ep, [h["val_loss"] for h in hist], label=f"{name} val")
        axes[0].plot(ep, [h["train_loss"] for h in hist], "--", color=line.get_color(), alpha=0.6)
        axes[1].plot(ep, [h["val_acc"] for h in hist], color=line.get_color(), label=f"{name} val")
        axes[1].plot(ep, [h["train_acc"] for h in hist], "--", color=line.get_color(), alpha=0.6)
    axes[0].set(title="Loss (net dut = train, lien = val)", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Accuracy (net dut = train, lien = val)", xlabel="epoch", ylabel="accuracy")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(res: dict, path: Path) -> None:
    cm = confusion_matrix(res["_labels"], res["_preds"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(10), CLASS_NAMES, fontsize=8)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Du doan", ylabel="Nhan that", title=f"Confusion matrix - {res['exp']}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_attention_maps(exp_names: list[str], results_dir: Path, test_loader, device, path: Path, n: int = 6):
    """Attention cua token CLS toi cac token anh o khoi cuoi, trung binh cac head,
    reshape ve luoi token roi phong to len 28x28 de chong len anh."""
    images, labels = next(iter(test_loader))
    images, labels = images[:n], labels[:n]
    exp_names = [e for e in exp_names if (results_dir / e / "best.pt").exists()]
    fig, axes = plt.subplots(len(exp_names) + 1, n, figsize=(2 * n, 2.1 * (len(exp_names) + 1)))
    raw = images * STD[0] + MEAN[0]
    for j in range(n):
        axes[0, j].imshow(raw[j, 0], cmap="gray")
        axes[0, j].set_title(CLASS_NAMES[labels[j]], fontsize=8)
    axes[0, 0].set_ylabel("anh goc", fontsize=8)

    for r, name in enumerate(exp_names, start=1):
        model, _ = load_model(name, results_dir, device)
        gh, gw = model.tokenizer.grid
        _, attns = model(images.to(device), return_attn=True)
        cls_attn = attns[-1][:, :, 0, 1:].mean(1)            # (n, N): CLS -> cac token anh
        maps = cls_attn.reshape(n, 1, gh, gw)
        maps = F.interpolate(maps, size=(28, 28), mode="bilinear", align_corners=False).cpu()
        for j in range(n):
            axes[r, j].imshow(raw[j, 0], cmap="gray")
            axes[r, j].imshow(maps[j, 0], cmap="jet", alpha=0.45)
        axes[r, 0].set_ylabel(name, fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Attention cua token CLS (khoi cuoi, trung binh cac head)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    results_dir = Path(args.out_dir)
    device = get_device(args.device)
    _, _, test_loader = get_dataloaders(batch_size=256, num_workers=args.num_workers, data_dir=args.data_dir)

    done = [e for e in EXPERIMENTS if (results_dir / e / "best.pt").exists()]
    missing = [e for e in EXPERIMENTS if e not in done]
    if missing:
        print("Chua train (bo qua):", missing)
    if not done:
        raise SystemExit("Chua co thi nghiem nao trong results/. Hay chay src.train truoc.")

    results = {e: evaluate_on_test(e, results_dir, test_loader, device) for e in done}
    for r in results.values():
        print(
            f"{r['exp']:18s} test_acc={r['test_acc']:.4f} f1={r['test_macro_f1']:.4f} "
            f"params={r['num_params']:,} tokens={r['num_tokens']} "
            f"epoch={r['avg_epoch_sec']:.1f}s infer={r['test_infer_imgs_per_sec']:.0f} img/s"
        )

    rows = list(results.values())
    write_table(rows, results_dir / "comparison_table.csv")
    (results_dir / "comparison_table.json").write_text(
        json.dumps([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], indent=2),
        encoding="utf-8",
    )

    att = [e for e in COMPARE_ATTENTION if e in results]
    tok = [e for e in COMPARE_TOKENIZERS if e in results]
    write_table([results[e] for e in att], results_dir / "table_attention.csv")
    write_table([{**results[e], "label": LABELS[results[e]["tokenizer"]]} for e in tok],
                results_dir / "table_tokenizers.csv")
    if att:
        plot_curves(att, results_dir, results_dir / "curves_attention.png", "(i) MSA tu viet vs MSA PyTorch (patch 4x4)")
    if tok:
        plot_curves(tok, results_dir, results_dir / "curves_tokenizers.png", "(ii) So sanh cach tokenize (MSA tu viet)")

    best = max(rows, key=lambda r: r["test_acc"])
    plot_confusion(best, results_dir / f"confusion_matrix_{best['exp']}.png")

    vis = [e for e in done if EXPERIMENTS[e]["tokenizer"] in ("patch4", "cnnstem")]
    if vis:
        plot_attention_maps(vis, results_dir, test_loader, device, results_dir / "attention_maps.png")
    print("Da luu bang + bieu do vao", results_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Danh gia + so sanh cac thi nghiem E2")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument("--data-dir", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
