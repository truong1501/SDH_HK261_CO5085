"""Danh gia lai MLP va CNN cua E1 tren test set, lam baseline khong hoi quy cho E3.

Khong train lai: load truc tiep checkpoint da co san tu E1/results/{mlp,cnn}/best.pt,
dung chung mot split/test set voi E3 (xem E3/src/data.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

E1_SRC = Path(__file__).resolve().parent.parent.parent / "E1" / "src"
sys.path.insert(0, str(E1_SRC))

from data import get_dataloaders  # noqa: E402
from models import build_model, count_parameters  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
E1_RESULTS_DIR = E1_SRC.parent / "results"

BASELINE_MODELS = ["mlp", "cnn"]


@torch.no_grad()
def evaluate_baseline(model_name: str, device: torch.device) -> dict:
    ckpt_path = E1_RESULTS_DIR / model_name / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, _, test_loader = get_dataloaders(batch_size=256, num_workers=2)

    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    e1_summary = json.loads((E1_RESULTS_DIR / model_name / "summary.json").read_text(encoding="utf-8"))
    return {
        "model": model_name,
        "seq": "-",
        "num_params": count_parameters(model),
        "best_val_acc": float(ckpt["val_acc"]),
        "test_acc": correct / total,
        "train_time_sec": e1_summary["train_time_sec"],
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = [evaluate_baseline(name, device) for name in BASELINE_MODELS]
    (RESULTS_DIR / "baseline_table.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Baseline tu E1 (khong train lai):")
    for row in rows:
        print(f"{row['model']:6s} params={row['num_params']:>8,d} test_acc={row['test_acc']:.4f}")


if __name__ == "__main__":
    main()
