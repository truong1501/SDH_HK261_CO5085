"""So sanh 3 model (softmax/mlp/cnn) da train: accuracy, so tham so, duong cong, confusion matrix, mau loi."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from .data import CLASS_NAMES, FASHION_MNIST_MEAN, FASHION_MNIST_STD, get_dataloaders
from .models import build_model, count_parameters

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODEL_NAMES = ["softmax", "mlp", "cnn"]


def load_history(model_name: str) -> list[dict]:
    path = RESULTS_DIR / model_name / "history.csv"
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()} for row in reader
        ]


@torch.no_grad()
def evaluate_on_test(model_name: str, device: torch.device) -> dict:
    ckpt = torch.load(RESULTS_DIR / model_name / "best.pt", map_location=device)
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, _, test_loader = get_dataloaders(batch_size=256, num_workers=2)

    all_preds, all_labels, all_images = [], [], []
    for images, labels in test_loader:
        logits = model(images.to(device))
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)
        all_images.append(images)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    images = torch.cat(all_images)

    return {
        "model": model_name,
        "test_acc": float((preds == labels).mean()),
        "test_macro_f1": float(f1_score(labels, preds, average="macro")),
        "confusion_matrix": confusion_matrix(labels, preds),
        "preds": preds,
        "labels": labels,
        "images": images,
        "num_params": count_parameters(model),
        "best_val_acc": float(ckpt["val_acc"]),
    }


def plot_curves(histories: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, hist in histories.items():
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_loss"] for h in hist], "--", label=f"{name} train")
        axes[0].plot(epochs, [h["val_loss"] for h in hist], label=f"{name} val")
        axes[1].plot(epochs, [h["train_acc"] for h in hist], "--", label=f"{name} train")
        axes[1].plot(epochs, [h["val_acc"] for h in hist], label=f"{name} val")

    axes[0].set_title("Loss theo epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Accuracy theo epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_accuracy_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(model_name: str, cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {model_name}")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j], ha="center", va="center", fontsize=6,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.close(fig)


def plot_misclassified(
    model_name: str, images: torch.Tensor, preds: np.ndarray, labels: np.ndarray, n: int = 12
) -> None:
    wrong_idx = np.where(preds != labels)[0]
    chosen = wrong_idx[:n]
    cols = 6
    rows = max(1, int(np.ceil(len(chosen) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, chosen):
        img = images[idx, 0].numpy() * FASHION_MNIST_STD[0] + FASHION_MNIST_MEAN[0]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"T:{CLASS_NAMES[labels[idx]]}\nP:{CLASS_NAMES[preds[idx]]}", fontsize=6)
        ax.axis("off")
    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle(f"Mau du doan sai — {model_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"misclassified_{model_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    histories = {name: load_history(name) for name in MODEL_NAMES}
    plot_curves(histories)

    table = []
    for name in MODEL_NAMES:
        result = evaluate_on_test(name, device)
        plot_confusion_matrix(name, result["confusion_matrix"])
        plot_misclassified(name, result["images"], result["preds"], result["labels"])

        train_summary = json.loads((RESULTS_DIR / name / "summary.json").read_text(encoding="utf-8"))
        table.append(
            {
                "model": name,
                "num_params": result["num_params"],
                "best_val_acc": result["best_val_acc"],
                "test_acc": result["test_acc"],
                "test_macro_f1": result["test_macro_f1"],
                "train_time_sec": train_summary["train_time_sec"],
            }
        )

    with (RESULTS_DIR / "comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    (RESULTS_DIR / "comparison_table.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nSo sanh 3 model:")
    print(f"{'model':10s} {'params':>10s} {'val_acc':>9s} {'test_acc':>9s} {'macro_f1':>9s} {'time(s)':>9s}")
    for row in table:
        print(
            f"{row['model']:10s} {row['num_params']:>10,d} {row['best_val_acc']:>9.4f} "
            f"{row['test_acc']:>9.4f} {row['test_macro_f1']:>9.4f} {row['train_time_sec']:>9.1f}"
        )


if __name__ == "__main__":
    main()
