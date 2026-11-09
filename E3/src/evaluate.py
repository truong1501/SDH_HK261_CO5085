"""So sanh 4 run LSTM/GRU (row/patch) voi baseline MLP/CNN tu E1.

Can chay `python -m src.baseline` truoc de co baseline_table.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import get_dataloaders
from .models import build_model
from .sequences import SEQUENCE_CONFIG, to_sequence

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

RECURRENT_RUNS = ["lstm_row", "gru_row", "lstm_patch", "gru_patch"]


def load_history(run_name: str) -> list[dict]:
    path = RESULTS_DIR / run_name / "history.csv"
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()} for row in reader
        ]


@torch.no_grad()
def evaluate_on_test(run_name: str, model_name: str, seq_kind: str, device: torch.device) -> float:
    ckpt = torch.load(RESULTS_DIR / run_name / "best.pt", map_location=device)
    input_size = SEQUENCE_CONFIG[seq_kind]["input_size"]
    model = build_model(model_name, input_size=input_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, _, test_loader = get_dataloaders(batch_size=256, num_workers=2)

    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        seq = to_sequence(images, seq_kind)
        preds = model(seq).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return correct / total


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
    axes[0].legend(fontsize=7)

    axes[1].set_title("Accuracy theo epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_accuracy_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    histories = {name: load_history(name) for name in RECURRENT_RUNS}
    plot_curves(histories)

    table = []
    for run_name in RECURRENT_RUNS:
        summary = json.loads((RESULTS_DIR / run_name / "summary.json").read_text(encoding="utf-8"))
        test_acc = evaluate_on_test(run_name, summary["model"], summary["seq"], device)
        table.append(
            {
                "model": summary["model"],
                "seq": summary["seq"],
                "num_params": summary["num_params"],
                "best_val_acc": summary["best_val_acc"],
                "test_acc": test_acc,
                "train_time_sec": summary["train_time_sec"],
            }
        )

    baseline_path = RESULTS_DIR / "baseline_table.json"
    if baseline_path.exists():
        table.extend(json.loads(baseline_path.read_text(encoding="utf-8")))
    else:
        print("Chua co baseline_table.json, chay 'python -m src.baseline' truoc.")

    with (RESULTS_DIR / "comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    (RESULTS_DIR / "comparison_table.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nSo sanh E3 (LSTM/GRU) voi baseline E1 (MLP/CNN):")
    print(f"{'model':6s} {'seq':6s} {'params':>10s} {'val_acc':>9s} {'test_acc':>9s} {'time(s)':>9s}")
    for row in table:
        print(
            f"{row['model']:6s} {row['seq']:6s} {row['num_params']:>10,d} "
            f"{row['best_val_acc']:>9.4f} {row['test_acc']:>9.4f} {row['train_time_sec']:>9.1f}"
        )


if __name__ == "__main__":
    main()
