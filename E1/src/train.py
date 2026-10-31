"""Training loop tu viet cho E1: forward -> loss -> backward() -> optimizer.step().

Dung chung cho ca 3 model (khong goi API cap cao kieu Trainer.fit()).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import get_dataloaders
from .models import build_model, count_parameters

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

DEFAULT_EPOCHS = {"softmax": 20, "mlp": 20, "cnn": 15}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train(
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    num_workers: int,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_name}] device={device}")

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, num_workers=num_workers)

    model = build_model(model_name).to(device)
    n_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run_dir = RESULTS_DIR / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "history.csv"
    ckpt_path = run_dir / "best.pt"

    best_val_acc = -1.0
    start = time.time()

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            f.flush()
            print(
                f"[{model_name}] epoch {epoch:02d}/{epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"model_state": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                    ckpt_path,
                )

    elapsed = time.time() - start
    summary = {
        "model": model_name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "num_params": n_params,
        "best_val_acc": best_val_acc,
        "train_time_sec": elapsed,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{model_name}] xong sau {elapsed/60:.1f} phut, best_val_acc={best_val_acc:.4f}, params={n_params:,}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 1 model cho E1 tren Fashion-MNIST")
    parser.add_argument("--model", choices=["softmax", "mlp", "cnn"], required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Mac dinh theo DEFAULT_EPOCHS neu bo trong")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    epochs = args.epochs or DEFAULT_EPOCHS[args.model]
    train(args.model, epochs, args.batch_size, args.lr, args.seed, args.num_workers)
