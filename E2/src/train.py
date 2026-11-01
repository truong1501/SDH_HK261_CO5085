"""Training loop tự viết cho E2: forward -> loss -> backward() -> optimizer.step().

Dùng chung cho mọi thí nghiệm trong experiments.py (không gọi API cấp cao kiểu Trainer.fit()).

Ví dụ:
    python -m src.train --exp patch4_scratch
    python -m src.train --exp patch4_scratch --subset 2000 --epochs 1   # chạy thử nhanh
    python -m src.train --exp patch4_scratch --resume                   # train tiếp từ last.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import get_dataloaders, get_device
from .experiments import EXPERIMENTS, MODEL_CFG, TRAIN_CFG
from .models import build_model, count_parameters

BASE_DIR = Path(__file__).resolve().parent.parent  # .../E2
RESULTS_DIR = BASE_DIR / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    """GPU chay bat dong bo: phai dong bo truoc khi bam gio thi thoi gian moi dung."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def warmup_cosine(total_steps: int, warmup_steps: int):
    """He so lr: tang tuyen tinh trong warmup, sau do giam theo cosine ve 0.
    Transformer tren du lieu nho thuong can warmup de khong bi mat on dinh luc dau."""

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def run_epoch(model, loader, criterion, optimizer, scheduler, device, train: bool):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)               # forward
            loss = criterion(logits, labels)     # loss

            if train:
                optimizer.zero_grad()
                loss.backward()                  # backward
                optimizer.step()                 # cap nhat trong so
                scheduler.step()                 # lr theo tung step

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train(args: argparse.Namespace) -> dict:
    exp = EXPERIMENTS[args.exp]
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[{args.exp}] device={device} tokenizer={exp['tokenizer']} attn={exp['attn_impl']}")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        subset=args.subset,
    )

    model = build_model(exp["tokenizer"], exp["attn_impl"], **MODEL_CFG).to(device)
    n_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch)
    )

    run_dir = Path(args.out_dir) / args.exp
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) / args.exp if args.ckpt_dir else run_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "history.csv"
    best_path = run_dir / "best.pt"     # nho (~0.5 MB), commit duoc
    last_path = ckpt_dir / "last.pt"    # co ca optimizer, dung de resume, khong commit

    start_epoch, best_val_acc, prev_time = 1, -1.0, 0.0
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optim_state"])
        scheduler.load_state_dict(ck["sched_state"])
        start_epoch, best_val_acc, prev_time = ck["epoch"] + 1, ck["best_val_acc"], ck["train_time_sec"]
        print(f"[{args.exp}] resume tu epoch {ck['epoch']}")

    mode = "a" if start_epoch > 1 else "w"
    train_time = prev_time
    with log_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_sec"])

        for epoch in range(start_epoch, args.epochs + 1):
            sync(device)
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, True)
            sync(device)
            epoch_time = time.time() - t0  # chi tinh thoi gian train (khong tinh validation)
            train_time += epoch_time
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, None, device, False)

            lr_now = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr_now, epoch_time])
            f.flush()
            print(
                f"[{args.exp}] epoch {epoch:02d}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({epoch_time:.1f}s)"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"model_state": model.state_dict(), "epoch": epoch, "val_acc": val_acc, "exp": args.exp},
                    best_path,
                )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "sched_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "train_time_sec": train_time,
                },
                last_path,
            )

    summary = {
        "exp": args.exp,
        **exp,
        **MODEL_CFG,
        "num_tokens": model.tokenizer.num_tokens,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "subset": args.subset,
        "device": str(device),
        "num_params": n_params,
        "best_val_acc": best_val_acc,
        "train_time_sec": train_time,
        "avg_epoch_sec": train_time / max(1, args.epochs),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[{args.exp}] xong sau {train_time/60:.1f} phut, best_val_acc={best_val_acc:.4f}, "
        f"params={n_params:,}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 1 thi nghiem E2 tren Fashion-MNIST")
    p.add_argument("--exp", choices=list(EXPERIMENTS), required=True)
    p.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
    p.add_argument("--batch-size", type=int, default=TRAIN_CFG["batch_size"])
    p.add_argument("--lr", type=float, default=TRAIN_CFG["lr"])
    p.add_argument("--weight-decay", type=float, default=TRAIN_CFG["weight_decay"])
    p.add_argument("--warmup-epochs", type=int, default=TRAIN_CFG["warmup_epochs"])
    p.add_argument("--seed", type=int, default=TRAIN_CFG["seed"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--data-dir", default=None, help="Thu muc du lieu (mac dinh E1/data)")
    p.add_argument("--out-dir", default=str(RESULTS_DIR), help="Noi luu log/summary/best.pt")
    p.add_argument("--ckpt-dir", default=None, help="Noi luu last.pt (vd Google Drive); mac dinh = out-dir")
    p.add_argument("--subset", type=int, default=None, help="Chi dung N anh train de chay thu")
    p.add_argument("--resume", action="store_true", help="Train tiep tu last.pt neu co")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
