"""Chạy một hoặc nhiều run A1.

Ví dụ (chạy trong thư mục A1):
    python -m src.train --run R4 --seeds 42 43 44 --data_root /content/data --out_root runs
    python -m src.train --run R2 --smoke   # kiểm tra nhanh pipeline

Mỗi run ghi vào <out_root>/<RUN>_s<seed>/:
    config.yaml, history.csv, metrics.json, confusion.npy, test_preds.npz,
    last.pt (để resume, không commit), best.pt (trọng số tốt nhất theo val top-1).
"""
import argparse
import csv
import json
import hashlib
import random
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from .config import RUNS, build_run_config, default_seeds
from .data import build_datasets, make_loader
from .engine import (autocast, full_metrics, logits_metrics, pick_amp_dtype, predict,
                     set_lr, set_seed, throughput, train_one_epoch, warmup_cosine_lr)
from .models import build_model, count_params, param_groups, set_frozen_bn_eval

HISTORY_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_top1", "val_top5", "lr", "epoch_time_s"]


class BudgetExhausted(RuntimeError):
    """Stop at an epoch boundary; the last completed epoch remains resumable."""


def check_budget(args, estimated_epoch_seconds=0):
    deadline = getattr(args, "deadline", None)
    reserve = max(120., estimated_epoch_seconds * 1.25 + 90.)
    if deadline is not None and deadline-time.monotonic() < reserve:
        raise BudgetExhausted("Time budget reached; resume in the next JupyterHub session")


def run_dir_name(cfg, lr_override):
    name = f"{cfg.run_id}_s{cfg.seed}"
    if lr_override is not None:
        name += f"_lr{lr_override:g}"
    if cfg.smoke:
        name += "_smoke"
    return name


def feature_cache_key(cfg, args):
    metadata = dict(model=cfg.timm_name, image_size=cfg.img_size, smoke=cfg.smoke,
                    data_root=os.path.abspath(args.data_root), preprocessing="resize256-centercrop-imagenet-v1")
    for filename in ("train_idx.json", "val_idx.json", "classes.json"):
        with open(os.path.join(args.splits_dir, filename), "rb") as stream:
            metadata[filename] = hashlib.sha256(stream.read()).hexdigest()
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]


def atomic_save(obj, path):
    """Ghi ra file tạm rồi os.replace: ngắt kernel giữa lúc lưu không làm hỏng checkpoint cũ."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_history(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def env_info(device, amp_dtype):
    import timm
    import torchvision
    return {
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "amp_dtype": str(amp_dtype),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
    }


def finalize(cfg, model, out_dir, test_logits, test_y, extra, device, amp_dtype, args):
    m, cm = full_metrics(test_logits, test_y, cfg.num_classes)
    np.save(os.path.join(out_dir, "confusion.npy"), cm)
    top5 = test_logits.topk(5, dim=1).indices.numpy().astype(np.int16)
    np.savez_compressed(os.path.join(out_dir, "test_preds.npz"), top5=top5, labels=test_y.numpy().astype(np.int16))
    bs, iters = (8, 2) if cfg.smoke else (256, 20)
    tput = throughput(model, device, amp_dtype, batch_size=bs, iters=iters)
    total, trainable = count_params(model)
    metrics = {
        "run_id": cfg.run_id, "seed": cfg.seed, "model": cfg.model_key, "mode": cfg.mode["name"],
        "timm_name": cfg.timm_name, "params_total": total, "params_trainable": trainable,
        "test_top1": m["top1"], "test_top5": m["top5"], "test_macro_f1": m["macro_f1"], "test_loss": m["loss"],
        "throughput_img_s": tput, "notes": cfg.notes, **extra, **env_info(device, amp_dtype),
    }
    tmp = os.path.join(out_dir, "metrics.json.tmp")
    with open(tmp, "w") as f:
        json.dump(metrics, f, indent=2)
    os.replace(tmp, os.path.join(out_dir, "metrics.json"))
    print(f"[{cfg.run_id} s{cfg.seed}] TEST top1={m['top1']:.4f} top5={m['top5']:.4f} "
          f"macroF1={m['macro_f1']:.4f} | trainable={trainable:,} | {tput:.0f} img/s")
    return metrics


# ---------------------------------------------------------------------------
# Chế độ freeze toàn bộ: trích feature 1 lần rồi train head (protocol mục 9)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract(model, loader, device, amp_dtype, max_steps=None):
    model.eval()
    feats, ys = [], []
    for i, (x, y) in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with autocast(device, amp_dtype):
            f = model.forward_head(model.forward_features(x), pre_logits=True)
        feats.append(f.float().cpu().half())
        ys.append(y)
    return torch.cat(feats), torch.cat(ys)


def get_features(cfg, model, args, device, amp_dtype):
    """Feature dùng chung cho mọi seed của cùng một backbone (trọng số pretrained giống nhau)."""
    cache_dir = os.path.join(args.out_root, "_features")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{cfg.model_key}_{feature_cache_key(cfg, args)}.pt")
    if os.path.exists(path):
        print(f"Dùng feature đã trích: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)
    # Ảnh train dùng transform eval (Resize 256 + CenterCrop 224) vì feature chỉ trích 1 lần
    tr, va, te = build_datasets(args.data_root, args.splits_dir, cfg.mode["aug"], cfg.img_size, train_aug=False)
    lim = 2 if cfg.smoke else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = {}
    for name, ds in (("train", tr), ("val", va), ("test", te)):
        loader = make_loader(ds, cfg.batch_size, False, args.num_workers)
        out[name] = extract(model, loader, device, amp_dtype, max_steps=lim)
        print(f"  feature {name}: {tuple(out[name][0].shape)}")
    out["extract_time_s"] = time.perf_counter() - t0
    out["extract_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    atomic_save(out, path)
    return out


def run_freeze_all(cfg, args, device, amp_dtype, out_dir):
    mode = cfg.mode
    check_budget(args)
    model = build_model(cfg.model_key, mode, cfg.num_classes).to(device).to(memory_format=torch.channels_last)
    feats = get_features(cfg, model, args, device, amp_dtype)
    set_seed(cfg.seed)
    head = model.get_classifier()
    nn.init.trunc_normal_(head.weight, std=0.02)
    nn.init.zeros_(head.bias)
    xtr, ytr = feats["train"]
    xva, yva = feats["val"]
    xte, yte = feats["test"]

    groups = param_groups(model, None, mode["lr_head"], mode["weight_decay"])
    optimizer = torch.optim.AdamW(groups)
    criterion = nn.CrossEntropyLoss()
    steps_per_epoch = math.ceil(len(xtr) / cfg.batch_size)
    total_steps = steps_per_epoch * mode["epochs"]
    hist_path = os.path.join(out_dir, "history.csv")
    if os.path.exists(hist_path):
        os.remove(hist_path)  # head train rất nhanh nên chạy lại từ đầu thay vì resume

    def head_logits(x):
        return head(x.to(device).float())

    best_acc, best_state, best_epoch, step = -1.0, None, 0, 0
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(cfg.seed)
    for epoch in range(1, mode["epochs"] + 1):
        te0 = time.perf_counter()
        head.train()
        perm = torch.randperm(len(xtr), generator=g)
        loss_sum, correct, n = 0.0, 0, 0
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            x, y = xtr[idx], ytr[idx].to(device)
            set_lr(optimizer, warmup_cosine_lr(step, total_steps, 0))
            logits = head_logits(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(idx)
            step += 1
        head.eval()
        with torch.no_grad():
            vm = logits_metrics(head_logits(xva).cpu(), yva)
        append_history(hist_path, {
            "epoch": epoch, "train_loss": loss_sum / n, "train_acc": correct / n,
            "val_loss": vm["loss"], "val_top1": vm["top1"], "val_top5": vm["top5"],
            "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": time.perf_counter() - te0,
        })
        if vm["top1"] > best_acc:
            best_acc, best_epoch = vm["top1"], epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        print(f"[{cfg.run_id} s{cfg.seed}] ep {epoch}/{mode['epochs']} loss={loss_sum / n:.4f} val_top1={vm['top1']:.4f}")
    head_time = time.perf_counter() - t0

    head.load_state_dict(best_state)
    atomic_save(model.state_dict(), os.path.join(out_dir, "best.pt"))
    with torch.no_grad():
        test_logits = head_logits(xte).float().cpu()
    extra = {
        "epochs_run": mode["epochs"], "best_epoch": best_epoch, "best_val_top1": best_acc,
        "train_time_s": feats["extract_time_s"] + head_time,
        "extract_time_s": feats["extract_time_s"], "head_train_time_s": head_time,
        "time_per_epoch_s": head_time / mode["epochs"],
        "peak_mem_gb": feats["extract_peak_mem_gb"],
        "feature_cache": True,
    }
    cfg.notes.append("freeze_all: train head trên feature trích 1 lần (ảnh train dùng CenterCrop, không augmentation)")
    return finalize(cfg, model, out_dir, test_logits, yte, extra, device, amp_dtype, args)


# ---------------------------------------------------------------------------
# Các chế độ train backbone: from-scratch, freeze một phần, full finetune
# ---------------------------------------------------------------------------
def run_standard(cfg, args, device, amp_dtype, out_dir):
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy

    mode = cfg.mode
    set_seed(cfg.seed)
    model = build_model(cfg.model_key, mode, cfg.num_classes).to(device).to(memory_format=torch.channels_last)
    tr, va, te = build_datasets(args.data_root, args.splits_dir, mode["aug"], cfg.img_size)
    train_loader = make_loader(tr, cfg.batch_size, True, args.num_workers, drop_last=True)
    val_loader = make_loader(va, cfg.batch_size, False, args.num_workers)

    groups = param_groups(model, mode["lr_backbone"], mode["lr_head"], mode["weight_decay"])
    optimizer = torch.optim.AdamW(groups)
    scaler = torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None
    if mode["mixup"]:
        mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, switch_prob=0.5,
                         label_smoothing=0.1, num_classes=cfg.num_classes)
        criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        criterion = nn.CrossEntropyLoss()
    bn_hook = set_frozen_bn_eval if mode["freeze"] != "none" else None

    max_steps = 3 if cfg.smoke else None
    steps_per_epoch = min(len(train_loader), max_steps or len(train_loader))
    total_steps = steps_per_epoch * mode["epochs"]
    warmup_steps = steps_per_epoch * mode["warmup_epochs"]

    last_path = os.path.join(out_dir, "last.pt")
    best_path = os.path.join(out_dir, "best.pt")
    hist_path = os.path.join(out_dir, "history.csv")
    start_epoch, best_acc, best_epoch, elapsed, peak_mem = 1, -1.0, 0, 0.0, 0.0
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch, best_acc, best_epoch = ck["epoch"] + 1, ck["best_acc"], ck["best_epoch"]
        elapsed, peak_mem = ck["elapsed"], ck["peak_mem_gb"]
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if "rng_python" in ck:
            random.setstate(ck["rng_python"])
            state = ck["rng_numpy"]
            np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        if device.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in ck["rng_cuda"]])
        print(f"Resume {cfg.run_id} s{cfg.seed} từ epoch {start_epoch}")

    estimated_epoch_seconds = elapsed / max(1, start_epoch-1) if start_epoch > 1 else 0.
    for epoch in range(start_epoch, mode["epochs"] + 1):
        check_budget(args, estimated_epoch_seconds)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        tl, ta, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, device, amp_dtype, scaler,
            step_offset=(epoch - 1) * steps_per_epoch, total_steps=total_steps,
            warmup_steps=warmup_steps, mixup_fn=mixup_fn, bn_hook=bn_hook, max_steps=max_steps)
        train_time = time.perf_counter() - t0
        vl, vy = predict(model, val_loader, device, amp_dtype, max_steps=2 if cfg.smoke else None)
        vm = logits_metrics(vl, vy)
        epoch_time = time.perf_counter() - t0
        elapsed += epoch_time
        estimated_epoch_seconds = epoch_time
        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)
        append_history(hist_path, {
            "epoch": epoch, "train_loss": tl, "train_acc": ta, "val_loss": vm["loss"],
            "val_top1": vm["top1"], "val_top5": vm["top5"],
            "lr": max(g["lr"] for g in optimizer.param_groups), "epoch_time_s": epoch_time,
        })
        if vm["top1"] > best_acc:
            best_acc, best_epoch = vm["top1"], epoch
            atomic_save(model.state_dict(), best_path)
        atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch, "best_acc": best_acc, "best_epoch": best_epoch,
            "elapsed": elapsed, "peak_mem_gb": peak_mem, "rng_cpu": torch.get_rng_state(),
            "rng_python": random.getstate(),
            "rng_numpy": (np.random.get_state()[0], np.random.get_state()[1].tolist(),
                          *np.random.get_state()[2:]),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }, last_path)
        print(f"[{cfg.run_id} s{cfg.seed}] ep {epoch}/{mode['epochs']} train_loss={tl:.4f} "
              f"val_top1={vm['top1']:.4f} ({train_time:.0f}s train, {epoch_time:.0f}s tổng)")

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_loader = make_loader(te, cfg.batch_size, False, args.num_workers)
    test_logits, test_y = predict(model, test_loader, device, amp_dtype, max_steps=2 if cfg.smoke else None)
    extra = {
        "epochs_run": mode["epochs"], "best_epoch": best_epoch, "best_val_top1": best_acc,
        "train_time_s": elapsed, "time_per_epoch_s": elapsed / max(1, mode["epochs"]),
        "peak_mem_gb": peak_mem, "feature_cache": False,
    }
    return finalize(cfg, model, out_dir, test_logits, test_y, extra, device, amp_dtype, args)


def run_one(run_id, seed, args):
    cfg = build_run_config(run_id, seed, epochs=args.epochs, lr_backbone=args.lr_backbone,
                           batch_size=args.batch_size, smoke=args.smoke)
    out_dir = os.path.join(args.out_root, run_dir_name(cfg, args.lr_backbone))
    if os.path.exists(os.path.join(out_dir, "metrics.json")) and not args.force:
        print(f"Bỏ qua {out_dir}: đã có metrics.json (dùng --force để chạy lại)")
        return
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as stream:
            existing = yaml.safe_load(stream)
        if existing != cfg.to_dict():
            raise ValueError(f"Config differs from existing run {out_dir}; choose a new --out_root")
    if args.force and os.path.exists(os.path.join(out_dir, "last.pt")):
        raise ValueError("Use a new --out_root to restart training and preserve existing checkpoints")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    amp_dtype = pick_amp_dtype(device)
    print(f"=== {cfg.run_id} seed {seed}: {cfg.model_key} / {cfg.mode['name']} | "
          f"{env_info(device, amp_dtype)['gpu']} | amp={amp_dtype} ===")
    if cfg.mode["freeze"] == "all":
        run_freeze_all(cfg, args, device, amp_dtype, out_dir)
    else:
        run_standard(cfg, args, device, amp_dtype, out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs="+", required=True, choices=list(RUNS))
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="mặc định theo protocol")
    p.add_argument("--data_root", default="/content/data")
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--out_root", default="runs")
    p.add_argument("--num_workers", type=int, default=min(12, os.cpu_count() or 2))
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="ghi đè số epoch (phải ghi trong báo cáo)")
    p.add_argument("--lr_backbone", type=float, default=None, help="dùng cho dò LR {5e-5,1e-4,3e-4}")
    p.add_argument("--smoke", action="store_true", help="chạy thử vài batch")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    for run_id in args.run:
        for seed in (args.seeds or default_seeds(run_id)):
            run_one(run_id, seed, args)


if __name__ == "__main__":
    main()
