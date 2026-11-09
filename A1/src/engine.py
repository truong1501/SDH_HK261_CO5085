"""Vòng lặp train/eval tự viết (không dùng Trainer.fit), lịch LR, metric."""
import math
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_amp_dtype(device):
    """bf16 theo protocol; GPU không hỗ trợ bf16 (vd T4) thì dùng fp16 + GradScaler."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(device, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def warmup_cosine_lr(step, total_steps, warmup_steps, min_ratio=1e-3):
    """Hệ số nhân LR: tăng tuyến tính trong warmup rồi giảm cosine về ~0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, factor):
    for g in optimizer.param_groups:
        g["lr"] = g["base_lr"] * factor


def topk_correct(logits, target, ks=(1, 5)):
    maxk = max(ks)
    pred = logits.topk(maxk, dim=1).indices
    correct = pred.eq(target.view(-1, 1))
    return [correct[:, :k].any(dim=1).sum().item() for k in ks]


def train_one_epoch(model, loader, optimizer, criterion, device, amp_dtype, scaler,
                    step_offset, total_steps, warmup_steps, mixup_fn=None,
                    bn_hook=None, max_steps=None):
    model.train()
    if bn_hook is not None:
        bn_hook(model)
    loss_sum, correct, n, step = 0.0, 0, 0, step_offset
    for i, (x, y) in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        y_hard = y
        if mixup_fn is not None:
            x, y = mixup_fn(x, y)
        set_lr(optimizer, warmup_cosine_lr(step, total_steps, warmup_steps))
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp_dtype):
            logits = model(x)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = x.size(0)
        loss_sum += loss.item() * bs
        n += bs
        if mixup_fn is None:
            correct += (logits.argmax(1) == y_hard).sum().item()
        step += 1
    train_acc = correct / n if (mixup_fn is None and n) else float("nan")
    return loss_sum / max(n, 1), train_acc, step


@torch.no_grad()
def predict(model, loader, device, amp_dtype, max_steps=None):
    """Trả về logits (float32, CPU) và nhãn."""
    model.eval()
    all_logits, all_y = [], []
    for i, (x, y) in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with autocast(device, amp_dtype):
            logits = model(x)
        all_logits.append(logits.float().cpu())
        all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)


def logits_metrics(logits, y):
    loss = F.cross_entropy(logits, y).item()
    c1, c5 = topk_correct(logits, y)
    return {"loss": loss, "top1": c1 / len(y), "top5": c5 / len(y)}


def full_metrics(logits, y, num_classes):
    from sklearn.metrics import confusion_matrix, f1_score

    m = logits_metrics(logits, y)
    pred = logits.argmax(1).numpy()
    yt = y.numpy()
    m["macro_f1"] = float(f1_score(yt, pred, labels=list(range(num_classes)), average="macro", zero_division=0))
    cm = confusion_matrix(yt, pred, labels=list(range(num_classes)))
    return m, cm


@torch.no_grad()
def throughput(model, device, amp_dtype, batch_size=256, img_size=224, warmup=5, iters=20):
    """Ảnh/giây khi suy luận (chỉ forward, input tổng hợp, batch 256)."""
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device).to(memory_format=torch.channels_last)
    for _ in range(warmup):
        with autocast(device, amp_dtype):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with autocast(device, amp_dtype):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return batch_size * iters / (time.perf_counter() - t0)
