import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from src.dataset import get_dataloader
from src.diffusion import GaussianDiffusion
from src.unet import UNetDDPM
from src.dit import DiT


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Train pixel-space U-Net or small DiT on CelebA")
    parser.add_argument("--model", choices=["unet", "dit"], default="dit")
    parser.add_argument("--data_dir", default="./data/celeba")
    parser.add_argument("--partition_file", help="Official list_eval_partition.txt (recommended)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--sample_dir", default="./samples")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_every", type=int, default=5)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.timesteps, args.sample_every) < 1:
        parser.error("epochs, batch_size, timesteps and sample_every must be positive")
    if args.image_size < 8 or args.image_size % 8:
        parser.error("image_size must be a positive multiple of 8")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.fp16 and device.type == "cuda"
    print(f"Device: {device}; model: {args.model}; seed: {args.seed}")
    for directory in [args.save_dir, args.sample_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    best_path = Path(args.save_dir) / f"{args.model}_best.pt"
    history_path = Path(args.save_dir) / f"{args.model}_history.csv"
    if best_path.exists() or history_path.exists():
        parser.error("Run already exists; choose a new --save_dir to preserve results")
    loader_args = dict(data_dir=args.data_dir, batch_size=args.batch_size,
                       image_size=args.image_size, max_samples=args.max_samples,
                       seed=args.seed, num_workers=args.num_workers, partition_file=args.partition_file)
    train_loader = get_dataloader(**loader_args, split="train")
    val_loader = get_dataloader(**loader_args, split="val")
    diffusion = GaussianDiffusion(args.timesteps, schedule=args.schedule, device=device)
    model = (UNetDDPM() if args.model == "unet" else DiT(input_size=args.image_size)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    num_params = count_parameters(model)
    best = float("inf")
    print(f"Parameters: {num_params:,}; train: {len(train_loader.dataset)}; val: {len(val_loader.dataset)}")
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss", "seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            model.train()
            total, count = 0., 0
            for x in train_loader:
                x = x.to(device)
                t = torch.randint(args.timesteps, (len(x),), device=device)
                noise = torch.randn_like(x)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = F.mse_loss(model(diffusion.q_sample(x, t, noise), t), noise)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * len(x)
                count += len(x)
            train_loss = total / count
            model.eval()
            val_total, val_count = 0., 0
            # Same validation noise/timesteps each epoch; do not perturb training RNG.
            generator = torch.Generator(device=device).manual_seed(args.seed)
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device)
                    t = torch.randint(args.timesteps, (len(x),), device=device, generator=generator)
                    noise = torch.randn(x.shape, device=device, generator=generator)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        loss = F.mse_loss(model(diffusion.q_sample(x, t, noise), t), noise)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite validation loss")
                    val_total += loss.item() * len(x)
                    val_count += len(x)
            val_loss = val_total / val_count
            row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                       seconds=time.perf_counter()-start)
            writer.writerow(row)
            stream.flush()
            print(row, flush=True)
            checkpoint = dict(epoch=epoch, model_state_dict=model.state_dict(),
                              optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
                              config=vars(args), num_params=num_params, model_type=args.model,
                              train_loss=train_loss, val_loss=val_loss)
            torch.save(checkpoint, Path(args.save_dir) / f"{args.model}_last.pt")
            if val_loss < best:
                best = val_loss
                torch.save(checkpoint, best_path)
            if epoch % args.sample_every == 0 or epoch == args.epochs:
                with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                    torch.manual_seed(args.seed)
                    samples = diffusion.sample(model, (4, 3, args.image_size, args.image_size))
                save_image((samples+1)/2, Path(args.sample_dir) / f"{args.model}_epoch_{epoch:03d}.png", nrow=2)
    (Path(args.save_dir) / f"{args.model}_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
