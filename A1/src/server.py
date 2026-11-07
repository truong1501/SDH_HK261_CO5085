"""Priority queue for one GPU / JupyterHub session; preserves A1.1 epoch counts."""
import argparse
import json
import time
from pathlib import Path

from .train import BudgetExhausted, run_one


def queue():
    # Cover all eight configurations before adding replicate seeds.
    primary = [(run, 42) for run in ["R2", "R6", "R4", "R8", "R3", "R7", "R1", "R5"]]
    replicates = [(run, seed) for seed in [43, 44] for run in ["R2", "R6", "R3", "R7", "R4", "R8"]]
    return primary + replicates


def pending(out_root):
    return [(run, seed) for run, seed in queue()
            if not (Path(out_root)/f"{run}_s{seed}"/"metrics.json").is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", default="runs")
    parser.add_argument("--splits_dir", default="splits")
    parser.add_argument("--hours", type=float, default=11)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.hours <= 11.5 or args.num_workers < 0:
        parser.error("hours must be in (0,11.5], num_workers >= 0")
    print("Pending priority queue:", pending(args.out_root), flush=True)
    if args.dry_run:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the official runs")
    from .data import load_food101, make_splits
    args.deadline = time.monotonic() + args.hours*3600
    # Download time counts toward the session budget.
    load_food101(args.data_root, "train", download=True)
    if not (Path(args.splits_dir)/"train_idx.json").exists():
        make_splits(args.data_root, args.splits_dir)
    args.epochs = args.lr_backbone = args.batch_size = None
    args.smoke = args.force = False
    status = "complete"
    try:
        for run, seed in pending(args.out_root):
            run_one(run, seed, args)
    except BudgetExhausted as error:
        status = "budget_exhausted"
        print(str(error), flush=True)
    finally:
        output = Path(args.out_root)
        output.mkdir(parents=True, exist_ok=True)
        remaining = pending(args.out_root)
        (output/"session_status.json").write_text(
            json.dumps(dict(status=status if status != "complete" or not remaining else "interrupted",
                            pending=remaining), indent=2), encoding="utf-8")
    if status == "budget_exhausted":
        raise SystemExit(75)


if __name__ == "__main__":
    main()
