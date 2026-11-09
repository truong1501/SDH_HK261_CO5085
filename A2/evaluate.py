"""Export held-out real images with identical preprocessing, then compute FID."""
import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image
from src.dataset import SimpleImageDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--partition_file")
    parser.add_argument("--generated_dir", required=True)
    parser.add_argument("--real_dir", required=True, help="Empty directory for exported test PNGs")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_images", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="fid.json")
    args = parser.parse_args()
    if args.num_images < 2 or args.batch_size < 1:
        parser.error("FID needs at least two images and a positive batch_size")
    from pytorch_fid.fid_score import calculate_fid_given_paths
    generated = list(Path(args.generated_dir).iterdir())
    if len(generated) != args.num_images or any(p.suffix.lower() != ".png" for p in generated):
        parser.error("generated_dir must contain exactly num_images individual PNGs")
    dataset = SimpleImageDataset(args.data_dir, args.image_size, args.num_images,
                                 split="test", seed=args.seed, partition_file=args.partition_file)
    if len(dataset) != args.num_images:
        parser.error("Not enough held-out test images for num_images")
    real_dir = Path(args.real_dir)
    if real_dir.exists() and any(real_dir.iterdir()):
        parser.error("real_dir must be empty; use a new directory for each run")
    real_dir.mkdir(parents=True, exist_ok=True)
    for index in range(len(dataset)):
        save_image((dataset[index]+1)/2, real_dir / f"{index:06d}.png")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = calculate_fid_given_paths([str(real_dir), args.generated_dir], args.batch_size, device, 2048, 0)
    result = dict(fid=float(fid), dims=2048, split="test", **vars(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
