import argparse
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.diffusion import GaussianDiffusion
from src.unet import UNetDDPM
from src.dit import DiT


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint.get("config")
    if not config:
        raise ValueError("Legacy checkpoint has no diffusion config; retrain with the current train.py")
    model = (UNetDDPM() if config["model"] == "unet" else DiT(input_size=config["image_size"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    diffusion = GaussianDiffusion(config["timesteps"], schedule=config["schedule"], device=device)
    return model, diffusion, config


def main():
    parser = argparse.ArgumentParser(description="Sample using the saved training configuration")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", default="generated_grid.png")
    parser.add_argument("--output_dir", help="Save individual PNGs for FID instead of a grid")
    parser.add_argument("--save_steps", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if min(args.num_images, args.batch_size) < 1:
        parser.error("num_images and batch_size must be positive")
    if args.save_steps and args.output_dir:
        parser.error("save_steps and output_dir are separate modes")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion, config = load_checkpoint(args.ckpt, device)
    size = config["image_size"]
    if args.output_dir:
        directory = Path(args.output_dir)
        if directory.exists() and any(directory.iterdir()):
            parser.error("output_dir must be empty to avoid mixing evaluation runs")
        directory.mkdir(parents=True, exist_ok=True)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    grids = []
    for offset in range(0, args.num_images, args.batch_size):
        batch = min(args.batch_size, args.num_images-offset)
        result = diffusion.sample(model, (batch, 3, size, size), save_intermediate=args.save_steps)
        if args.save_steps:
            _, steps = result
            result = torch.cat([image for _, image in steps])
        result = ((result+1)/2).cpu()
        if args.output_dir:
            for index, image in enumerate(result, offset):
                save_image(image, directory / f"{index:06d}.png")
        else:
            grids.append(result)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    if grids:
        save_image(torch.cat(grids), args.output, nrow=args.batch_size if args.save_steps else max(1, int(args.num_images**.5)))
    print(f"{args.num_images} images in {elapsed:.2f}s ({args.num_images/elapsed:.3f} images/s, includes PNG I/O)")


if __name__ == "__main__":
    main()
