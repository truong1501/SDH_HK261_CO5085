"""Evaluate one pipeline on a fixed held-out split with paired reference proxies."""
import argparse
import csv
import json
import time
from pathlib import Path


def main():
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from datasets import load_from_disk
    from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
    from transformers import CLIPModel, CLIPProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--unet", help="Run directory or checkpoint-N containing the fine-tuned unet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--text_guidance", type=float, default=7.5)
    parser.add_argument("--image_guidance", type=float, default=1.5)
    args = parser.parse_args()
    import math
    if not all(math.isfinite(x) and x >= 1 for x in (args.text_guidance,args.image_guidance)):
        parser.error("guidance values must be finite and >= 1")
    if args.limit < 1 or args.steps < 1 or args.resolution < 64 or args.resolution % 8:
        parser.error("Positive limit/steps and resolution divisible by 8 (>=64) required")
    if not torch.cuda.is_available():
        raise RuntimeError("Run comparison on a CUDA GPU")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new output directory to preserve comparisons")
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(args.data)[args.split]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model, torch_dtype=dtype)
    if args.unet:
        pipe.unet = UNet2DConditionModel.from_pretrained(args.unet, subfolder="unet", torch_dtype=dtype)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    clip_id = "openai/clip-vit-base-patch32"
    clip = CLIPModel.from_pretrained(clip_id).to("cuda").eval()
    processor = CLIPProcessor.from_pretrained(clip_id)
    rows = []
    with torch.inference_mode():
        for i in range(min(args.limit, len(dataset))):
            row = dataset[i]
            source = ImageOps.fit(row["input_image"].convert("RGB"), (args.resolution, args.resolution))
            target = ImageOps.fit(row["edited_image"].convert("RGB"), source.size)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = pipe(row["edit_prompt"], image=source, num_inference_steps=args.steps,
                          guidance_scale=args.text_guidance, image_guidance_scale=args.image_guidance,
                          generator=torch.Generator("cuda").manual_seed(args.seed+i))
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            prediction = result.images[0]
            flagged = bool(result.nsfw_content_detected[0]) if result.nsfw_content_detected is not None else False
            pixels = processor(images=[prediction, target], return_tensors="pt").to("cuda")
            features = clip.get_image_features(**pixels)
            similarity = torch.nn.functional.cosine_similarity(features[:1], features[1:]).item()
            mae = float(np.abs(np.asarray(prediction, dtype=np.float32)-np.asarray(target, dtype=np.float32)).mean()/255)
            panel = Image.new("RGB", (3*args.resolution, args.resolution))
            for col, image in enumerate([source, target, prediction]):
                panel.paste(image, (col*args.resolution, 0))
            panel.save(output/f"{i:04d}_input_target_prediction.png")
            prediction.save(output/f"{i:04d}_prediction.png")
            rows.append(dict(index=i, prompt=row["edit_prompt"], clip_target_similarity=similarity,
                             paired_mae=mae, seconds=seconds, safety_flagged=flagged))
    with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(config=vars(args), count=len(rows),
                   mean_clip_target_similarity=sum(r["clip_target_similarity"] for r in rows)/len(rows),
                   mean_paired_mae=sum(r["paired_mae"] for r in rows)/len(rows),
                   mean_seconds=sum(r["seconds"] for r in rows)/len(rows),
                   safety_flagged=sum(r["safety_flagged"] for r in rows),
                   note="Reference similarity proxies; not instruction-following accuracy. Pretrained training overlap is possible.")
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
