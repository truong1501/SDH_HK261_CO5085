"""Download the small public dataset and split by decoded source-image hash."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def image_key(image):
    image = image.convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def split_groups(keys, seed=42):
    groups = sorted(set(keys))
    if len(groups) < 10:
        raise ValueError("Need at least 10 distinct source images")
    random.Random(seed).shuffle(groups)
    n = len(groups)
    sets = {"train": set(groups[:int(.8*n)]), "validation": set(groups[int(.8*n):int(.9*n)]),
            "test": set(groups[int(.9*n):])}
    return {name: [i for i, key in enumerate(keys) if key in group] for name, group in sets.items()}


def main(argv=None):
    from datasets import load_dataset, DatasetDict
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; reuse it or choose another path")
    dataset = load_dataset("fusing/instructpix2pix-1000-samples", split="train")
    required = {"input_image", "edited_image", "edit_prompt"}
    if not required.issubset(dataset.column_names):
        raise ValueError(f"Unexpected columns: {dataset.column_names}")
    keys = []
    for row in dataset:
        if not isinstance(row["edit_prompt"], str) or not row["edit_prompt"].strip():
            raise ValueError("Empty editing instruction")
        if row["input_image"].size != row["edited_image"].size:
            raise ValueError("Unaligned input/target dimensions")
        keys.append(image_key(row["input_image"]))
    indices = split_groups(keys, args.seed)
    DatasetDict({name: dataset.select(ids) for name, ids in indices.items()}).save_to_disk(str(output))
    manifest = dict(dataset="fusing/instructpix2pix-1000-samples", fingerprint=dataset._fingerprint,
                    seed=args.seed, indices=indices, source_image_hashes=keys,
                    counts={name: len(ids) for name, ids in indices.items()})
    (output/"split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest["counts"])


if __name__ == "__main__":
    main()
