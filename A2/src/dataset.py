import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SimpleImageDataset(Dataset):
    """Official CelebA partitions, or a deterministic 80/10/10 fallback split."""
    def __init__(self, root_dir, image_size=64, max_samples=None, split="train",
                 seed=42, partition_file=None):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive")
        paths = sorted(p for p in Path(root_dir).rglob("*")
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not paths:
            raise ValueError(f"No images found in {root_dir}")
        if partition_file:
            partitions = dict(line.split() for line in Path(partition_file).read_text().splitlines() if line.strip())
            missing = [p.name for p in paths if p.name not in partitions]
            if missing:
                raise ValueError(f"Images absent from partition file: {missing[:3]}")
            code = {"train": "0", "val": "1", "test": "2"}[split]
            paths = [p for p in paths if partitions[p.name] == code]
        else:
            random.Random(seed).shuffle(paths)
            n = len(paths)
            start, end = {"train": (0, int(.8*n)), "val": (int(.8*n), int(.9*n)),
                          "test": (int(.9*n), n)}[split]
            paths = paths[start:end]
        self.image_paths = paths[:max_samples] if max_samples else paths
        if not self.image_paths:
            raise ValueError(f"Empty {split} split; provide more images or check partition_file")
        operations = [transforms.Resize((image_size, image_size))]
        if split == "train":
            operations.append(transforms.RandomHorizontalFlip())
        operations += [transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)]
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            return self.transform(image.convert("RGB"))


def get_dataloader(data_dir, batch_size=128, image_size=64, num_workers=4,
                   max_samples=None, split="train", seed=42, partition_file=None):
    dataset = SimpleImageDataset(data_dir, image_size, max_samples, split, seed, partition_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=False, generator=torch.Generator().manual_seed(seed))
