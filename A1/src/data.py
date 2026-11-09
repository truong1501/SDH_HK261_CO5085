"""Food-101: split stratified, transforms theo mức augmentation, DataLoader."""
import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(aug: str, train: bool, img_size: int = 224):
    """aug: light (freeze toàn bộ) | medium (partial/full) | strong (from-scratch).

    Mixup/CutMix và label smoothing áp dụng ở mức batch trong vòng lặp train, không ở đây.
    """
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            norm,
        ])
    t = [transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip()]
    if aug in ("medium", "strong"):
        t.append(transforms.RandAugment(num_ops=2, magnitude=9))
    t += [transforms.ToTensor(), norm]
    if aug in ("medium", "strong"):
        t.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(t)


def load_food101(root, split, download=False):
    return datasets.Food101(root=root, split=split, download=download)


class Food101Subset(Dataset):
    """Bọc torchvision Food101, chỉ lấy các chỉ số trong `indices`, transform riêng."""

    def __init__(self, base, indices, transform):
        self.files = [base._image_files[i] for i in indices]
        self.labels = [base._labels[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as image:
            img = image.convert("RGB")
        return self.transform(img), self.labels[i]


def make_splits(data_root, splits_dir, val_frac=0.1, seed=42):
    """Tách 10% train chính thức làm validation, stratified theo lớp, seed 42."""
    from sklearn.model_selection import train_test_split

    os.makedirs(splits_dir, exist_ok=True)
    base = load_food101(data_root, "train")
    idx = np.arange(len(base._labels))
    tr, va = train_test_split(idx, test_size=val_frac, stratify=base._labels, random_state=seed)
    tr, va = sorted(tr.tolist()), sorted(va.tolist())
    with open(os.path.join(splits_dir, "train_idx.json"), "w") as f:
        json.dump(tr, f)
    with open(os.path.join(splits_dir, "val_idx.json"), "w") as f:
        json.dump(va, f)
    with open(os.path.join(splits_dir, "classes.json"), "w") as f:
        json.dump(base.classes, f)
    print(f"Split: train={len(tr)}, val={len(va)} (seed={seed}) -> {splits_dir}")
    return tr, va


def load_splits(splits_dir):
    with open(os.path.join(splits_dir, "train_idx.json")) as f:
        tr = json.load(f)
    with open(os.path.join(splits_dir, "val_idx.json")) as f:
        va = json.load(f)
    return tr, va


def build_datasets(data_root, splits_dir, aug, img_size=224, train_aug=True):
    tr_idx, va_idx = load_splits(splits_dir)
    base_train = load_food101(data_root, "train")
    base_test = load_food101(data_root, "test")
    validate_splits(tr_idx, va_idx, len(base_train._labels))
    eval_tf = build_transform(aug, train=False, img_size=img_size)
    train_tf = build_transform(aug, train=True, img_size=img_size) if train_aug else eval_tf
    return (
        Food101Subset(base_train, tr_idx, train_tf),
        Food101Subset(base_train, va_idx, eval_tf),
        Food101Subset(base_test, list(range(len(base_test._labels))), eval_tf),
    )


def make_loader(ds, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def validate_splits(train, val, size):
    if not train or not val:
        raise ValueError("Train and validation splits must be nonempty")
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("Duplicate indices in split")
    if set(train) & set(val):
        raise ValueError("Train and validation overlap")
    if set(train) | set(val) != set(range(size)):
        raise ValueError("Splits must cover the official training set exactly")
