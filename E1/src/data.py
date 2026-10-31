"""Dataset/DataLoader cho E1 (và dùng lại ở E2, E3) trên Fashion-MNIST.

Split: giữ nguyên test chính thức (10k ảnh); tách 5k ảnh (500/lớp) từ train
chính thức (60k, cân bằng 6000/lớp) làm validation, stratified, seed cố định.
Index của train/validation được lưu vào E1/splits/ để mọi run (E1-E3) và mọi
thành viên trong nhóm dùng chung một split, tái lập được.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent.parent  # .../E1
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = BASE_DIR / "splits"

SEED = 42
VAL_PER_CLASS = 500  # 500 x 10 lop = 5000 anh validation
NUM_CLASSES = 10

# mean/std cua Fashion-MNIST (tinh tren train set, gia tri pho bien dung lai)
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def _stratified_train_val_indices(
    targets: torch.Tensor, val_per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Chia index thanh train/val, lay dung `val_per_class` anh moi lop cho validation."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    targets_np = targets.numpy()
    for c in range(NUM_CLASSES):
        class_idx = np.where(targets_np == c)[0]
        rng.shuffle(class_idx)
        val_idx.append(class_idx[:val_per_class])
        train_idx.append(class_idx[val_per_class:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_splits(
    data_dir: Path = DATA_DIR,
    splits_dir: Path = SPLITS_DIR,
    val_per_class: int = VAL_PER_CLASS,
    seed: int = SEED,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Tao (hoac doc lai neu da ton tai) index train/val, luu vao splits_dir."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_idx_path = splits_dir / "train_idx.npy"
    val_idx_path = splits_dir / "val_idx.npy"
    meta_path = splits_dir / "split_meta.json"

    if not force and train_idx_path.exists() and val_idx_path.exists():
        return np.load(train_idx_path), np.load(val_idx_path)

    full_train = datasets.FashionMNIST(root=data_dir, train=True, download=True)
    train_idx, val_idx = _stratified_train_val_indices(
        full_train.targets, val_per_class, seed
    )

    np.save(train_idx_path, train_idx)
    np.save(val_idx_path, val_idx)
    meta_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "val_per_class": val_per_class,
                "num_train": int(len(train_idx)),
                "num_val": int(len(val_idx)),
                "class_names": CLASS_NAMES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return train_idx, val_idx


def get_transforms(augment: bool) -> transforms.Compose:
    """Transform cho train (augment=True) hoac val/test (augment=False).

    Khong dung RandomHorizontalFlip: nhieu lop (Sandal, Sneaker, Ankle boot,
    Bag) co huong nhat dinh, lat ngang co the tao anh phi thuc te.
    """
    ops = []
    if augment:
        ops.append(transforms.RandomCrop(28, padding=2))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ]
    )
    return transforms.Compose(ops)


def get_datasets(
    data_dir: Path = DATA_DIR,
    splits_dir: Path = SPLITS_DIR,
    augment_train: bool = True,
) -> tuple[Subset, Subset, datasets.FashionMNIST]:
    train_idx, val_idx = build_splits(data_dir=data_dir, splits_dir=splits_dir)

    train_base = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=get_transforms(augment_train)
    )
    val_base = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=get_transforms(False)
    )
    test_ds = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=get_transforms(False)
    )

    train_ds = Subset(train_base, train_idx.tolist())
    val_ds = Subset(val_base, val_idx.tolist())
    return train_ds, val_ds, test_ds


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_dir: Path = DATA_DIR,
    splits_dir: Path = SPLITS_DIR,
    augment_train: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = get_datasets(
        data_dir=data_dir, splits_dir=splits_dir, augment_train=augment_train
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets()
    print(f"Train: {len(train_ds)} anh | Validation: {len(val_ds)} anh | Test: {len(test_ds)} anh")

    val_targets = np.array([val_ds.dataset.targets[i] for i in val_ds.indices])
    print("Phan bo lop trong validation (phai dong deu 500/lop):")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {c} {name:<12} {int((val_targets == c).sum())}")
