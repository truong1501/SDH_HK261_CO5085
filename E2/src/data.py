"""DataLoader cho E2: dùng lại đúng dataset + split của E1 (Fashion-MNIST, seed 42,
train 55k / validation 5k / test 10k), theo yêu cầu "cùng tập ảnh cho E1–E3".

Index split nằm ở E1/splits/ (sinh tự động lần đầu), nên E1, E2, E3 luôn cùng split.
Tham số `data_dir` cho phép đặt dữ liệu ở chỗ khác (vd. Google Drive / ổ /content trên Colab).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../SDH_HK261_CO5085
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from E1.src import data as e1_data  # noqa: E402  (dung lai split + transform cua E1)

CLASS_NAMES = e1_data.CLASS_NAMES
MEAN, STD = e1_data.FASHION_MNIST_MEAN, e1_data.FASHION_MNIST_STD
DEFAULT_DATA_DIR = e1_data.DATA_DIR  # E1/data


def get_device(name: str = "auto") -> torch.device:
    """auto: uu tien GPU NVIDIA (Colab), roi Apple GPU (MPS tren Mac), cuoi cung CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_dir: str | Path | None = None,
    subset: int | None = None,
    augment_train: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """subset: chi lay `subset` anh dau cua train (va validation) de chay thu nhanh."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    train_ds, val_ds, test_ds = e1_data.get_datasets(data_dir=data_dir, augment_train=augment_train)

    if subset:
        train_ds = Subset(train_ds, range(min(subset, len(train_ds))))
        val_ds = Subset(val_ds, range(min(max(subset // 5, 500), len(val_ds))))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    return train_loader, val_loader, test_loader
