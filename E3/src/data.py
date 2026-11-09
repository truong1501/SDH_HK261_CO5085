"""Du lieu cho E3: dung lai dung split/transform/Dataset da dinh nghia o E1.

Theo quy dinh cua de (dung cung mot tap anh nho cho ca E1-E3), E3 khong tu
tai Fashion-MNIST hay tu chia split rieng, ma import truc tiep module
E1/src/data.py, nen moi anh, moi split deu giong het E1.
"""

from __future__ import annotations

import sys
from pathlib import Path

E1_SRC = Path(__file__).resolve().parent.parent.parent / "E1" / "src"
sys.path.insert(0, str(E1_SRC))

from data import (  # noqa: E402
    CLASS_NAMES,
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    NUM_CLASSES,
    get_dataloaders,
    get_datasets,
)

__all__ = [
    "CLASS_NAMES",
    "FASHION_MNIST_MEAN",
    "FASHION_MNIST_STD",
    "NUM_CLASSES",
    "get_dataloaders",
    "get_datasets",
]
