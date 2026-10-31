"""Ba kien truc phan loai cho E1, dung chung dau vao Fashion-MNIST (1x28x28, 10 lop)."""

from __future__ import annotations

import torch
from torch import nn

NUM_CLASSES = 10
IMG_SIZE = 28
IN_CHANNELS = 1


class SoftmaxClassifier(nn.Module):
    """Flatten anh + anh xa tuyen tinh. Softmax nam trong CrossEntropyLoss (logits -> softmax -> NLL)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(IMG_SIZE * IMG_SIZE, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(x))


class MLPClassifier(nn.Module):
    """Fully-connected nhieu tang tren vector flatten, dau ra softmax (qua CrossEntropyLoss)."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.flatten = nn.Flatten()

        layers: list[nn.Module] = []
        in_dim = IMG_SIZE * IMG_SIZE
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x))


class CNNClassifier(nn.Module):
    """2 khoi Conv-BN-ReLU-MaxPool + head FC, phu hop anh nho 28x28."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


MODEL_REGISTRY = {
    "softmax": SoftmaxClassifier,
    "mlp": MLPClassifier,
    "cnn": CNNClassifier,
}


def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}', available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy = torch.randn(4, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    for name in MODEL_REGISTRY:
        model = build_model(name)
        out = model(dummy)
        print(f"{name:8s} out={tuple(out.shape)} params={count_parameters(model):,}")
