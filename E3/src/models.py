"""LSTM va GRU classifier cho E3, nhan dau vao la chuoi sinh tu sequences.py."""

from __future__ import annotations

import torch
from torch import nn

NUM_CLASSES = 10
HIDDEN_SIZE = 128


class LSTMClassifier(nn.Module):
    """1 tang LSTM, lay hidden state o buoc cuoi cung lam dac trung cho head phan loai."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.rnn(x)
        return self.head(h_n[-1])


class GRUClassifier(nn.Module):
    """1 tang GRU, lay hidden state o buoc cuoi cung lam dac trung cho head phan loai."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(x)
        return self.head(h_n[-1])


MODEL_REGISTRY = {
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
}


def build_model(name: str, input_size: int, num_classes: int = NUM_CLASSES) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}', available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](input_size=input_size, num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for seq_kind, input_size, seq_len in [("row", 28, 28), ("patch", 16, 49)]:
        dummy = torch.randn(4, seq_len, input_size)
        for name in MODEL_REGISTRY:
            model = build_model(name, input_size)
            out = model(dummy)
            print(f"{name:5s} seq={seq_kind:6s} out={tuple(out.shape)} params={count_parameters(model):,}")
