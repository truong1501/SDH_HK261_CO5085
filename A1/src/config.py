"""Cấu hình thí nghiệm A1, bám đúng protocol A1.1 (reports/A1.1_Dataset-Protocol.pdf).

Mỗi run = (mô hình, chế độ). Cặp chính R1-R8 (ResNet-50 vs DeiT-S),
cặp mở rộng X1-X8 (ConvNeXt-T vs Swin-T).
"""
from dataclasses import dataclass, asdict, field
from typing import List, Optional

NUM_CLASSES = 101
IMG_SIZE = 224
BATCH_SIZE = 256

# Tiền tố tên tham số được train ở chế độ "freeze một phần" (ngoài head).
# Head (model.get_classifier()) luôn được train ở mọi chế độ.
MODELS = {
    "resnet50": dict(
        timm_name="resnet50.a1_in1k",
        kind="CNN",
        # freeze stem + layer1-3, train layer4 (~60% tham số backbone)
        partial_trainable=["layer4."],
    ),
    "deit_s": dict(
        timm_name="deit_small_patch16_224.fb_in1k",
        kind="ViT",
        # freeze patch_embed, pos_embed, cls_token, block 0-5; train block 6-11 + norm cuối
        partial_trainable=[f"blocks.{i}." for i in range(6, 12)] + ["norm."],
    ),
    "convnext_t": dict(
        timm_name="convnext_tiny.fb_in1k",
        kind="CNN",
        partial_trainable=["stages.3.", "head."],
    ),
    "swin_t": dict(
        timm_name="swin_tiny_patch4_window7_224.ms_in1k",
        kind="Transformer",
        partial_trainable=["layers.3.", "norm.", "head."],
    ),
}


@dataclass
class ModeConfig:
    name: str
    pretrained: bool
    freeze: str            # "none" | "all" | "partial"
    epochs: int
    lr_backbone: Optional[float]
    lr_head: float
    warmup_epochs: int
    weight_decay: float
    drop_path: float
    aug: str               # "light" | "medium" | "strong"
    mixup: bool = False    # Mixup 0.8 / CutMix 1.0 + label smoothing 0.1


# Bảng siêu tham số mục 7 của protocol
MODES = {
    "scratch": ModeConfig("scratch", False, "none", 100, 5e-4, 5e-4, 5, 0.05, 0.1, "strong", mixup=True),
    "freeze_all": ModeConfig("freeze_all", True, "all", 30, None, 1e-3, 0, 0.0, 0.0, "light"),
    "partial": ModeConfig("partial", True, "partial", 15, 1e-4, 1e-3, 2, 0.05, 0.1, "medium"),
    "full": ModeConfig("full", True, "none", 15, 1e-4, 1e-3, 2, 0.05, 0.1, "medium"),
}

RUNS = {
    "R1": ("resnet50", "scratch"), "R2": ("resnet50", "freeze_all"),
    "R3": ("resnet50", "partial"), "R4": ("resnet50", "full"),
    "R5": ("deit_s", "scratch"), "R6": ("deit_s", "freeze_all"),
    "R7": ("deit_s", "partial"), "R8": ("deit_s", "full"),
    "X1": ("convnext_t", "scratch"), "X2": ("convnext_t", "freeze_all"),
    "X3": ("convnext_t", "partial"), "X4": ("convnext_t", "full"),
    "X5": ("swin_t", "scratch"), "X6": ("swin_t", "freeze_all"),
    "X7": ("swin_t", "partial"), "X8": ("swin_t", "full"),
}

# Run pretrained của cặp chính chạy 3 seed; còn lại 1 seed
MULTI_SEED_RUNS = {"R2", "R3", "R4", "R6", "R7", "R8"}
SEEDS_MULTI = [42, 43, 44]
SEEDS_SINGLE = [42]


def default_seeds(run_id: str) -> List[int]:
    return SEEDS_MULTI if run_id in MULTI_SEED_RUNS else SEEDS_SINGLE


@dataclass
class RunConfig:
    run_id: str
    seed: int
    model_key: str
    timm_name: str
    mode: dict
    batch_size: int = BATCH_SIZE
    img_size: int = IMG_SIZE
    num_classes: int = NUM_CLASSES
    smoke: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def build_run_config(run_id, seed, epochs=None, lr_backbone=None, batch_size=None, smoke=False):
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    if lr_backbone is not None and lr_backbone <= 0:
        raise ValueError("lr_backbone must be positive")
    model_key, mode_key = RUNS[run_id]
    mode = ModeConfig(**asdict(MODES[mode_key]))
    notes = []
    if epochs is not None and epochs != mode.epochs:
        notes.append(f"epochs override: {mode.epochs} -> {epochs} (khác protocol, phải ghi trong báo cáo)")
        mode.epochs = epochs
    if lr_backbone is not None and mode.lr_backbone is not None:
        notes.append(f"lr_backbone override: {mode.lr_backbone} -> {lr_backbone}")
        mode.lr_backbone = lr_backbone
    if batch_size is not None and batch_size != BATCH_SIZE:
        notes.append(f"batch_size override: {BATCH_SIZE} -> {batch_size}")
    if smoke:
        mode.epochs = 1
        notes.append("SMOKE TEST - không dùng cho kết quả")
    return RunConfig(
        run_id=run_id,
        seed=seed,
        model_key=model_key,
        timm_name=MODELS[model_key]["timm_name"],
        mode=asdict(mode),
        batch_size=batch_size or BATCH_SIZE,
        smoke=smoke,
        notes=notes,
    )
