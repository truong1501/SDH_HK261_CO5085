"""Tạo mô hình timm, áp dụng chế độ freeze, nhóm tham số cho AdamW."""
import timm
import torch.nn as nn

from .config import MODELS


def build_model(model_key, mode, num_classes=101):
    model = timm.create_model(
        MODELS[model_key]["timm_name"],
        pretrained=mode["pretrained"],
        num_classes=num_classes,
        drop_path_rate=mode["drop_path"],
    )
    apply_freeze(model, model_key, mode["freeze"])
    return model


def head_param_ids(model):
    return {id(p) for p in model.get_classifier().parameters()}


def apply_freeze(model, model_key, freeze):
    head_ids = head_param_ids(model)
    prefixes = MODELS[model_key]["partial_trainable"]
    for name, p in model.named_parameters():
        if freeze == "none":
            p.requires_grad = True
        elif freeze == "all":
            p.requires_grad = id(p) in head_ids
        elif freeze == "partial":
            p.requires_grad = id(p) in head_ids or any(name.startswith(pf) for pf in prefixes)
        else:
            raise ValueError(freeze)


def set_frozen_bn_eval(model):
    """BatchNorm bị đóng băng giữ ở eval() để không cập nhật running mean/var (protocol mục 6)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            params = list(m.parameters())
            if params and not any(p.requires_grad for p in params):
                m.eval()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def param_groups(model, lr_backbone, lr_head, weight_decay):
    """Head dùng LR riêng; bias và tham số norm (ndim<=1) không weight decay."""
    head_ids = head_param_ids(model)
    groups = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in head_ids
        decay = p.ndim > 1
        key = ("head" if is_head else "backbone", decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr_head if is_head else lr_backbone,
                "weight_decay": weight_decay if decay else 0.0,
            }
        groups[key]["params"].append(p)
    for g in groups.values():
        g["base_lr"] = g["lr"]
    return list(groups.values())
