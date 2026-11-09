"""Thứ tự chạy A1 trên Colab và chọn LR backbone theo protocol A1.1 mục 7.

Protocol: với full finetune, thử LR backbone {5e-5, 1e-4, 3e-4} trên validation (seed 42,
cùng ngân sách cho cả hai kiến trúc), rồi dùng giá trị tốt nhất cho freeze một phần. Nhóm
dùng luôn giá trị đó cho các seed lặp của full finetune, để mean ± std đo ở cùng một cấu hình.

Run 1e-4 seed 42 chính là R4_s42 / R8_s42; hai LR còn lại ghi vào thư mục có hậu tố _lr
(ví dụ R4_s42_lr0.0003), giống quy ước của train.run_dir_name. Module này không import torch.
"""
import json
import os

from .config import MODES

SWEEP_LRS = (5e-5, 1e-4, 3e-4)
DEFAULT_LR = MODES["full"].lr_backbone
FULL = {"resnet50": "R4", "deit_s": "R8"}
PARTIAL = {"resnet50": "R3", "deit_s": "R7"}
TUNED = {run: model for group in (FULL, PARTIAL) for model, run in group.items()}  # run_id -> model
SELECTION_FILE = "lr_selection.json"


def lr_override(lr):
    """None khi bằng LR mặc định, để thư mục và config trùng với các run đã có."""
    return None if lr is None or lr == DEFAULT_LR else lr


def run_dir(run_id, seed, lr=None):
    lr = lr_override(lr)
    return f"{run_id}_s{seed}" + (f"_lr{lr:g}" if lr is not None else "")


def _metrics(out_root, name):
    path = os.path.join(out_root, name, "metrics.json")
    if os.path.isfile(path):
        with open(path) as stream:
            return json.load(stream)
    return None


def sweep_table(out_root):
    """{model: {lr: best_val_top1 hoặc None nếu chưa chạy xong}}."""
    return {model: {lr: (_metrics(out_root, run_dir(run, 42, lr)) or {}).get("best_val_top1")
                    for lr in SWEEP_LRS}
            for model, run in FULL.items()}


def selected_lr(out_root, write=False):
    """LR backbone có val top-1 cao nhất cho từng mô hình; None khi dò LR chưa đủ."""
    table = sweep_table(out_root)
    if any(v is None for row in table.values() for v in row.values()):
        return None
    selection = {model: max(row, key=row.get) for model, row in table.items()}
    if write:
        with open(os.path.join(out_root, SELECTION_FILE), "w") as stream:
            json.dump({"selected_lr_backbone": selection,
                       "best_val_top1": {m: {f"{lr:g}": v for lr, v in row.items()} for m, row in table.items()},
                       "rule": "full finetune seed 42, chọn theo validation top-1 (protocol A1.1 mục 7)"},
                      stream, indent=2)
    return selection


def jobs():
    """(run_id, seed, cần LR đã chọn?) theo thứ tự ưu tiên.

    Seed 42 của đủ 8 run trước (dò LR nằm ở giữa vì R3/R7 cần kết quả của nó), rồi các seed lặp.
    """
    sweep = [(run, 42, lr) for run in FULL.values() for lr in SWEEP_LRS]
    primary = [("R2", 42, None), ("R6", 42, None)] + sweep
    primary += [("R3", 42, "tuned"), ("R7", 42, "tuned"), ("R1", 42, None), ("R5", 42, None)]
    replicates = [(run, seed, None if run in ("R2", "R6") else "tuned")
                  for seed in (43, 44) for run in ("R2", "R6", "R3", "R7", "R4", "R8")]
    return primary + replicates


def plan(out_root):
    """Danh sách job kèm LR cụ thể và trạng thái: xong / dở / chưa chạy / chờ dò LR."""
    selection = selected_lr(out_root)
    rows = []
    for run, seed, lr in jobs():
        if lr == "tuned":
            lr = selection[TUNED[run]] if selection else None
            blocked = selection is None
        else:
            blocked = False
        name = run_dir(run, seed, lr)
        directory = os.path.join(out_root, name)
        if os.path.isfile(os.path.join(directory, "metrics.json")):
            state = "xong"
        elif blocked:
            state = "chờ dò LR"
        elif os.path.isfile(os.path.join(directory, "last.pt")):
            state = "dở"
        else:
            state = "chưa chạy"
        rows.append(dict(run_id=run, seed=seed, lr=lr_override(lr), dir=name, state=state))
    return rows


def next_job(out_root, skip=()):
    """Job kế tiếp có thể chạy ngay (tính lại sau mỗi run vì dò LR mở khóa R3/R7)."""
    for row in plan(out_root):
        if row["state"] in ("dở", "chưa chạy") and row["dir"] not in skip:
            return row
    return None


def selected_dirs(out_root):
    """Tên thư mục được đưa vào bảng chính cho các run có dò LR; None nếu chưa chọn LR."""
    selection = selected_lr(out_root)
    if selection is None:
        return None
    return {run_dir(run, seed, selection[model])
            for run, model in TUNED.items() for seed in (42, 43, 44)}
