"""Tổng hợp kết quả các run thành bảng + biểu đồ cho báo cáo A1.2.

    python -m src.summarize --out_root runs --fig_dir figures

Sinh: summary.csv, summary.md, figures/*.png
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODE_ORDER = ["scratch", "freeze_all", "partial", "full"]
MODE_LABEL = {"scratch": "From-scratch", "freeze_all": "Freeze toàn bộ",
              "partial": "Freeze một phần", "full": "Full finetune"}


def load_runs(out_root, include_smoke=False):
    rows = []
    for path in glob.glob(os.path.join(out_root, "*", "metrics.json")):
        d = os.path.basename(os.path.dirname(path))
        if d.endswith("_smoke") and not include_smoke:
            continue
        if "_lr" in d:  # run dò LR, không đưa vào bảng chính
            continue
        with open(path) as f:
            m = json.load(f)
        m["dir"] = os.path.dirname(path)
        rows.append(m)
    if not rows:
        raise SystemExit(f"Không tìm thấy metrics.json trong {out_root}")
    return pd.DataFrame(rows)


def fmt(mean, std, n, scale=100, digits=2):
    if n > 1 and not np.isnan(std):
        return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f}"


def summary_table(df):
    agg = df.groupby(["run_id", "model", "mode"]).agg(
        n_seeds=("seed", "count"),
        val_top1=("best_val_top1", "mean"),
        top1=("test_top1", "mean"), top1_std=("test_top1", "std"),
        top5=("test_top5", "mean"), top5_std=("test_top5", "std"),
        f1=("test_macro_f1", "mean"), f1_std=("test_macro_f1", "std"),
        trainable=("params_trainable", "first"), total=("params_total", "first"),
        train_min=("train_time_s", lambda s: s.mean() / 60),
        peak_mem=("peak_mem_gb", "mean"), tput=("throughput_img_s", "mean"),
        gpu=("gpu", "first"),
    ).reset_index().sort_values("run_id")
    return agg


def to_markdown(agg):
    lines = ["| Run | Mô hình | Chế độ | Seeds | Top-1 (%) | Top-5 (%) | Macro-F1 (%) | Trainable | "
             "Thời gian train (phút) | VRAM đỉnh (GB) | Ảnh/s | GPU |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r.run_id} | {r.model} | {MODE_LABEL.get(r['mode'], r['mode'])} | {r.n_seeds} | "
            f"{fmt(r.top1, r.top1_std, r.n_seeds)} | {fmt(r.top5, r.top5_std, r.n_seeds)} | "
            f"{fmt(r.f1, r.f1_std, r.n_seeds)} | {r.trainable / 1e6:.2f}M | {r.train_min:.1f} | "
            f"{r.peak_mem:.1f} | {r.tput:.0f} | {r.gpu} |")
    return "\n".join(lines)


def plot_curves(df, fig_dir):
    seed0 = df.sort_values("seed").groupby("run_id").first().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for _, r in seed0.iterrows():
        h = pd.read_csv(os.path.join(r["dir"], "history.csv"))
        label = f"{r.run_id} {r.model}/{r['mode']}"
        axes[0].plot(h.epoch, h.val_loss, label=label)
        axes[1].plot(h.epoch, h.val_top1 * 100, label=label)
    axes[0].set(title="Validation loss", xlabel="epoch")
    axes[1].set(title="Validation top-1 (%)", xlabel="epoch")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "curves.png"), dpi=150)
    plt.close(fig)


def plot_tradeoff(agg, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    markers = {m: k for m, k in zip(MODE_ORDER, ["o", "s", "^", "D"])}
    for model, g in agg.groupby("model"):
        for _, r in g.iterrows():
            mk = markers.get(r["mode"], "o")
            axes[0].scatter(r.trainable, r.top1 * 100, marker=mk, s=60, label=f"{model}/{r['mode']}")
            axes[1].scatter(r.train_min, r.top1 * 100, marker=mk, s=60, label=f"{model}/{r['mode']}")
    axes[0].set(xscale="log", xlabel="Tham số trainable", ylabel="Test top-1 (%)", title="Accuracy vs tham số trainable")
    axes[1].set(xscale="log", xlabel="Thời gian train (phút)", ylabel="Test top-1 (%)", title="Accuracy vs thời gian train")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "tradeoff.png"), dpi=150)
    plt.close(fig)


def plot_pretrain_gap(agg, fig_dir):
    """Khoảng cách (pretrained tốt nhất - from-scratch) của từng mô hình."""
    rows = []
    for model, g in agg.groupby("model"):
        sc = g[g["mode"] == "scratch"]
        pt = g[g["mode"] != "scratch"]
        if len(sc) and len(pt):
            rows.append((model, sc.top1.iloc[0] * 100, pt.sort_values("val_top1", ascending=False).iloc[0].top1 * 100))
    if not rows:
        return None
    gap = pd.DataFrame(rows, columns=["model", "scratch", "best_pretrained"])
    gap["gap"] = gap.best_pretrained - gap.scratch
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(gap))
    ax.bar(x - 0.2, gap.scratch, 0.4, label="From-scratch")
    ax.bar(x + 0.2, gap.best_pretrained, 0.4, label="Pretrained tốt nhất")
    for i, r in gap.iterrows():
        ax.text(i, r.best_pretrained + 1, f"+{r.gap:.1f}", ha="center")
    ax.set_xticks(x, gap.model)
    ax.set(ylabel="Test top-1 (%)", title="Mức lợi của pretrain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pretrain_gap.png"), dpi=150)
    plt.close(fig)
    return gap


def top_confusions(cm, classes, k=10):
    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:k]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, cm.shape)
        if cm[i, j] == 0:
            break
        out.append((classes[i], classes[j], int(cm[i, j])))
    return out


def confusion_report(df, classes, fig_dir):
    """Confusion matrix của run tốt nhất mỗi mô hình + 10 cặp lớp nhầm nhiều nhất."""
    lines = []
    for model, g in df.groupby("model"):
        best = g.sort_values("best_val_top1", ascending=False).iloc[0]
        cm = np.load(os.path.join(best["dir"], "confusion.npy"))
        fig, ax = plt.subplots(figsize=(8, 8))
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="viridis")
        ax.set(title=f"Confusion matrix (chuẩn hóa) - {model} {best.run_id} s{best.seed}",
               xlabel="Dự đoán", ylabel="Nhãn thật")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"confusion_{model}.png"), dpi=150)
        plt.close(fig)
        lines.append(f"\n### {model} ({best.run_id}, seed {best.seed}, top-1 {best.test_top1 * 100:.2f}%)\n")
        lines.append("| Nhãn thật | Bị đoán thành | Số ảnh |\n|---|---|---|")
        for a, b, c in top_confusions(cm, classes):
            lines.append(f"| {a} | {b} | {c} |")
    return "\n".join(lines)


def misclassified_grid(run_dir, data_root, classes, out_path, n=24, seed=0):
    """Lưới ảnh test bị đoán sai của một run (dùng cho phần phân tích lỗi)."""
    from PIL import Image
    from torchvision import datasets

    base = datasets.Food101(root=data_root, split="test")
    d = np.load(os.path.join(run_dir, "test_preds.npz"))
    pred, lab = d["top5"][:, 0], d["labels"]
    wrong = np.where(pred != lab)[0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(wrong, size=min(n, len(wrong)), replace=False)
    if len(pick) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(.5, .5, "No misclassified images", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return
    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, pick):
        ax.imshow(Image.open(base._image_files[i]).convert("RGB").resize((224, 224)))
        ax.set_title(f"thật: {classes[lab[i]]}\nđoán: {classes[pred[i]]}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default="runs")
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--fig_dir", default="figures")
    p.add_argument("--include_smoke", action="store_true")
    args = p.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    df = load_runs(args.out_root, args.include_smoke)
    df.to_csv("runs_all.csv", index=False)
    agg = summary_table(df)
    agg.to_csv("summary.csv", index=False)
    md = ["# Kết quả A1\n", to_markdown(agg)]

    plot_curves(df, args.fig_dir)
    plot_tradeoff(agg, args.fig_dir)
    gap = plot_pretrain_gap(agg, args.fig_dir)
    if gap is not None:
        md.append("\n## Khoảng cách pretrained - from-scratch\n")
        md.append(gap.round(2).to_markdown(index=False))

    with open(os.path.join(args.splits_dir, "classes.json")) as f:
        classes = json.load(f)
    md.append("\n## 10 cặp lớp nhầm nhiều nhất\n")
    md.append(confusion_report(df, classes, args.fig_dir))

    with open("summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
