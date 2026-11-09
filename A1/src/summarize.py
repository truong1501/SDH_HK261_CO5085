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
import yaml  # noqa: E402

from .plan import FULL, SWEEP_LRS, TUNED, selected_dirs, selected_lr, sweep_table  # noqa: E402

MODE_ORDER = ["scratch", "freeze_all", "partial", "full"]
MODE_LABEL = {"scratch": "From-scratch", "freeze_all": "Freeze toàn bộ",
              "partial": "Freeze một phần", "full": "Full finetune"}


def load_runs(out_root, include_smoke=False):
    """metrics.json của các run đưa vào bảng chính.

    R3/R4/R7/R8: khi đã dò LR xong, chỉ lấy thư mục chạy ở LR được chọn (src/plan.py);
    các điểm dò LR còn lại chỉ xuất hiện trong lr_sweep. Chưa dò xong thì dùng LR mặc định.
    """
    chosen = selected_dirs(out_root)
    rows = []
    for path in glob.glob(os.path.join(out_root, "*", "metrics.json")):
        d = os.path.basename(os.path.dirname(path))
        if d.endswith("_smoke") and not include_smoke:
            continue
        with open(path) as f:
            m = json.load(f)
        if chosen is not None and m["run_id"] in TUNED and not d.endswith("_smoke"):
            if d not in chosen:
                continue
        elif "_lr" in d:
            continue
        with open(os.path.join(os.path.dirname(path), "config.yaml"), encoding="utf-8") as f:
            m["lr_backbone"] = yaml.safe_load(f)["mode"]["lr_backbone"]
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
        gpu=("gpu", "first"), lr=("lr_backbone", "first"),
    ).reset_index().sort_values("run_id")
    return agg


def to_markdown(agg):
    lines = ["| Run | Mô hình | Chế độ | LR backbone | Seeds | Top-1 (%) | Top-5 (%) | Macro-F1 (%) | Trainable | "
             "Thời gian train (phút) | VRAM đỉnh (GB) | Ảnh/s | GPU |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        lr = "—" if pd.isna(r.lr) else f"{r.lr:g}"
        lines.append(
            f"| {r.run_id} | {r.model} | {MODE_LABEL.get(r['mode'], r['mode'])} | {lr} | {r.n_seeds} | "
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


# ---------------------------------------------------------------------------
# Hình bổ sung cho báo cáo: so sánh theo chế độ, đường cong train/val, per-class,
# mức đồng thuận CNN/ViT (McNemar), hiệu năng tính toán, bảng LaTeX.
# ---------------------------------------------------------------------------
MODEL_COLOR = {"resnet50": "#1f77b4", "deit_s": "#d62728", "convnext_t": "#2ca02c", "swin_t": "#9467bd"}
MODEL_LABEL = {"resnet50": "ResNet-50 (CNN)", "deit_s": "DeiT-S (ViT)",
               "convnext_t": "ConvNeXt-T (CNN)", "swin_t": "Swin-T (Transformer)"}


def _modes(df):
    return [m for m in MODE_ORDER if m in set(df["mode"])]


def _models(df):
    return [m for m in MODEL_LABEL if m in set(df["model"])]


def _grouped_bars(ax, df, value, modes, models, scale=100, ylabel=""):
    """Cột trung bình theo (mode, model), error bar = std qua seed, chấm = từng seed."""
    width = 0.8 / max(1, len(models))
    x = np.arange(len(modes))
    for k, model in enumerate(models):
        offs = x - 0.4 + width * (k + 0.5)
        for xi, mode in zip(offs, modes):
            vals = df[(df.model == model) & (df["mode"] == mode)][value].dropna().to_numpy() * scale
            if not len(vals):
                continue
            ax.bar(xi, vals.mean(), width * 0.9, color=MODEL_COLOR.get(model), alpha=0.85,
                   yerr=vals.std(ddof=1) if len(vals) > 1 else None, capsize=3,
                   label=MODEL_LABEL.get(model, model) if mode == next(
                       m for m in modes if len(df[(df.model == model) & (df["mode"] == m)])) else None)
            ax.scatter(np.full(len(vals), xi), vals, s=10, color="black", zorder=3)
            v = vals.mean()
            ax.text(xi, v, f"{v:.2f}" if abs(v) < 1 else f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [MODE_LABEL[m] for m in modes])
    ax.set_ylabel(ylabel)


def plot_mode_compare(df, fig_dir):
    """Hình chính: CNN vs ViT theo từng chế độ (top-1, macro-F1)."""
    modes, models = _modes(df), _models(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    _grouped_bars(axes[0], df, "test_top1", modes, models, ylabel="Test top-1 (%)")
    _grouped_bars(axes[1], df, "test_macro_f1", modes, models, ylabel="Test macro-F1 (%)")
    axes[0].set_title("Top-1 theo chế độ (cột = TB, vạch = std qua seed, chấm = từng seed)")
    axes[1].set_title("Macro-F1 theo chế độ")
    axes[0].legend(fontsize=8)
    for ax in axes:
        lo = 100 * min(df.test_top1.min(), df.test_macro_f1.min())
        ax.set_ylim(max(0, lo - 10), 100)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "mode_compare.png"), dpi=150)
    plt.close(fig)


def plot_curves_by_mode(df, fig_dir):
    """Mỗi cột một chế độ; hàng trên loss, hàng dưới accuracy; nét liền = train, nét đứt = val."""
    modes, models = _modes(df), _models(df)
    fig, axes = plt.subplots(2, len(modes), figsize=(4.2 * len(modes), 7), squeeze=False)
    for c, mode in enumerate(modes):
        for model in models:
            g = df[(df.model == model) & (df["mode"] == mode)].sort_values("seed")
            if not len(g):
                continue
            h = pd.read_csv(os.path.join(g.iloc[0]["dir"], "history.csv"))
            col, lab = MODEL_COLOR.get(model), MODEL_LABEL.get(model, model)
            axes[0, c].plot(h.epoch, h.train_loss, color=col, label=f"{lab} train")
            axes[0, c].plot(h.epoch, h.val_loss, color=col, ls="--", label=f"{lab} val")
            axes[1, c].plot(h.epoch, h.train_acc * 100, color=col)
            axes[1, c].plot(h.epoch, h.val_top1 * 100, color=col, ls="--")
            best = h.val_top1.idxmax()
            axes[1, c].scatter(h.epoch[best], h.val_top1[best] * 100, color=col, marker="*", s=80, zorder=3)
        axes[0, c].set_title(MODE_LABEL[mode])
        axes[1, c].set_xlabel("epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy (%)  (★ = epoch chọn theo val)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Train (liền) vs validation (đứt), seed nhỏ nhất của mỗi run.\n"
                 "Train acc tính trên ảnh đã augment (from-scratch: Mixup/CutMix) nên có thể thấp hơn val.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "curves_by_mode.png"), dpi=150)
    plt.close(fig)


def _best_per_model(df):
    """Run tốt nhất theo validation (không chọn bằng test) cho mỗi mô hình."""
    return {m: g.sort_values("best_val_top1", ascending=False).iloc[0] for m, g in df.groupby("model")}


def _preds(run_dir):
    d = np.load(os.path.join(run_dir, "test_preds.npz"))
    return d["top5"][:, 0].astype(int), d["labels"].astype(int)


def per_class_report(df, classes, fig_dir, k=8):
    """Accuracy từng lớp của run tốt nhất (theo val) mỗi mô hình; scatter CNN vs ViT."""
    best = _best_per_model(df)
    models = [m for m in _models(df) if m in best][:2]
    acc = {}
    for m in models:
        pred, lab = _preds(best[m]["dir"])
        acc[m] = np.array([(pred[lab == c] == c).mean() if (lab == c).any() else np.nan
                           for c in range(len(classes))])
    table = pd.DataFrame({"class": classes, **{f"acc_{m}": acc[m] for m in models}})
    lines = []
    if len(models) == 2:
        a, b = models
        table["diff"] = table[f"acc_{a}"] - table[f"acc_{b}"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.2]})
        ax = axes[0]
        ax.scatter(table[f"acc_{a}"] * 100, table[f"acc_{b}"] * 100, s=14, alpha=0.7)
        ax.plot([0, 100], [0, 100], "k--", lw=0.8)
        for _, r in pd.concat([table.nlargest(4, "diff"), table.nsmallest(4, "diff")]).iterrows():
            ax.annotate(r["class"], (r[f"acc_{a}"] * 100, r[f"acc_{b}"] * 100), fontsize=7)
        ax.set(xlabel=f"{MODEL_LABEL[a]}: acc/lớp (%)  [{best[a].run_id} s{best[a].seed}]",
               ylabel=f"{MODEL_LABEL[b]}: acc/lớp (%)  [{best[b].run_id} s{best[b].seed}]",
               title="Accuracy từng lớp (dưới đường chéo: CNN tốt hơn)")
        ax = axes[1]
        sel = pd.concat([table[table["diff"] > 0].nlargest(k, "diff"),
                         table[table["diff"] < 0].nsmallest(k, "diff").iloc[::-1]])
        colors = [MODEL_COLOR[a] if d > 0 else MODEL_COLOR[b] for d in sel["diff"]]
        ax.barh(range(len(sel)), sel["diff"] * 100, color=colors)
        ax.set_yticks(range(len(sel)), sel["class"], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        ax.set(xlabel=f"Chênh lệch acc (%): {MODEL_LABEL[a]} − {MODEL_LABEL[b]}",
               title=f"Tối đa {k} lớp mỗi bên có chênh lệch lớn nhất")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "per_class.png"), dpi=150)
        plt.close(fig)
        lines.append(f"\n## Lớp dễ/khó (run tốt nhất theo val)\n")
        for m in models:
            t = table.sort_values(f"acc_{m}")
            lines.append(f"- **{MODEL_LABEL[m]}** khó nhất: " + ", ".join(
                f"{r['class']} ({r[f'acc_{m}'] * 100:.0f}%)" for _, r in t.head(5).iterrows()) +
                "; dễ nhất: " + ", ".join(f"{r['class']} ({r[f'acc_{m}'] * 100:.0f}%)"
                                         for _, r in t.tail(5).iloc[::-1].iterrows()))
    table.to_csv("per_class.csv", index=False)
    return "\n".join(lines)


def mcnemar(a_ok, b_ok):
    """McNemar có hiệu chỉnh liên tục; p xấp xỉ từ chi2(1) = erfc(sqrt(x/2))."""
    import math
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return b, c, stat, math.erfc(math.sqrt(stat / 2))


def agreement_report(df, fig_dir):
    """Cùng chế độ, cùng seed 42: CNN và ViT đúng/sai trên cùng ảnh test như thế nào."""
    models = _models(df)[:2]
    if len(models) < 2:
        return ""
    a, b = models
    rows = []
    for mode in _modes(df):
        ra = df[(df.model == a) & (df["mode"] == mode) & (df.seed == 42)]
        rb = df[(df.model == b) & (df["mode"] == mode) & (df.seed == 42)]
        if not len(ra) or not len(rb):
            continue
        pa, la = _preds(ra.iloc[0]["dir"])
        pb, lb = _preds(rb.iloc[0]["dir"])
        if not np.array_equal(la, lb):
            continue
        oka, okb = pa == la, pb == lb
        only_a, only_b, stat, p = mcnemar(oka, okb)
        n = len(la)
        rows.append(dict(mode=mode, both_ok=(oka & okb).mean(), only_a=only_a / n, only_b=only_b / n,
                         both_wrong=(~oka & ~okb).mean(), same_wrong=((~oka & ~okb) & (pa == pb)).mean(),
                         chi2=stat, p=p))
    if not rows:
        return ""
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.7 * len(t)))
    left = np.zeros(len(t))
    parts = [("both_ok", "Cả hai đúng", "#8fbc8f"), ("only_a", f"Chỉ {MODEL_LABEL[a]} đúng", MODEL_COLOR[a]),
             ("only_b", f"Chỉ {MODEL_LABEL[b]} đúng", MODEL_COLOR[b]), ("both_wrong", "Cả hai sai", "#999999")]
    for key, lab, col in parts:
        ax.barh(range(len(t)), t[key] * 100, left=left, color=col, label=lab)
        for i, v in enumerate(t[key] * 100):
            if v > 3:
                ax.text(left[i] + v / 2, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white")
        left += t[key].to_numpy() * 100
    ax.set_yticks(range(len(t)), [MODE_LABEL[m] for m in t["mode"]])
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), title="Mức đồng thuận CNN vs ViT trên cùng ảnh test (% ảnh, seed 42)")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.savefig(os.path.join(fig_dir, "agreement.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    lines = ["\n## Đồng thuận CNN vs ViT (seed 42, McNemar)\n",
             f"| Chế độ | Cả hai đúng | Chỉ {a} đúng | Chỉ {b} đúng | Cả hai sai | Sai cùng nhãn | χ² | p |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {MODE_LABEL[r['mode']]} | {r.both_ok * 100:.2f}% | {r.only_a * 100:.2f}% | "
                     f"{r.only_b * 100:.2f}% | {r.both_wrong * 100:.2f}% | {r.same_wrong * 100:.2f}% | "
                     f"{r.chi2:.1f} | {r.p:.2g} |")
    return "\n".join(lines)


def plot_efficiency(df, fig_dir):
    """Chi phí tính toán: tham số trainable, thời gian/epoch, VRAM đỉnh, throughput suy luận."""
    modes, models = _modes(df), _models(df)
    specs = [("params_trainable", 1e-6, "Tham số trainable (M)"),
             ("time_per_epoch_s", 1, "Thời gian / epoch (s)"),
             ("peak_mem_gb", 1, "VRAM đỉnh khi train (GB)"),
             ("throughput_img_s", 1, "Throughput suy luận (ảnh/s)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (col, scale, title) in zip(axes, specs):
        _grouped_bars(ax, df, col, modes, models, scale=scale)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle("Freeze toàn bộ: thời gian/epoch chỉ là train head trên feature cache; "
                 "throughput đo forward batch 256 tổng hợp.", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "efficiency.png"), dpi=150)
    plt.close(fig)


def lr_sweep_report(out_root, fig_dir):
    """Val top-1 của full finetune seed 42 theo LR backbone (protocol mục 7); chỉ dùng validation."""
    table = sweep_table(out_root)
    if all(v is None for row in table.values() for v in row.values()):
        return ""
    selection = selected_lr(out_root, write=True) or {}
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, row in table.items():
        pts = [(lr, v * 100) for lr, v in row.items() if v is not None]
        ax.plot(*zip(*pts), marker="o", color=MODEL_COLOR.get(model), label=MODEL_LABEL.get(model, model))
        if model in selection:
            ax.scatter(selection[model], row[selection[model]] * 100, s=160, facecolors="none",
                       edgecolors=MODEL_COLOR.get(model), linewidths=2)
    ax.set(xscale="log", xlabel="LR backbone", ylabel="Validation top-1 (%)", xticks=list(SWEEP_LRS),
           title="Dò LR backbone, full finetune seed 42 (vòng tròn = LR được chọn)")
    ax.set_xticklabels([f"{lr:g}" for lr in SWEEP_LRS])
    ax.minorticks_off()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "lr_sweep.png"), dpi=150)
    plt.close(fig)
    lines = ["\n## Dò LR backbone (full finetune, seed 42, validation top-1 %)\n",
             "| Mô hình | " + " | ".join(f"{lr:g}" for lr in SWEEP_LRS) + " | Chọn |",
             "|---|" + "---|" * (len(SWEEP_LRS) + 1)]
    for model, row in table.items():
        cells = ["—" if v is None else f"{v * 100:.2f}" for v in row.values()]
        pick = f"{selection[model]:g}" if model in selection else "chưa đủ"
        lines.append(f"| {MODEL_LABEL.get(model, model)} | " + " | ".join(cells) + f" | {pick} |")
    return "\n".join(lines)


def to_latex(agg):
    """Bảng booktabs cho báo cáo LaTeX (\\usepackage{booktabs})."""
    def tex(s):
        return s.replace("±", r"$\pm$").replace("_", r"\_").replace("%", r"\%")
    lines = [r"\begin{tabular}{llllrrrrrr}", r"\toprule",
             r"Run & Mô hình & Chế độ & Seeds & Top-1 (\%) & Top-5 (\%) & Macro-F1 (\%) & Trainable (M) "
             r"& Train (phút) & Ảnh/s \\", r"\midrule"]
    for _, r in agg.iterrows():
        lines.append(" & ".join([r.run_id, tex(MODEL_LABEL.get(r.model, r.model)),
                                 tex(MODE_LABEL.get(r["mode"], r["mode"])), str(r.n_seeds),
                                 tex(fmt(r.top1, r.top1_std, r.n_seeds)), tex(fmt(r.top5, r.top5_std, r.n_seeds)),
                                 tex(fmt(r.f1, r.f1_std, r.n_seeds)), f"{r.trainable / 1e6:.2f}",
                                 f"{r.train_min:.1f}", f"{r.tput:.0f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default="runs")
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--fig_dir", default="figures")
    p.add_argument("--data_root", default=None, help="nếu có Food-101: vẽ lưới ảnh test bị đoán sai")
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

    md.append(lr_sweep_report(args.out_root, args.fig_dir))
    plot_mode_compare(df, args.fig_dir)
    plot_curves_by_mode(df, args.fig_dir)
    plot_efficiency(df, args.fig_dir)
    md.append(agreement_report(df, args.fig_dir))
    md.append(per_class_report(df, classes, args.fig_dir))
    with open("summary.tex", "w", encoding="utf-8") as f:
        f.write(to_latex(agg))
    if args.data_root:
        for model, r in _best_per_model(df).items():
            misclassified_grid(r["dir"], args.data_root, classes,
                               os.path.join(args.fig_dir, f"misclassified_{model}.png"))

    with open("summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
