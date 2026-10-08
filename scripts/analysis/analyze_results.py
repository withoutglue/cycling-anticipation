"""
Analyze probe training results and produce plots.

Plots produced (written to --output_dir):
  1. training_curves.png  — loss curves + per-class accuracy over epochs
  2. confusion_matrix.png — normalized confusion matrix on val set
  3. gap_analysis.png     — accuracy per anticipation-gap bin, per class
  4. confidence.png       — confidence distribution: correct vs. incorrect

Usage:
    python analyze_results.py \\
        --metrics     checkpoints/probe_enc/metrics.csv \\
        --predictions predictions.csv \\
        --output_dir  plots/

    # Either --metrics or --predictions may be omitted if not available.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

LABEL_NAMES = ["left", "right", "straight", "stop"]
CLASS_COLORS = {
    "left":     "#4C72B0",
    "right":    "#DD8452",
    "straight": "#55A868",
    "stop":     "#C44E52",
}
GAP_BINS   = [0, 1, 2, 3, float("inf")]
GAP_LABELS = ["<1 s", "1–2 s", "2–3 s", "3+ s"]


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_training_curves(metrics: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss, ax_acc = axes

    # Loss
    ax_loss.plot(metrics["epoch"], metrics["train_loss"], label="train loss", color="#4C72B0")
    ax_loss.plot(metrics["epoch"], metrics["val_loss"],   label="val loss",   color="#DD8452")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Loss curves")
    ax_loss.legend()
    ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Accuracy
    ax_acc.plot(metrics["epoch"], metrics["val_acc"], label="overall",
                color="black", linewidth=2, linestyle="--")
    for name in LABEL_NAMES:
        col = f"acc_{name}"
        if col in metrics.columns:
            ax_acc.plot(metrics["epoch"], metrics[col], label=name,
                        color=CLASS_COLORS[name])
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Validation accuracy per class")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend(ncol=2)
    ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    _save(fig, out_dir / "training_curves.png")


def plot_confusion_matrix(preds: pd.DataFrame, out_dir: Path):
    n = len(LABEL_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for _, row in preds.iterrows():
        cm[int(row["label"]), int(row["pred"])] += 1

    # Row-normalize (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Recall")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=9, color=color)

    fig.tight_layout()
    _save(fig, out_dir / "confusion_matrix.png")


def plot_gap_analysis(preds: pd.DataFrame, out_dir: Path):
    df = preds[preds["anticipation_gap_s"].notna()].copy()
    if df.empty or df["anticipation_gap_s"].isna().all():
        print("  gap_analysis.png skipped — no anticipation_gap_s values in predictions")
        return

    df["gap_bin"] = pd.cut(df["anticipation_gap_s"], bins=GAP_BINS, labels=GAP_LABELS,
                           right=False)

    n_bins = len(GAP_LABELS)
    n_classes = len(LABEL_NAMES)
    bar_width = 0.18
    x = np.arange(n_bins)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7),
                                          gridspec_kw={"height_ratios": [3, 1]})

    for ci, name in enumerate(LABEL_NAMES):
        label_id = ci
        accs = []
        for bin_label in GAP_LABELS:
            mask = (preds["label"] == label_id) & (df["gap_bin"] == bin_label)
            accs.append(preds.loc[mask, "correct"].mean() if mask.sum() > 0 else float("nan"))
        offset = (ci - n_classes / 2 + 0.5) * bar_width
        ax_top.bar(x + offset, accs, width=bar_width, label=name,
                   color=CLASS_COLORS[name], alpha=0.85)

    ax_top.set_ylabel("Accuracy")
    ax_top.set_title("Accuracy by anticipation gap — per class")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(GAP_LABELS)
    ax_top.set_ylim(0, 1)
    ax_top.axhline(0.25, color="gray", linewidth=0.8, linestyle=":", label="chance (0.25)")
    ax_top.legend(ncol=3)

    # Sample counts per bin
    counts = df.groupby("gap_bin", observed=True).size().reindex(GAP_LABELS, fill_value=0)
    ax_bot.bar(x, counts.values, color="#888888", alpha=0.7)
    ax_bot.set_ylabel("Clip count")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(GAP_LABELS)
    ax_bot.set_xlabel("Anticipation gap")

    fig.tight_layout()
    _save(fig, out_dir / "gap_analysis.png")


def plot_confidence(preds: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, len(LABEL_NAMES), figsize=(12, 4), sharey=True)
    for ax, name in zip(axes, LABEL_NAMES):
        label_id = LABEL_NAMES.index(name)
        sub = preds[preds["label"] == label_id]
        correct   = sub.loc[sub["correct"] == 1, "confidence"].values
        incorrect = sub.loc[sub["correct"] == 0, "confidence"].values
        data = [x for x in [correct, incorrect] if len(x)]
        labels_bp = (["correct", "incorrect"] if len(correct) and len(incorrect)
                     else ["correct"] if len(correct) else ["incorrect"])
        colors = ["#55A868", "#C44E52"][:len(data)]
        bp = ax.boxplot(data, tick_labels=labels_bp, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f"{name}\n(n={len(sub)})")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Confidence" if ax == axes[0] else "")

    fig.suptitle("Softmax confidence: correct vs. incorrect predictions", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "confidence.png")


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics:
        metrics = pd.read_csv(args.metrics)
        print(f"Loaded metrics: {len(metrics)} epochs")
        plot_training_curves(metrics, out_dir)
    else:
        print("--metrics not provided, skipping training_curves.png")

    if args.predictions:
        preds = pd.read_csv(args.predictions)
        print(f"Loaded predictions: {len(preds)} clips")
        plot_confusion_matrix(preds, out_dir)
        plot_gap_analysis(preds, out_dir)
        plot_confidence(preds, out_dir)
    else:
        print("--predictions not provided, skipping confusion/gap/confidence plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics",     default=None,
                        help="metrics.csv from train_probe.py")
    parser.add_argument("--predictions", default=None,
                        help="predictions.csv from eval_probe.py")
    parser.add_argument("--output_dir",  default="plots/")
    args = parser.parse_args()
    if not args.metrics and not args.predictions:
        parser.error("Provide at least one of --metrics or --predictions")
    main(args)
