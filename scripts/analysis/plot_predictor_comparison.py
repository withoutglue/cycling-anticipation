"""
Compare encoder-only vs encoder+predictor features for all 4 probes.
Reads per-class accuracy from metrics.csv (best macro epoch), avoiding
dependency on predictions.csv which may reflect the wrong checkpoint.

Produces:
  predictor_macro.png    — macro accuracy grouped bar chart (4 probes × 2 feature sets)
  predictor_perclass.png — per-class recall panels (one per probe)

Usage:
    python plot_predictor_comparison.py \
        --checkpoints_dir ./Checkpoints \
        --output_dir      ./Checkpoints/plots/predictor
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CLASSES     = ["left", "right", "straight", "stop"]
PROBES      = ["linear", "mlp", "attn1", "attentive"]
PROBE_LABELS = {"linear": "Linear", "mlp": "MLP", "attn1": "Attn-1", "attentive": "Attn-4"}

ENC_COLOR  = "#2a78d6"
PRED_COLOR = "#eb6834"


def best_macro_row(metrics_path: Path) -> dict:
    rows = list(csv.DictReader(open(metrics_path)))
    for r in rows:
        r["macro"] = sum(float(r[f"acc_{c}"]) for c in CLASSES) / len(CLASSES)
    return max(rows, key=lambda r: r["macro"])


def load_results(checkpoints_dir: Path) -> dict:
    """Returns {probe: {enc: {macro, recalls}, pred: {macro, recalls}}}"""
    results = {}
    for probe in PROBES:
        results[probe] = {}
        for feat, dir_name in [("enc", f"probe_{probe}"), ("pred", f"probe_pred_{probe}")]:
            path = checkpoints_dir / dir_name / "metrics.csv"
            if not path.exists():
                print(f"WARNING: missing {path}")
                continue
            row = best_macro_row(path)
            recalls = {c: float(row[f"acc_{c}"]) for c in CLASSES}
            print(f"{feat:4s} {probe:10s}  epoch={row['epoch']:>2s}  "
                  f"macro={row['macro']:.3f}  " +
                  "  ".join(f"{c}={recalls[c]:.3f}" for c in CLASSES))
            results[probe][feat] = {"macro": row["macro"], "recalls": recalls}
    return results


def plot_macro(results: dict, output_dir: Path):
    probes = [p for p in PROBES if "enc" in results[p] and "pred" in results[p]]
    x = np.arange(len(probes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))

    enc_vals  = [results[p]["enc"]["macro"]  for p in probes]
    pred_vals = [results[p]["pred"]["macro"] for p in probes]

    bars_enc  = ax.bar(x - width/2, enc_vals,  width, label="Encoder only",
                       color=ENC_COLOR,  alpha=0.9)
    bars_pred = ax.bar(x + width/2, pred_vals, width, label="Encoder + predictor",
                       color=PRED_COLOR, alpha=0.9)

    for bar in list(bars_enc) + list(bars_pred):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.25)")
    ax.set_xticks(x)
    ax.set_xticklabels([PROBE_LABELS[p] for p in probes])
    ax.set_ylabel("Macro accuracy")
    ax.set_ylim(0, 0.72)
    pass
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = output_dir / "predictor_macro.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_perclass(results: dict, output_dir: Path):
    probes = [p for p in PROBES if "enc" in results[p] and "pred" in results[p]]
    x = np.arange(len(CLASSES))
    width = 0.35

    fig, axes = plt.subplots(1, len(probes), figsize=(14, 4), sharey=True)

    for ax, probe in zip(axes, probes):
        enc_vals  = [results[probe]["enc"]["recalls"][c]  for c in CLASSES]
        pred_vals = [results[probe]["pred"]["recalls"][c] for c in CLASSES]

        ax.bar(x - width/2, enc_vals,  width, label="Encoder only",       color=ENC_COLOR,  alpha=0.9)
        ax.bar(x + width/2, pred_vals, width, label="Encoder + predictor", color=PRED_COLOR, alpha=0.9)

        ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in CLASSES])
        ax.set_title(PROBE_LABELS[probe])
        if probe == probes[0]:
            ax.set_ylabel("Per-class recall")
        ax.set_ylim(0, 0.90)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    pass
    fig.tight_layout()
    path = output_dir / "predictor_perclass.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", default="./Checkpoints")
    parser.add_argument("--output_dir",      default="./Checkpoints/plots/predictor")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoints_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(ckpt_dir)
    plot_macro(results, out_dir)
    plot_perclass(results, out_dir)


if __name__ == "__main__":
    main()
