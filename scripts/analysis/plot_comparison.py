"""
Plot probe capacity and learning curve comparisons.

Reads metrics.csv files from checkpoints/probe_*/ directories and produces:
  1. learning_curves.png  — macro accuracy vs training clips, one line per probe
  2. capacity_bars.png    — per-class accuracy at full data, grouped by probe
  3. heatmap.png          — probe × data fraction → macro accuracy

Usage:
    python plot_comparison.py \\
        --checkpoints_dir ./checkpoints \\
        --output_dir      ./plots/comparison \\
        --total_train     5384
"""

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── palette (validated, slots 1-4) ───────────────────────────────────────────
PROBE_COLORS = {
    "linear":    "#2a78d6",  # blue
    "mlp":       "#eb6834",  # orange
    "attn1":     "#1baf7a",  # aqua  (below 3:1 → direct labels required)
    "attentive": "#eda100",  # yellow (below 3:1 → direct labels required)
}
CLASS_COLORS = {
    "left":     "#2a78d6",
    "right":    "#eb6834",
    "straight": "#1baf7a",
    "stop":     "#eda100",
}

PROBE_ORDER   = ["linear", "mlp", "attn1", "attentive"]
PROBE_LABELS  = {"linear": "Linear", "mlp": "MLP", "attn1": "Attn-1", "attentive": "Attn-4"}
CLASS_ORDER   = ["left", "right", "straight", "stop"]

# Chart chrome
SURFACE   = "#fcfcfb"
GRIDLINE  = "#e1e0d9"
AXIS_INK  = "#c3c2b7"
TEXT_PRI  = "#0b0b0b"
TEXT_SEC  = "#52514e"
TEXT_MUT  = "#898781"


def read_metrics(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def macro_acc(row: dict) -> float | None:
    vals = []
    for c in CLASS_ORDER:
        v = row.get(f"acc_{c}", "")
        if v and v != "—":
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else None


def best_row(rows: list[dict]) -> dict | None:
    best, best_mac = None, -1.0
    for r in rows:
        m = macro_acc(r)
        if m is not None and m > best_mac:
            best_mac, best = m, r
    return best


def discover_runs(checkpoints_dir: Path) -> dict:
    """
    Returns { (probe_type, fraction): metrics_rows }.
    Directory pattern: probe_{type} or probe_{type}_f{fraction}
    """
    pattern = re.compile(r"^probe_([a-z0-9]+?)(?:_f([\d.]+))?$")
    runs = {}
    for d in sorted(checkpoints_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        probe = m.group(1)
        frac  = float(m.group(2)) if m.group(2) else 1.0
        csv_path = d / "metrics.csv"
        if not csv_path.exists():
            continue
        rows = read_metrics(csv_path)
        if rows:
            runs[(probe, frac)] = rows
    return runs


def apply_style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS_INK)
    ax.tick_params(colors=TEXT_MUT, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ── 1. Learning curves ────────────────────────────────────────────────────────

def plot_learning_curves(runs: dict, total_train: int, output_dir: Path):
    # Collect (probe, frac) → best macro acc
    data = defaultdict(dict)  # probe → {frac: macro}
    for (probe, frac), rows in runs.items():
        r = best_row(rows)
        if r:
            data[probe][frac] = macro_acc(r)

    probes_present = [p for p in PROBE_ORDER if p in data and len(data[p]) > 1]
    if not probes_present:
        return

    fracs_present = sorted({f for p in probes_present for f in data[p]})

    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=SURFACE)
    apply_style(ax)

    for probe in probes_present:
        fracs = sorted(data[probe].keys())
        xs = [round(f * total_train) for f in fracs]
        ys = [data[probe][f] for f in fracs]
        color = PROBE_COLORS[probe]

        ax.plot(xs, ys, color=color, linewidth=2, marker="o",
                markersize=6, markerfacecolor=SURFACE, markeredgewidth=1.8,
                markeredgecolor=color, zorder=3, clip_on=False)

        # Direct label at rightmost point (relief rule for sub-3:1 series)
        if xs:
            ax.annotate(PROBE_LABELS[probe], xy=(xs[-1], ys[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=color, fontweight="medium")

    ax.axhline(0.25, color=TEXT_MUT, linewidth=1, linestyle="--", zorder=2)
    ax.annotate("chance", xy=(ax.get_xlim()[0], 0.25),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7.5, color=TEXT_MUT)

    ax.set_xlabel("Training clips", fontsize=10, color=TEXT_SEC)
    ax.set_ylabel("Macro-averaged accuracy", fontsize=10, color=TEXT_SEC)

    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Legend (secondary identity channel alongside direct labels)
    handles = [mpatches.Patch(color=PROBE_COLORS[p], label=PROBE_LABELS[p])
               for p in probes_present]
    ax.legend(handles=handles, fontsize=8.5, frameon=False,
              labelcolor=TEXT_SEC, loc="lower right")

    fig.tight_layout()
    out = output_dir / "learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


# ── 2. Capacity bars (full data only) ────────────────────────────────────────

def plot_capacity_bars(runs: dict, output_dir: Path):
    full = {}  # probe → best row at fraction=1.0
    for (probe, frac), rows in runs.items():
        if frac == 1.0:
            r = best_row(rows)
            if r:
                full[probe] = r

    probes_present = [p for p in PROBE_ORDER if p in full]
    if not probes_present:
        return

    n_probes  = len(probes_present)
    n_classes = len(CLASS_ORDER)
    bar_w     = 0.14
    group_gap = 0.2
    group_w   = n_classes * bar_w + group_gap

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    apply_style(ax)

    xticks, xlabels = [], []
    for pi, probe in enumerate(probes_present):
        row = full[probe]
        group_center = pi * group_w
        xticks.append(group_center)
        xlabels.append(PROBE_LABELS[probe])

        for ci, cls in enumerate(CLASS_ORDER):
            x = group_center + (ci - (n_classes - 1) / 2) * bar_w
            val_s = row.get(f"acc_{cls}", "0")
            val   = float(val_s) if val_s and val_s != "—" else 0.0
            color = CLASS_COLORS[cls]
            ax.bar(x, val, width=bar_w * 0.88, color=color, zorder=3,
                   linewidth=0)
            if val >= 0.06:
                ax.text(x, val + 0.01, f"{val:.0%}", ha="center",
                        va="bottom", fontsize=7, color=TEXT_MUT)

        # Macro accuracy as a horizontal line spanning the group
        mac = macro_acc(row)
        if mac is not None:
            half = (n_classes / 2) * bar_w
            ax.hlines(mac, group_center - half + bar_w * 0.1,
                      group_center + half - bar_w * 0.1,
                      colors=TEXT_PRI, linewidth=1.5, linestyle="--", zorder=4)
            ax.text(group_center + half, mac, f" macro {mac:.0%}",
                    va="center", fontsize=7.5, color=TEXT_SEC)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=10, color=TEXT_SEC)
    ax.set_ylabel("Per-class accuracy (recall)", fontsize=10, color=TEXT_SEC)

    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Class legend
    handles = [mpatches.Patch(color=CLASS_COLORS[c], label=c.capitalize())
               for c in CLASS_ORDER]
    ax.legend(handles=handles, fontsize=8.5, frameon=False,
              labelcolor=TEXT_SEC, loc="upper right", ncol=2)

    fig.tight_layout()
    out = output_dir / "capacity_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


# ── 3. Heatmap (probe × fraction) ────────────────────────────────────────────

def plot_heatmap(runs: dict, output_dir: Path):
    # Collect all probes and fractions present
    fracs_present  = sorted({f for (_, f) in runs})
    probes_present = [p for p in PROBE_ORDER
                      if any(p == probe for (probe, _) in runs)]

    if not probes_present or not fracs_present:
        return

    data = np.full((len(probes_present), len(fracs_present)), np.nan)
    for (probe, frac), rows in runs.items():
        if probe not in probes_present:
            continue
        r = best_row(rows)
        if r:
            pi = probes_present.index(probe)
            fi = fracs_present.index(frac)
            data[pi, fi] = macro_acc(r)

    fig, ax = plt.subplots(figsize=(5.5, 3.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, labelsize=9, colors=TEXT_SEC)

    # Sequential blue: #cde2fb (light) → #1c5cab (dark)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq_blue", ["#cde2fb", "#3987e5", "#1c5cab"])

    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Annotate cells
    for pi in range(len(probes_present)):
        for fi in range(len(fracs_present)):
            v = data[pi, fi]
            if not np.isnan(v):
                lum = 0.299 * int("3987e5"[0:2], 16) + 0.587 * int("3987e5"[2:4], 16) + 0.114 * int("3987e5"[4:6], 16)
                text_color = "#ffffff" if v > 0.5 else TEXT_PRI
                ax.text(fi, pi, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="medium")
            else:
                ax.text(fi, pi, "—", ha="center", va="center",
                        fontsize=9, color=TEXT_MUT)

    ax.set_xticks(range(len(fracs_present)))
    ax.set_xticklabels([f"{f:.0%}" for f in fracs_present])
    ax.set_yticks(range(len(probes_present)))
    ax.set_yticklabels([PROBE_LABELS[p] for p in probes_present])
    ax.set_xlabel("Training data fraction", fontsize=10, color=TEXT_SEC)


    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=TEXT_MUT)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    cbar.outline.set_visible(False)

    fig.tight_layout()
    out = output_dir / "heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=Path, required=True,
                        help="Directory containing probe_*/ subdirectories")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--total_train", type=int, default=5384,
                        help="Total training clips at fraction=1.0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.checkpoints_dir)
    if not runs:
        print(f"No probe runs found in {args.checkpoints_dir}")
        raise SystemExit(1)

    print(f"Found {len(runs)} run(s):")
    for (probe, frac) in sorted(runs):
        r = best_row(runs[(probe, frac)])
        mac = f"{macro_acc(r):.3f}" if r else "n/a"
        print(f"  {probe:10s}  frac={frac:.2f}  macro={mac}")
    print()

    plot_learning_curves(runs, args.total_train, args.output_dir)
    plot_capacity_bars(runs, args.output_dir)
    plot_heatmap(runs, args.output_dir)

    print("Done.")
