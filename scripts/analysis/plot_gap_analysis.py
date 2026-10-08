"""
Plot anticipation gap analysis across probes and data fractions.

Reads predictions_*.csv files and produces:
  1. gap_by_probe.png      — macro accuracy by gap bin, one line per probe (full data)
  2. gap_by_fraction.png   — macro accuracy by gap bin, linear + mlp at each data fraction

Usage:
    python plot_gap_analysis.py \\
        --predictions_dir ./Checkpoints \\
        --output_dir      ./plots/gap
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

# ── palette ───────────────────────────────────────────────────────────────────
PROBE_COLORS = {
    "linear":      "#2a78d6",
    "mlp":         "#eb6834",
    "attn1":       "#1baf7a",
    "attentive":   "#eda100",
    "pred_linear":    "#2a78d6",
    "pred_mlp":       "#eb6834",
    "pred_attn1":     "#1baf7a",
    "pred_attentive": "#eda100",
}
PROBE_DASH = {p: (None, None) for p in ["linear", "mlp", "attn1", "attentive"]}
PROBE_DASH.update({p: (5, 3) for p in ["pred_linear", "pred_mlp", "pred_attn1", "pred_attentive"]})
PROBE_ORDER  = ["linear", "mlp", "attn1", "attentive",
                "pred_linear", "pred_mlp", "pred_attn1", "pred_attentive"]
PROBE_LABELS = {
    "linear": "Linear", "mlp": "MLP",
    "attn1": "Attn-1", "attentive": "Attn-4",
    "pred_linear": "Linear+pred", "pred_mlp": "MLP+pred",
    "pred_attn1": "Attn-1+pred", "pred_attentive": "Attn-4+pred",
}

# Sequential blues for data-fraction lines (ordinal, light→dark)
FRAC_COLORS = {0.25: "#86b6ef", 0.50: "#3987e5", 0.75: "#256abf", 1.0: "#1c5cab"}
FRAC_LABELS = {0.25: "25 %", 0.50: "50 %", 0.75: "75 %", 1.0: "100 %"}

CLASS_ORDER  = ["left", "right", "straight", "stop"]

SURFACE  = "#fcfcfb"
GRIDLINE = "#e1e0d9"
AXIS_INK = "#c3c2b7"
TEXT_PRI = "#0b0b0b"
TEXT_SEC = "#52514e"
TEXT_MUT = "#898781"

# Gap bins: (label, lo, hi)  hi=None means open-ended
GAP_BINS = [
    ("<3 s",  0.0,  3.0),
    ("3–4 s", 3.0,  4.0),
    ("4–5 s", 4.0,  5.0),
    ("≥5 s",  5.0,  None),
]
BIN_LABELS = [b[0] for b in GAP_BINS]

# Broad bins: gaps are counted in 1 Hz GPS steps and uncertain by about ±1 s, so
# each bin spans at least two steps. Gaps <= 0 s or > 10 s are detection errors.
BROAD_BINS   = [("1–4 s", 0.0, 4.5), ("5–10 s", 4.5, 10.0)]
BROAD_COLORS = ["#86b6ef", "#1c5cab"]


def assign_bin(gap_s: float) -> int | None:
    for i, (_, lo, hi) in enumerate(GAP_BINS):
        if gap_s >= lo and (hi is None or gap_s < hi):
            return i
    return None


def read_predictions(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compute_gap_macro(rows: list[dict]) -> tuple[list[float | None], list[int]]:
    """
    Returns (macro_per_bin, n_per_bin).
    Macro = mean per-class recall across the four bins.
    """
    # bin → class → (correct, total)
    stats = [[defaultdict(lambda: [0, 0]) for _ in GAP_BINS]
             for _ in range(len(GAP_BINS))]
    bin_cls = [defaultdict(lambda: [0, 0]) for _ in GAP_BINS]

    for r in rows:
        try:
            gap = float(r["anticipation_gap_s"])
        except (ValueError, KeyError):
            continue
        b = assign_bin(gap)
        if b is None:
            continue
        cls = r["maneuver"]
        if cls not in CLASS_ORDER:
            continue
        correct = int(r["correct"])
        bin_cls[b][cls][0] += correct
        bin_cls[b][cls][1] += 1

    macros, ns = [], []
    for b in range(len(GAP_BINS)):
        n_total = sum(v[1] for v in bin_cls[b].values())
        ns.append(n_total)
        class_recalls = []
        for cls in CLASS_ORDER:
            c, t = bin_cls[b][cls]
            if t > 0:
                class_recalls.append(c / t)
        macros.append(sum(class_recalls) / len(class_recalls) if class_recalls else None)

    return macros, ns


def discover_predictions(predictions_dir: Path) -> dict:
    """Returns { (probe, frac): rows }.
    Handles both predictions_linear.csv and predictions_pred_linear.csv.
    """
    pattern = re.compile(r"^predictions_((?:pred_)?[a-z0-9]+?)(?:_f([\d.]+))?\.csv$")
    runs = {}
    for p in sorted(predictions_dir.glob("predictions_*.csv")):
        m = pattern.match(p.name)
        if not m:
            continue
        probe = m.group(1)
        frac  = float(m.group(2)) if m.group(2) else 1.0
        runs[(probe, frac)] = read_predictions(p)
    return runs


def apply_style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS_INK)
    ax.tick_params(colors=TEXT_MUT, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ── 1. Gap accuracy by probe (full data only) ─────────────────────────────────

def plot_gap_by_probe(runs: dict, output_dir: Path):
    probes_present = [p for p in PROBE_ORDER
                      if (p, 1.0) in runs]
    if not probes_present:
        print("No full-data predictions found.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=SURFACE)
    apply_style(ax)

    # Sample counts from first probe (same val set for all)
    _, ns = compute_gap_macro(runs[(probes_present[0], 1.0)])

    xs = np.arange(len(GAP_BINS))

    for probe in probes_present:
        macros, _ = compute_gap_macro(runs[(probe, 1.0)])
        ys = [m if m is not None else np.nan for m in macros]
        color = PROBE_COLORS[probe]
        dash  = PROBE_DASH[probe]

        ax.plot(xs, ys, color=color, linewidth=2, marker="o",
                markersize=6, markerfacecolor=SURFACE,
                markeredgewidth=1.8, markeredgecolor=color,
                dashes=dash if dash[0] else (None, None),
                linestyle="--" if dash[0] else "-",
                zorder=3, clip_on=False)

        last = max((i for i, v in enumerate(ys) if not np.isnan(v)), default=None)
        if last is not None:
            ax.annotate(PROBE_LABELS[probe],
                        xy=(xs[last], ys[last]),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=color,
                        fontweight="medium")

    ax.axhline(0.25, color=TEXT_MUT, linewidth=1, linestyle="--", zorder=2)
    ax.annotate("chance", xy=(xs[0], 0.25), xytext=(-2, 4),
                textcoords="offset points", fontsize=7.5, color=TEXT_MUT)

    # Sample counts as secondary x-axis labels
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{lbl}\n(n={n})" for lbl, n in zip(BIN_LABELS, ns)],
        fontsize=9, color=TEXT_SEC)

    ax.set_ylabel("Macro-averaged accuracy", fontsize=10, color=TEXT_SEC)

    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    enc_probes = [p for p in probes_present if not p.startswith("pred_")]
    handles = [mpatches.Patch(color=PROBE_COLORS[p], label=PROBE_LABELS[p])
               for p in enc_probes]
    from matplotlib.lines import Line2D
    handles += [
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-",  label="Encoder only"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="Encoder + predictor"),
    ]
    ax.legend(handles=handles, fontsize=8.5, frameon=False,
              labelcolor=TEXT_SEC, loc="lower left")

    fig.tight_layout()
    out = output_dir / "gap_by_probe.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


# ── 2. Gap accuracy by data fraction (linear + mlp) ──────────────────────────

def plot_gap_by_fraction(runs: dict, output_dir: Path):
    probe_groups = {}
    for probe in ["linear", "mlp"]:
        fracs = sorted(f for (p, f) in runs if p == probe)
        if fracs:
            probe_groups[probe] = fracs

    if not probe_groups:
        print("No linear/mlp predictions found.")
        return

    n_probes = len(probe_groups)
    fig, axes = plt.subplots(1, n_probes, figsize=(5.5 * n_probes, 4.2),
                             facecolor=SURFACE, sharey=True)
    if n_probes == 1:
        axes = [axes]

    xs = np.arange(len(GAP_BINS))

    for ax, (probe, fracs) in zip(axes, probe_groups.items()):
        apply_style(ax)
        _, ns = compute_gap_macro(runs[(probe, fracs[-1])])

        for frac in fracs:
            macros, _ = compute_gap_macro(runs[(probe, frac)])
            ys = [m if m is not None else np.nan for m in macros]
            color = FRAC_COLORS.get(frac, "#2a78d6")

            ax.plot(xs, ys, color=color, linewidth=2, marker="o",
                    markersize=6, markerfacecolor=SURFACE,
                    markeredgewidth=1.8, markeredgecolor=color,
                    zorder=3, clip_on=False)

            last = max((i for i, v in enumerate(ys) if not np.isnan(v)), default=None)
            if last is not None:
                ax.annotate(FRAC_LABELS[frac],
                            xy=(xs[last], ys[last]),
                            xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=color,
                            fontweight="medium")

        ax.axhline(0.25, color=TEXT_MUT, linewidth=1, linestyle="--", zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"{lbl}\n(n={n})" for lbl, n in zip(BIN_LABELS, ns)],
            fontsize=9, color=TEXT_SEC)
        pass
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        if ax is axes[0]:
            ax.set_ylabel("Macro-averaged accuracy", fontsize=10, color=TEXT_SEC)

        handles = [mpatches.Patch(color=FRAC_COLORS[f], label=FRAC_LABELS[f])
                   for f in fracs]
        ax.legend(handles=handles, fontsize=8.5, frameon=False,
                  labelcolor=TEXT_SEC, loc="lower left")

    fig.tight_layout()
    out = output_dir / "gap_by_fraction.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


# ── 3. Broad bins: per-class recall + macro ──────────────────────────────────

def broad_bin(gap_s: float) -> int | None:
    for i, (_, lo, hi) in enumerate(BROAD_BINS):
        if lo < gap_s <= hi if i == len(BROAD_BINS) - 1 else lo < gap_s < hi:
            return i
    return None


def broad_stats(rows: list[dict]):
    """Returns (recall[bin][class], n[bin][class], n_excluded)."""
    counts = [[[0, 0] for _ in CLASS_ORDER] for _ in BROAD_BINS]
    excluded = 0
    for r in rows:
        if r["maneuver"] not in CLASS_ORDER:
            continue
        b = broad_bin(float(r["anticipation_gap_s"]))
        if b is None:
            excluded += 1
            continue
        c = counts[b][CLASS_ORDER.index(r["maneuver"])]
        c[0] += int(r["correct"])
        c[1] += 1
    recall = [[c / t if t else np.nan for c, t in row] for row in counts]
    n = [[t for _, t in row] for row in counts]
    return recall, n, excluded


def plot_gap_broad(runs: dict, probes: list, out: Path):
    probes = [p for p in probes if (p, 1.0) in runs]
    if not probes:
        return
    _, n, _ = broad_stats(runs[(probes[0], 1.0)])

    groups = [c.capitalize() for c in CLASS_ORDER] + ["Macro"]
    xs = np.append(np.arange(len(CLASS_ORDER)), len(CLASS_ORDER) + 0.4)
    w = 0.36

    fig, axes = plt.subplots(1, len(probes), figsize=(3.6 * len(probes), 3.9),
                             facecolor=SURFACE, sharey=True)
    for ax, probe in zip(axes, probes):
        apply_style(ax)
        recall, _, _ = broad_stats(runs[(probe, 1.0)])
        for bi, (label, _, _) in enumerate(BROAD_BINS):
            vals = recall[bi] + [np.nanmean(recall[bi])]
            x = xs + (bi - 0.5) * w
            ax.bar(x, vals, w * 0.94, color=BROAD_COLORS[bi], zorder=3, linewidth=0)
            ax.text(x[-1], vals[-1] + 0.015, f"{vals[-1]:.2f}", ha="center",
                    va="bottom", fontsize=7.5, color=TEXT_SEC)
        ax.axvline(len(CLASS_ORDER) - 0.3, color=GRIDLINE, linewidth=1, zorder=1)
        ax.axhline(0.25, color=TEXT_MUT, linewidth=1, linestyle="--", zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{g}\n{n[0][i]} / {n[1][i]}" for i, g in enumerate(groups[:-1])]
                           + ["Macro\n "], fontsize=8.5, color=TEXT_SEC)
        ax.set_title(PROBE_LABELS[probe], fontsize=10, color=TEXT_SEC)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].set_ylabel("Recall", fontsize=10, color=TEXT_SEC)

    from matplotlib.lines import Line2D
    handles = [plt.Rectangle((0, 0), 1, 1, color=BROAD_COLORS[i],
                             label=f"Anticipation gap {lbl}")
               for i, (lbl, _, _) in enumerate(BROAD_BINS)]
    handles.append(Line2D([0], [0], color=TEXT_MUT, linestyle="--", label="Chance"))
    fig.legend(handles=handles, fontsize=8.5, frameon=False, labelcolor=TEXT_SEC,
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=Path, required=True,
                        help="Directory containing predictions_*.csv files")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_predictions(args.predictions_dir)
    if not runs:
        print(f"No predictions files found in {args.predictions_dir}")
        raise SystemExit(1)

    print(f"Found {len(runs)} predictions file(s):")
    for (probe, frac) in sorted(runs):
        macros, ns = compute_gap_macro(runs[(probe, frac)])
        mac_str = " | ".join(
            f"{lbl}: {m:.2f}(n={n})" if m is not None else f"{lbl}: —"
            for lbl, m, n in zip(BIN_LABELS, macros, ns))
        print(f"  {probe:<12} frac={frac:.2f}  {mac_str}")
    print()

    plot_gap_by_probe(runs, args.output_dir)
    plot_gap_by_fraction(runs, args.output_dir)

    print("\nBroad bins (excluding gaps <= 0 s or > 10 s):")
    for probe in PROBE_ORDER:
        if (probe, 1.0) not in runs:
            continue
        recall, n, excluded = broad_stats(runs[(probe, 1.0)])
        print(f"  {probe:<15}" + " | ".join(
            f"{lbl}: macro {np.nanmean(recall[i]):.2f} "
            f"(" + " ".join(f"{c[0]}={r:.2f}" for c, r in zip(CLASS_ORDER, recall[i])) + ")"
            for i, (lbl, _, _) in enumerate(BROAD_BINS)))
    print(f"  n per class {CLASS_ORDER}: {BROAD_BINS[0][0]} {n[0]}, "
          f"{BROAD_BINS[1][0]} {n[1]}, excluded {excluded}")
    plot_gap_broad(runs, PROBE_ORDER[:4], args.output_dir / "gap_broad_enc.png")
    plot_gap_broad(runs, PROBE_ORDER[4:], args.output_dir / "gap_broad_pred.png")

    print("Done.")
