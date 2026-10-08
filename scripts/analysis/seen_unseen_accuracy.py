"""
Macro accuracy on validation clips at intersections seen vs. unseen in training.

Joins predictions_*.csv (row order = labeled validation manifest order, verified
on session and maneuver) to geo_overlap_events.csv from plot_geo_overlap.py.
"Seen" = the merged intersection has at least one training event.

Reports, per probe, for all cities and for Helsinki only (other cities are almost
entirely unseen and London has left-hand traffic, which would confound the split):
macro accuracy seen / unseen, the difference with a 95 % bootstrap interval
(whole intersections resampled, separately within seen and unseen, because clips
at one intersection are correlated), and per-class recall in each group.

Usage:
    python seen_unseen_accuracy.py \\
        --manifest        clips/manifest.csv \\
        --val_sessions    clips/val_sessions.txt \\
        --geo_events      plots/geo/geo_overlap_events.csv \\
        --predictions_dir Checkpoints \\
        --output          plots/geo/seen_unseen_accuracy.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

CLASSES = ["left", "right", "straight", "stop"]
PROBES = ["linear", "mlp", "attn1", "attentive",
          "pred_linear", "pred_mlp", "pred_attn1", "pred_attentive"]
PROBE_LABELS = {"linear": "Linear", "mlp": "MLP", "attn1": "Attn-1", "attentive": "Attn-4",
                "pred_linear": "Linear + pred", "pred_mlp": "MLP + pred",
                "pred_attn1": "Attn-1 + pred", "pred_attentive": "Attn-4 + pred"}

SEEN_COLOR   = "#2a78d6"
UNSEEN_COLOR = "#eb6834"
SURFACE  = "#fcfcfb"
GRIDLINE = "#e1e0d9"
AXIS_INK = "#c3c2b7"
TEXT_SEC = "#52514e"
TEXT_MUT = "#898781"


def apply_style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS_INK)
    ax.tick_params(colors=TEXT_MUT, labelsize=9)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def plot_dumbbell(res: pd.DataFrame, scope: str, out: Path):
    r = res[res["scope"] == scope].set_index("probe")
    probes = [p for p in PROBES if p in r.index]
    # Encoder-only block on top, predictor block below, with a gap between
    ys = {p: -(i + (1 if p.startswith("pred_") else 0)) for i, p in enumerate(probes)}

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    apply_style(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for p in probes:
        y, s, u = ys[p], r.at[p, "macro_seen"], r.at[p, "macro_unseen"]
        ax.plot([u, s], [y, y], color=AXIS_INK, linewidth=2, zorder=2)
        ax.scatter(s, y, s=70, color=SEEN_COLOR, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter(u, y, s=70, color=UNSEEN_COLOR, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        d, lo, hi = (round(r.at[p, k], 2) + 0.0 for k in ("diff", "ci_low", "ci_high"))  # no "-0.00"
        ax.text(0.86, y, f"{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]",
                va="center", fontsize=8.5, color=TEXT_SEC)

    top = max(ys.values())
    ax.text(0.86, top + 0.9, "Seen − unseen (95% CI)", fontsize=8.5, color=TEXT_MUT)
    ax.axvline(0.25, color=TEXT_MUT, linewidth=1, linestyle="--", zorder=1)
    ax.text(0.255, min(ys.values()) - 0.7, "chance", fontsize=7.5, color=TEXT_MUT)

    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([PROBE_LABELS[p] for p in probes], fontsize=9.5, color=TEXT_SEC)
    ax.set_ylim(min(ys.values()) - 0.9, top + 1.3)
    ax.set_xlim(0.2, 0.85)
    ax.set_xlabel("Macro-averaged accuracy", fontsize=10, color=TEXT_SEC)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=8, color=SEEN_COLOR,
                      label="Intersection seen in training"),
               Line2D([0], [0], marker="o", linestyle="", markersize=8, color=UNSEEN_COLOR,
                      label="Intersection not seen in training")]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, labelcolor=TEXT_SEC,
              loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


def plot_perclass(res: pd.DataFrame, counts: dict, scope: str, probes: list, out: Path):
    r = res[res["scope"] == scope].set_index("probe")
    probes = [p for p in probes if p in r.index]
    x = np.arange(len(CLASSES))
    w = 0.36

    fig, axes = plt.subplots(1, len(probes), figsize=(3.4 * len(probes), 3.9),
                             facecolor=SURFACE, sharey=True)
    for ax, p in zip(axes, probes):
        apply_style(ax)
        seen = [r.at[p, f"recall_{c}_seen"] for c in CLASSES]
        unseen = [r.at[p, f"recall_{c}_unseen"] for c in CLASSES]
        ax.bar(x - w / 2, seen, w * 0.94, color=SEEN_COLOR, zorder=3, linewidth=0)
        ax.bar(x + w / 2, unseen, w * 0.94, color=UNSEEN_COLOR, zorder=3, linewidth=0)
        ax.axhline(0.25, color=TEXT_MUT, linewidth=1, linestyle="--", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c.capitalize()}\n{counts['seen'][c]} / {counts['unseen'][c]}"
                            for c in CLASSES], fontsize=8.5, color=TEXT_SEC)
        ax.set_title(PROBE_LABELS[p], fontsize=10, color=TEXT_SEC)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].set_ylabel("Per-class recall", fontsize=10, color=TEXT_SEC)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SEEN_COLOR, label="Seen in training"),
               plt.Rectangle((0, 0), 1, 1, color=UNSEEN_COLOR, label="Not seen in training"),
               Line2D([0], [0], color=TEXT_MUT, linestyle="--", label="Chance")]
    fig.legend(handles=handles, fontsize=8.5, frameon=False, labelcolor=TEXT_SEC,
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.text(0.5, -0.1, "Clips per class: seen / not seen", ha="center",
             fontsize=8, color=TEXT_MUT)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


def macro(correct, maneuver, mask) -> float:
    return float(np.mean([correct[mask & (maneuver == c)].mean()
                          for c in CLASSES if (mask & (maneuver == c)).any()]))


def bootstrap_diff(correct, maneuver, seen, intersection, scope, n_boot, rng):
    # Clips at one intersection are correlated, so resample whole intersections,
    # separately within the seen and unseen groups (seen is an intersection property)
    strata = {}
    for flag in (1, 0):
        ids = np.unique(intersection[scope & (seen == flag)])
        strata[flag] = (ids, {g: np.where(scope & (intersection == g))[0] for g in ids})
    diffs = []
    for _ in range(n_boot):
        m = {}
        for flag, (ids, members) in strata.items():
            i = np.concatenate([members[g] for g in rng.choice(ids, len(ids))])
            m[flag] = macro(correct[i], maneuver[i], np.ones(len(i), bool))
        diffs.append(m[1] - m[0])
    return np.percentile(diffs, [2.5, 97.5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("clips/manifest.csv"))
    parser.add_argument("--val_sessions", type=Path, default=Path("clips/val_sessions.txt"))
    parser.add_argument("--geo_events", type=Path,
                        default=Path("plots/geo/geo_overlap_events.csv"))
    parser.add_argument("--predictions_dir", type=Path, default=Path("Checkpoints"))
    parser.add_argument("--output", type=Path,
                        default=Path("plots/geo/seen_unseen_accuracy.csv"))
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()

    man = pd.read_csv(args.manifest)
    man = man[man["maneuver"].isin(CLASSES)]
    val = set(args.val_sessions.read_text().split())
    mv = man[man["session_id"].isin(val)].reset_index(drop=True)

    geo = pd.read_csv(args.geo_events)
    geo = geo[geo["split"] == "val"].drop_duplicates(["session_id", "crossing_frame"])
    key = mv[["session_id", "crossing_frame", "maneuver"]].merge(
        geo[["session_id", "crossing_frame", "region", "intersection_id", "seen_int"]],
        on=["session_id", "crossing_frame"], how="left", validate="many_to_one")
    if key["seen_int"].isna().any():
        raise ValueError("Some validation clips have no intersection in geo_events")

    maneuver = key["maneuver"].values
    seen = key["seen_int"].values
    intersection = key["intersection_id"].values
    scopes = {"all cities": np.ones(len(key), bool),
              "Helsinki": (key["region"] == "Helsinki").values}

    for name, scope in scopes.items():
        print(f"{name}: n per class (left, right, straight, stop)")
        for label, mask in [("seen", scope & (seen == 1)), ("unseen", scope & (seen == 0))]:
            print(f"  {label:7s}", [int((mask & (maneuver == c)).sum()) for c in CLASSES],
                  f"  {len(np.unique(intersection[mask]))} intersections")

    rng = np.random.default_rng(0)
    rows = []
    for probe in PROBES:
        path = args.predictions_dir / f"predictions_{probe}.csv"
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        pr = pd.read_csv(path)
        if not ((pr["session_id"].values == mv["session_id"].values).all()
                and (pr["maneuver"].values == mv["maneuver"].values).all()):
            raise ValueError(f"{path.name} rows do not align with the validation manifest")
        correct = pr["correct"].values.astype(float)

        for name, scope in scopes.items():
            s_mask, u_mask = scope & (seen == 1), scope & (seen == 0)
            lo, hi = bootstrap_diff(correct, maneuver, seen, intersection, scope,
                                    args.n_boot, rng)
            row = {"probe": probe, "scope": name,
                   "macro_all": macro(correct, maneuver, scope),
                   "macro_seen": macro(correct, maneuver, s_mask),
                   "macro_unseen": macro(correct, maneuver, u_mask),
                   "ci_low": lo, "ci_high": hi}
            row["diff"] = row["macro_seen"] - row["macro_unseen"]
            for c in CLASSES:
                for label, mask in [("seen", s_mask), ("unseen", u_mask)]:
                    m = mask & (maneuver == c)
                    row[f"recall_{c}_{label}"] = correct[m].mean() if m.any() else np.nan
            rows.append(row)

    res = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(args.output, index=False)

    for name in scopes:
        print(f"\n{name}")
        print(f"  {'probe':16s} {'all':>5s} {'seen':>6s} {'unseen':>7s} {'diff':>7s}   95% CI")
        for r in res[res["scope"] == name].itertuples():
            print(f"  {r.probe:16s} {r.macro_all:5.3f} {r.macro_seen:6.3f} {r.macro_unseen:7.3f} "
                  f"{r.diff:+7.3f}   [{r.ci_low:+.2f}, {r.ci_high:+.2f}]")
    print(f"\nSaved {args.output}")

    hel = scopes["Helsinki"]
    counts = {label: {c: int((hel & (seen == flag) & (maneuver == c)).sum()) for c in CLASSES}
              for label, flag in [("seen", 1), ("unseen", 0)]}
    out_dir = args.output.parent
    plot_dumbbell(res, "Helsinki", out_dir / "seen_unseen_macro.png")
    plot_perclass(res, counts, "Helsinki", PROBES[:4], out_dir / "seen_unseen_perclass_enc.png")
    plot_perclass(res, counts, "Helsinki", PROBES[4:], out_dir / "seen_unseen_perclass_pred.png")
