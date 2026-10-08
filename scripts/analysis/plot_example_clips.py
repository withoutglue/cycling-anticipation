"""
Example validation clips for the right-turn / stop confusion.

One row per case — right correct, right predicted as stop, stop correct, stop
predicted as right — choosing the most confident prediction in each case. Each
row shows five of the 32 model input frames, from ~4 s before the crossing to the
crossing frame. Frames are read from clips.h5 (h5 index = labeled manifest row;
predictions rows = labeled validation manifest rows, verified on session and maneuver).

Usage:
    python plot_example_clips.py \\
        --probe     pred_attentive \\
        --clips_h5  clips/clips.h5 \\
        --output    plots/examples/right_stop_examples.png \\
        --exclude_sessions VID_20240318_132503_00_100_gps   # suspect GPS labels
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ["left", "right", "straight", "stop"]
PROBE_LABELS = {"linear": "Linear", "mlp": "MLP", "attn1": "Attn-1", "attentive": "Attn-4",
                "pred_linear": "Linear + pred", "pred_mlp": "MLP + pred",
                "pred_attn1": "Attn-1 + pred", "pred_attentive": "Attn-4 + pred"}
CASES = [("right", "right"), ("right", "stop"), ("stop", "stop"), ("stop", "right")]

# Clip = 32 frames sampled evenly from the 120 source frames (4 s at 30 fps) before
# the crossing; these indices fall at about -4, -3, -2, -1 and 0 s.
FRAME_IDX = [0, 8, 16, 23, 31]
CLIP_SRC_FRAMES, FPS, N_FRAMES = 120, 30.0, 32

TEXT_SEC = "#52514e"
SURFACE = "#fcfcfb"


def frame_time_s(k: int) -> float:
    return (-CLIP_SRC_FRAMES + k * (CLIP_SRC_FRAMES - 1) / (N_FRAMES - 1)) / FPS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="pred_attentive")
    parser.add_argument("--predictions_dir", type=Path, default=Path("Checkpoints"))
    parser.add_argument("--manifest", type=Path, default=Path("clips/manifest.csv"))
    parser.add_argument("--val_sessions", type=Path, default=Path("clips/val_sessions.txt"))
    parser.add_argument("--clips_h5", type=Path, default=Path("clips/clips.h5"))
    parser.add_argument("--output", type=Path,
                        default=Path("plots/examples/right_stop_examples.png"))
    parser.add_argument("--exclude_sessions", nargs="*", default=[],
                        help="Sessions not to pick examples from (e.g. suspect GPS labels)")
    args = parser.parse_args()

    man = pd.read_csv(args.manifest)
    lab = man[man["maneuver"].isin(CLASSES)].reset_index(drop=True)
    val = set(args.val_sessions.read_text().split())
    h5_idx = lab.index[lab["session_id"].isin(val)].to_numpy()
    mv = lab.loc[h5_idx].reset_index(drop=True)

    pr = pd.read_csv(args.predictions_dir / f"predictions_{args.probe}.csv")
    if not ((pr["session_id"].values == mv["session_id"].values).all()
            and (pr["maneuver"].values == mv["maneuver"].values).all()):
        raise ValueError("Predictions rows do not align with the validation manifest")
    pr["h5_index"] = h5_idx
    pr["crossing_frame"] = mv["crossing_frame"].values

    picks = []
    for true, pred in CASES:
        cand = pr[(pr["maneuver"] == true) & (pr["pred_name"] == pred)
                  & ~pr["session_id"].isin(args.exclude_sessions)]
        if cand.empty:
            raise ValueError(f"No validation clip with true={true}, predicted={pred}")
        row = cand.loc[cand["confidence"].idxmax()]
        picks.append(row)
        print(f"true={true:5s} pred={pred:5s}  n={len(cand):3d}  conf={row['confidence']:.2f}  "
              f"h5={row['h5_index']}  {row['session_id']}  crossing_frame={row['crossing_frame']}  "
              f"gap={row['anticipation_gap_s']:.1f}s  "
              + " ".join(f"p_{c}={row[f'prob_{c}']:.2f}" for c in CLASSES))

    with h5py.File(args.clips_h5, "r") as f:
        frames = [f["clips"][int(r["h5_index"])][FRAME_IDX] for r in picks]

    fig, axes = plt.subplots(len(picks), len(FRAME_IDX),
                             figsize=(2.0 * len(FRAME_IDX) + 1.6, 2.05 * len(picks)),
                             facecolor=SURFACE)
    for ri, (row, clip) in enumerate(zip(picks, frames)):
        for ci, img in enumerate(clip):
            ax = axes[ri, ci]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if ri == 0:
                t = frame_time_s(FRAME_IDX[ci])
                ax.set_title(f"{round(t):+d} s" if round(t) else "0 s (crossing)",
                             fontsize=9, color=TEXT_SEC)
        verdict = "correct" if row["correct"] else "wrong"
        axes[ri, 0].set_ylabel(
            f"True: {row['maneuver']}\nPredicted: {row['pred_name']}\n"
            f"({verdict}, p = {row['confidence']:.2f})\ngap {row['anticipation_gap_s']:.0f} s",
            fontsize=8.5, color=TEXT_SEC, rotation=0, ha="right", va="center", labelpad=8)

    fig.tight_layout(w_pad=0.3, h_pad=0.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {args.output}  ({PROBE_LABELS.get(args.probe, args.probe)})")
