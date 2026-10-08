"""
Evaluate trained attentive probe on the validation set.
Saves per-sample predictions CSV for downstream analysis with analyze_results.py.

Output CSV columns:
    session_id, maneuver, label, pred, pred_name, correct,
    confidence, prob_left, prob_right, prob_straight, prob_stop,
    anticipation_gap_s

Usage:
    python eval_probe.py \\
        --features    features.h5 \\
        --checkpoint  checkpoints/probe_enc/best_probe.pt \\
        --val_sessions val_sessions.txt \\
        --vjepa2_dir  vjepa2/ \\
        --output      predictions.csv
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

LABEL_MAP   = {"left": 0, "right": 1, "straight": 2, "stop": 3}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
EMBED_DIM   = 1408
NUM_CLASSES = 4
PROBE_DEPTH = 4
PROBE_HEADS = 16


class _MeanPool(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


def build_probe(embed_dim, num_classes, depth, num_heads, probe_type="attentive"):
    if probe_type == "linear":
        return torch.nn.Sequential(
            _MeanPool(),
            torch.nn.Linear(embed_dim, num_classes),
        )
    if probe_type == "mlp":
        return torch.nn.Sequential(
            _MeanPool(),
            torch.nn.Linear(embed_dim, embed_dim // 4),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(embed_dim // 4, num_classes),
        )
    from src.models.attentive_pooler import AttentivePooler
    if probe_type == "attn1":
        pooler = AttentivePooler(num_queries=1, embed_dim=embed_dim,
                                 num_heads=num_heads, depth=1)
        return torch.nn.Sequential(pooler, torch.nn.Flatten(1),
                                   torch.nn.Linear(embed_dim, num_classes))
    pooler = AttentivePooler(num_queries=1, embed_dim=embed_dim,
                             num_heads=num_heads, depth=depth)
    classifier = torch.nn.Linear(embed_dim, num_classes)
    return torch.nn.Sequential(pooler, torch.nn.Flatten(1), classifier)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_sessions = set(Path(args.val_sessions).read_text().splitlines())

    sys.path.insert(0, str(Path(args.vjepa2_dir)))
    probe = build_probe(EMBED_DIM, NUM_CLASSES, PROBE_DEPTH, PROBE_HEADS,
                        probe_type=args.probe_type).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    probe.load_state_dict(ckpt["probe"])
    probe.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.3f})")

    with h5py.File(args.features, "r") as f:
        maneuvers   = [m.decode() if isinstance(m, bytes) else m for m in f["maneuvers"][:]]
        session_ids = [s.decode() if isinstance(s, bytes) else s for s in f["session_ids"][:]]
        labels      = f["labels"][:].tolist()
        is_flipped  = (f["is_flipped"][:].tolist()
                       if "is_flipped" in f else [False] * len(labels))
        ant_gaps    = (f["anticipation_gaps"][:].tolist()
                       if "anticipation_gaps" in f else [float("nan")] * len(labels))

    df_meta = pd.DataFrame({
        "maneuver":           maneuvers,
        "session_id":         session_ids,
        "label":              labels,
        "is_flipped":         is_flipped,
        "anticipation_gap_s": ant_gaps,
        "h5_index":           list(range(len(maneuvers))),
    })
    df_val = df_meta[
        df_meta["session_id"].isin(val_sessions) &
        (~df_meta["is_flipped"].astype(bool)) &
        (df_meta["label"] >= 0)
    ].reset_index(drop=True)
    print(f"Val clips (non-flipped, labeled): {len(df_val)}")

    records = []
    BATCH = args.batch_size
    indices = df_val["h5_index"].tolist()

    with h5py.File(args.features, "r") as f:
        feat_ds = f["features"]
        for start in range(0, len(indices), BATCH):
            batch_rows = df_val.iloc[start:start + BATCH]
            batch_idx  = indices[start:start + BATCH]

            sorted_idx = sorted(set(batch_idx))
            raw = feat_ds[sorted_idx]
            idx_map = {h: pos for pos, h in enumerate(sorted_idx)}
            feats = np.stack([raw[idx_map[i]] for i in batch_idx])

            feats_t = torch.from_numpy(feats.astype(np.float32)).to(device)
            with torch.no_grad():
                logits = probe(feats_t)
                probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            for j, (_, row) in enumerate(batch_rows.iterrows()):
                records.append({
                    "session_id":         row["session_id"],
                    "maneuver":           row["maneuver"],
                    "label":              int(row["label"]),
                    "pred":               int(preds[j]),
                    "pred_name":          LABEL_NAMES[int(preds[j])],
                    "correct":            int(preds[j] == int(row["label"])),
                    "confidence":         float(probs[j].max()),
                    "prob_left":          float(probs[j][0]),
                    "prob_right":         float(probs[j][1]),
                    "prob_straight":      float(probs[j][2]),
                    "prob_stop":          float(probs[j][3]),
                    "anticipation_gap_s": float(row["anticipation_gap_s"]),
                })

            if (start // BATCH + 1) % 10 == 0 or start + BATCH >= len(indices):
                print(f"  {min(start + BATCH, len(indices))}/{len(indices)}")

    out = pd.DataFrame(records)
    out.to_csv(args.output, index=False)

    acc = out["correct"].mean()
    print(f"\nOverall val accuracy: {acc:.3f}  (n={len(out)})")
    print("Per-class:")
    for name, label_id in LABEL_MAP.items():
        mask = out["label"] == label_id
        if mask.sum():
            print(f"  {name:10s}: {out.loc[mask, 'correct'].mean():.3f}  (n={mask.sum()})")
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",     required=True,
                        help="features.h5 from encode_features.py")
    parser.add_argument("--checkpoint",   required=True,
                        help="best_probe.pt from train_probe.py")
    parser.add_argument("--val_sessions", required=True,
                        help="val_sessions.txt — one session_id per line")
    parser.add_argument("--vjepa2_dir",   required=True,
                        help="Path to vjepa2 repo root (for AttentivePooler import)")
    parser.add_argument("--output",       default="predictions.csv")
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--probe_type",   default="attentive",
                        choices=["attentive", "linear", "mlp", "attn1"])
    args = parser.parse_args()
    evaluate(args)
