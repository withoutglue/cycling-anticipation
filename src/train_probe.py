"""
Train an attentive probe on top of frozen V-JEPA 2 ViT-g encoder
for 4-class cyclist maneuver anticipation.

Architecture:
    Frozen ViT-g encoder (1B params, embed_dim=1408)
    + Frozen predictor imagining future tokens at anticipation_gap_s into the future
    → concatenated (B, N+N_pred, 1408) tokens
    → AttentivePooler (4 transformer blocks, 1 query token)
    → Linear classifier (1408 → 4 classes)

Training uses weighted cross-entropy to handle class imbalance
(straight dominates; stop/left/right are minority classes).

Per-class accuracy is reported at each validation epoch.

Usage:
    python train_probe.py \\
        --manifest manifest.csv \\
        --train_sessions train_sessions.txt \\
        --val_sessions val_sessions.txt \\
        --vjepa2_dir /path/to/vjepa2 \\
        --output_dir probe_checkpoints/

    # train_sessions.txt / val_sessions.txt: one session_id per line
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

EMBED_DIM = 1408   # ViT-g output dimension
NUM_CLASSES = 4
PROBE_DEPTH = 4
PROBE_HEADS = 16


ANTICIPATION_TIME_S = 1.0   # fixed predictor horizon (matches paper)


def load_models(vjepa2_dir: Path, device: torch.device, use_predictor: bool,
                checkpoint_path: str | None = None):
    sys.path.insert(0, str(vjepa2_dir))
    from src.hub.backbones import vjepa2_vit_giant

    if use_predictor:
        from evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar import AnticipativeWrapper
        # num_frames=64 gives predictor positional space for 32 context frames
        # (4096 tokens) + 1s future (256 tokens) with room to spare.
        encoder, predictor = vjepa2_vit_giant(pretrained=True, num_frames=64,
                                               checkpoint_path=checkpoint_path)
        model = AnticipativeWrapper(
            encoder=encoder,
            predictor=predictor,
            frames_per_second=8,   # 32 frames / 4s = 8 fps effective
            crop_size=256,
            patch_size=16,
            tubelet_size=2,
            num_output_frames=2,
            num_steps=1,
        )
    else:
        encoder, _ = vjepa2_vit_giant(pretrained=True, num_frames=32,
                                       checkpoint_path=checkpoint_path)
        model = encoder

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_probe(embed_dim: int, num_classes: int,
                depth: int, num_heads: int,
                probe_type: str = "attentive") -> nn.Module:
    if probe_type == "linear":
        # Mean-pool tokens → linear classifier. No capacity to memorise.
        return nn.Sequential(
            _MeanPool(),
            nn.Linear(embed_dim, num_classes),
        )
    if probe_type == "mlp":
        return nn.Sequential(
            _MeanPool(),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 4, num_classes),
        )
    from src.models.attentive_pooler import AttentivePooler
    if probe_type == "attn1":
        pooler = AttentivePooler(num_queries=1, embed_dim=embed_dim,
                                 num_heads=num_heads, depth=1)
        return nn.Sequential(pooler, nn.Flatten(1), nn.Linear(embed_dim, num_classes))
    # Default: attentive pooler depth=4 (paper default)
    pooler = AttentivePooler(
        num_queries=1,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
    )
    classifier = nn.Linear(embed_dim, num_classes)
    return nn.Sequential(pooler, nn.Flatten(1), classifier)


class _MeanPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)  # (B, N, D) → (B, D)


def encode_clips(model, clips: torch.Tensor, use_predictor: bool) -> torch.Tensor:
    """
    clips: (B, T, C, H, W) float
    use_predictor=True:  returns (B, N+N_pred, 1408) encoder+predictor tokens
    use_predictor=False: returns (B, N, 1408) encoder tokens only
    ViT-g expects (B, C, T, H, W) — permute accordingly.
    """
    x = clips.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
    with torch.no_grad():
        if use_predictor:
            B = x.shape[0]
            times = torch.full((B,), ANTICIPATION_TIME_S, device=x.device)
            features = model(x, times)
        else:
            features = model(x)
    return features


def stratified_subsample(dataset, fraction: float, seed: int = 42):
    """Return a Subset of dataset keeping `fraction` of each class."""
    from torch.utils.data import Subset
    labels = dataset.df["label"].tolist()
    rng = np.random.default_rng(seed)
    indices = []
    for cls in sorted(set(labels)):
        cls_idx = [i for i, l in enumerate(labels) if l == cls]
        k = max(1, round(len(cls_idx) * fraction))
        indices.extend(rng.choice(cls_idx, size=k, replace=False).tolist())
    return Subset(dataset, sorted(indices))


def per_class_accuracy(preds: torch.Tensor, labels: torch.Tensor,
                       num_classes: int) -> dict[int, float]:
    accs = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            accs[c] = float("nan")
        else:
            accs[c] = (preds[mask] == c).float().mean().item()
    return accs


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Sessions
    train_sessions = Path(args.train_sessions).read_text().splitlines() if args.train_sessions else None
    val_sessions   = Path(args.val_sessions).read_text().splitlines()   if args.val_sessions   else None

    # Datasets
    sys.path.insert(0, str(Path(__file__).parent))

    if args.features:
        from dataset import FeaturesDataset
        train_ds = FeaturesDataset(args.features, sessions=train_sessions, include_flipped=True)
        val_ds   = FeaturesDataset(args.features, sessions=val_sessions,   include_flipped=False)
    else:
        from dataset import ManeuverDataset
        src = dict(hdf5_path=args.hdf5) if args.hdf5 else dict(manifest_path=args.manifest)
        train_ds = ManeuverDataset(**src, augment=True,  sessions=train_sessions)
        val_ds   = ManeuverDataset(**src, augment=False, sessions=val_sessions)

    if args.data_fraction < 1.0:
        train_ds = stratified_subsample(train_ds, args.data_fraction)
        print(f"Subsampled train set to {args.data_fraction:.0%} ({len(train_ds)} clips)")

    print("\nTrain set:")
    train_ds.dataset.summary() if hasattr(train_ds, "dataset") else train_ds.summary()
    print("\nVal set:")
    val_ds.summary()

    # class_weights from the underlying dataset (before subsetting)
    base_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds
    class_weights = base_ds.class_weights().to(device)
    print(f"\nClass weights: {class_weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model — skip encoder loading when using pre-encoded features
    if not args.features:
        model = load_models(Path(args.vjepa2_dir), device, args.use_predictor,
                            checkpoint_path=args.checkpoint)
    else:
        model = None
        sys.path.insert(0, str(Path(args.vjepa2_dir)))  # needed for AttentivePooler import
    probe = build_probe(EMBED_DIM, NUM_CLASSES, PROBE_DEPTH, PROBE_HEADS,
                        probe_type=args.probe_type).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_names = {0: "left", 1: "right", 2: "straight", 3: "stop"}
    best_val_acc = 0.0
    best_val_macro = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── train ────────────────────────────────────────────────────────────
        probe.train()
        train_loss = 0.0
        for clips, labels in train_loader:
            clips  = clips.to(device)
            labels = labels.to(device)

            features = clips if args.features else encode_clips(model, clips, args.use_predictor)
            logits   = probe(features)
            loss     = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ── validate ─────────────────────────────────────────────────────────
        probe.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips  = clips.to(device)
                labels = labels.to(device)
                features = clips if args.features else encode_clips(model, clips, args.use_predictor)
                logits   = probe(features)
                val_loss += criterion(logits, labels).item()
                all_preds.append(logits.argmax(1).cpu())
                all_labels.append(labels.cpu())

        val_loss /= len(val_loader)
        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_acc    = (all_preds == all_labels).float().mean().item()
        per_class  = per_class_accuracy(all_preds, all_labels, NUM_CLASSES)

        print(f"\nEpoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}")
        for c, name in label_names.items():
            acc = per_class[c]
            print(f"  {name:10s}: {acc:.3f}" if not (acc != acc) else f"  {name:10s}: —")

        # Append metrics row
        metrics_path = output_dir / "metrics.csv"
        row = {"epoch": epoch, "train_loss": round(train_loss, 6),
               "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 6),
               **{f"acc_{label_names[c]}": round(per_class[c], 6)
                  if per_class[c] == per_class[c] else float("nan")
                  for c in range(NUM_CLASSES)}}
        write_header = not metrics_path.exists()
        with open(metrics_path, "a", newline="") as mf:
            writer = csv.DictWriter(mf, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        valid_class_accs = [per_class[c] for c in range(NUM_CLASSES)
                            if per_class[c] == per_class[c]]
        val_macro = sum(valid_class_accs) / len(valid_class_accs)
        print(f"  macro_acc={val_macro:.3f}")

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_val_acc   = val_acc
            ckpt = output_dir / "best_probe.pt"
            torch.save({"epoch": epoch, "probe": probe.state_dict(),
                        "val_acc": val_acc, "val_macro": val_macro}, ckpt)
            print(f"  → saved best checkpoint ({ckpt})")

    print(f"\nBest val macro: {best_val_macro:.3f}  (overall: {best_val_acc:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train attentive probe on V-JEPA 2 encoder.")
    parser.add_argument("--features",       type=str, default=None,
                        help="features.h5 from encode_features.py (fastest — no encoder at train time)")
    parser.add_argument("--hdf5",           type=str, default=None,
                        help="clips.h5 from pack_clips_hdf5.py (recommended for CSC)")
    parser.add_argument("--manifest",       type=str, default=None,
                        help="manifest.csv from extract_clips.py (local testing fallback)")
    parser.add_argument("--vjepa2_dir",     type=str, required=True,
                        help="Path to vjepa2 repo root")
    parser.add_argument("--output_dir",     type=str, required=True)
    parser.add_argument("--train_sessions", type=str, default=None,
                        help="Text file with one train session_id per line")
    parser.add_argument("--val_sessions",   type=str, default=None,
                        help="Text file with one val session_id per line")
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--wd",             type=float, default=1e-4)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--checkpoint",    type=str, default=None,
                        help="Local path to vitg.pt checkpoint (avoids download on CSC)")
    parser.add_argument("--use_predictor", action="store_true",
                        help="Use frozen predictor (encoder+predictor) instead of encoder-only")
    parser.add_argument("--probe_type", default="attentive",
                        choices=["attentive", "linear", "mlp", "attn1"],
                        help="Probe architecture: attentive (4-layer, default), attn1 (1-layer), mlp, linear")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (stratified per class). Default=1.0 (all)")

    args = parser.parse_args()
    train(args)
