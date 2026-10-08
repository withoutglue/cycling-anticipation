"""
Pre-encode all clips through the frozen ViT-g encoder and save token features
to HDF5. Flipped versions of left/right/stop clips are also encoded (left↔right
label swap for directional classes).

This decouples the frozen encoder pass from probe training, reducing epoch time
from ~40 min to ~10 sec. Standard practice for frozen-encoder probing.

Augmentation strategy:
  - All clips: encode original
  - left / right / stop: also encode horizontally flipped version
    (left↔right label is swapped; stop label is unchanged)
  - straight: original only (4100+ clips; flip adds little for majority class)

Output HDF5 schema:
  /features     (N, num_tokens, 1408)  float16
  /labels       (N,)                   int32
  /maneuvers    (N,)                   str
  /session_ids  (N,)                   str
  /is_flipped   (N,)                   bool

Usage:
    python encode_features.py \\
        --clips_h5   clips.h5 \\
        --vjepa2_dir vjepa2/ \\
        --checkpoint vitg.pt \\
        --output     features.h5

    # With predictor (encoder+predictor tokens concatenated):
    python encode_features.py ... --use_predictor
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

LABEL_MAP     = {"left": 0, "right": 1, "straight": 2, "stop": 3}
FLIP_CLASSES  = {"left", "right", "stop"}
LABEL_FLIP    = {"left": "right", "right": "left", "stop": "stop"}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

ANTICIPATION_TIME_S = 1.0


def load_model(vjepa2_dir, checkpoint_path, use_predictor, device):
    sys.path.insert(0, str(vjepa2_dir))
    from src.hub.backbones import vjepa2_vit_giant

    if use_predictor:
        from evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar import AnticipativeWrapper
        encoder, predictor = vjepa2_vit_giant(
            pretrained=True, num_frames=64, checkpoint_path=checkpoint_path)
        model = AnticipativeWrapper(
            encoder=encoder, predictor=predictor,
            frames_per_second=8, crop_size=256,
            patch_size=16, tubelet_size=2,
            num_output_frames=2, num_steps=1,
        )
    else:
        encoder, _ = vjepa2_vit_giant(
            pretrained=True, num_frames=32, checkpoint_path=checkpoint_path)
        model = encoder

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode_batch(model, clips_uint8: np.ndarray, use_predictor: bool,
                 device: torch.device) -> np.ndarray:
    """
    clips_uint8: (B, 32, 256, 256, 3) uint8
    Returns (B, num_tokens, 1408) float16 numpy array.
    """
    x = torch.from_numpy(clips_uint8).float() / 255.0      # (B, T, H, W, C)
    x = x.permute(0, 1, 4, 2, 3)                           # (B, T, C, H, W)
    x = (x - MEAN) / STD
    x = x.permute(0, 2, 1, 3, 4).to(device)                # (B, C, T, H, W)
    with torch.no_grad():
        if use_predictor:
            B = x.shape[0]
            times = torch.full((B,), ANTICIPATION_TIME_S, device=device)
            feats = model(x, times)
        else:
            feats = model(x)
    return feats.half().cpu().numpy()


def encode_features(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(Path(args.vjepa2_dir), args.checkpoint,
                       args.use_predictor, device)

    # Read all metadata from clips.h5 (not the clip data yet)
    with h5py.File(args.clips_h5, "r") as src:
        maneuvers   = [m.decode() if isinstance(m, bytes) else m
                       for m in src["maneuvers"][:]]
        session_ids = [s.decode() if isinstance(s, bytes) else s
                       for s in src["session_ids"][:]]
        labels_src  = src["labels"][:].tolist()
        N_src       = len(maneuvers)

    # Load anticipation gaps from manifest if provided (h5_index == labeled manifest row index)
    gap_map: dict[int, float] = {}
    if args.manifest:
        mf = pd.read_csv(args.manifest)
        # clips.h5 holds only labeled clips (pack_clips_hdf5.py drops unknown rows),
        # so raw manifest row numbers would be offset by every preceding unknown row
        mf = mf[mf["maneuver"].isin(LABEL_MAP)].reset_index(drop=True)
        if len(mf) != N_src:
            raise ValueError(f"Labeled manifest rows ({len(mf)}) != clips in h5 ({N_src})")
        for idx, row in mf.iterrows():
            try:
                gap_map[int(idx)] = float(row["anticipation_gap_s"])
            except (KeyError, ValueError):
                pass

    # Build index: (h5_index, maneuver, session_id, label, is_flipped, anticipation_gap_s)
    entries = []
    for i, (maneuver, session_id, label) in enumerate(
            zip(maneuvers, session_ids, labels_src)):
        if label < 0:
            continue   # skip unknown
        gap = gap_map.get(i, float("nan"))
        entries.append((i, maneuver, session_id, label, False, gap))
        if maneuver in FLIP_CLASSES:
            flipped_label = LABEL_MAP[LABEL_FLIP[maneuver]]
            entries.append((i, LABEL_FLIP[maneuver], session_id, flipped_label, True, float("nan")))

    N_out = len(entries)
    print(f"Source clips: {N_src}  →  Output entries: {N_out} "
          f"(+{N_out - sum(1 for e in entries if not e[4])} flipped)")
    if gap_map:
        n_gaps = sum(1 for e in entries if not np.isnan(e[5]))
        print(f"Anticipation gaps loaded: {n_gaps}/{sum(1 for e in entries if not e[4])}")

    # Probe a single clip to get num_tokens and embed_dim
    with h5py.File(args.clips_h5, "r") as src:
        probe_clip = src["clips"][0:1]
    probe_feat = encode_batch(model, probe_clip, args.use_predictor, device)
    _, num_tokens, embed_dim = probe_feat.shape
    print(f"Feature shape per clip: ({num_tokens}, {embed_dim})  "
          f"→ total {N_out * num_tokens * embed_dim * 2 / 1e9:.1f} GB fp16")

    str_dt = h5py.special_dtype(vlen=str)
    BATCH = args.batch_size

    with h5py.File(args.output, "w") as dst:
        feat_ds  = dst.create_dataset("features",          shape=(N_out, num_tokens, embed_dim),
                                      dtype=np.float16, chunks=(1, num_tokens, embed_dim))
        label_ds = dst.create_dataset("labels",            shape=(N_out,), dtype=np.int32)
        man_ds   = dst.create_dataset("maneuvers",         shape=(N_out,), dtype=str_dt)
        sess_ds  = dst.create_dataset("session_ids",       shape=(N_out,), dtype=str_dt)
        flip_ds  = dst.create_dataset("is_flipped",        shape=(N_out,), dtype=bool)
        gap_ds   = dst.create_dataset("anticipation_gaps", shape=(N_out,), dtype=np.float32)

        with h5py.File(args.clips_h5, "r") as src:
            clips_ds = src["clips"]

            out_idx = 0
            i = 0
            while i < len(entries):
                batch_entries = entries[i:i + BATCH]

                # Group consecutive entries by h5_index to minimise seeks
                h5_indices = list(dict.fromkeys(e[0] for e in batch_entries))
                clips_raw = clips_ds[h5_indices]   # (k, 32, 256, 256, 3) uint8
                idx_map = {h5i: pos for pos, h5i in enumerate(h5_indices)}

                # Separate original and flipped
                orig_mask  = [not e[4] for e in batch_entries]
                flip_mask  = [e[4]     for e in batch_entries]

                orig_clips = np.stack([clips_raw[idx_map[e[0]]] for e in batch_entries if not e[4]])
                flip_clips = np.stack([clips_raw[idx_map[e[0]]][:, :, ::-1, :].copy()
                                       for e in batch_entries if e[4]]) if any(flip_mask) else None

                orig_feats = encode_batch(model, orig_clips, args.use_predictor, device) if len(orig_clips) else None
                flip_feats = encode_batch(model, flip_clips, args.use_predictor, device) if flip_clips is not None else None

                orig_iter = iter(orig_feats) if orig_feats is not None else iter([])
                flip_iter = iter(flip_feats) if flip_feats is not None else iter([])

                for entry in batch_entries:
                    _, maneuver, session_id, label, is_flipped, gap = entry
                    feat = next(flip_iter) if is_flipped else next(orig_iter)
                    feat_ds[out_idx]  = feat
                    label_ds[out_idx] = label
                    man_ds[out_idx]   = maneuver
                    sess_ds[out_idx]  = session_id
                    flip_ds[out_idx]  = is_flipped
                    gap_ds[out_idx]   = gap if not np.isnan(gap) else np.float32("nan")
                    out_idx += 1

                i += BATCH
                if out_idx % 500 == 0 or i >= len(entries):
                    print(f"  {out_idx}/{N_out}")

    size_gb = Path(args.output).stat().st_size / 1e9
    print(f"\nDone. {args.output} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_h5",    required=True)
    parser.add_argument("--vjepa2_dir",  required=True)
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--manifest",    default=None,
                        help="manifest.csv — used to store anticipation_gap_s in output h5")
    parser.add_argument("--use_predictor", action="store_true")
    parser.add_argument("--batch_size",  type=int, default=8)
    args = parser.parse_args()

    encode_features(args)
