"""
PyTorch Dataset for cyclist maneuver anticipation probe training.

Supports two backends:
  - HDF5 (recommended for CSC/Lustre): pass hdf5_path to ManeuverDataset
  - npy files (local testing): pass manifest_path to ManeuverDataset

Labels:
    0 = left
    1 = right
    2 = straight
    3 = stop

Unknown events are excluded. Horizontal flip augmentation mirrors
left↔right clips, doubling both classes effectively.

Usage:
    # HDF5 (CSC training)
    ds = ManeuverDataset(hdf5_path="clips.h5", augment=True)

    # npy files (local testing)
    ds = ManeuverDataset(manifest_path="manifest.csv", augment=True)

    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_MAP = {"left": 0, "right": 1, "straight": 2, "stop": 3}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ManeuverDataset(Dataset):
    """
    Args:
        hdf5_path:     path to clips.h5 produced by pack_clips_hdf5.py
        manifest_path: path to manifest.csv (or list) — used when no hdf5_path
        augment:       randomly apply horizontal flip (left↔right mirrored)
        sessions:      optional list of session_ids to include (train/val split)
    """

    def __init__(
        self,
        hdf5_path=None,
        manifest_path=None,
        augment: bool = False,
        sessions: list[str] | None = None,
    ):
        if hdf5_path is None and manifest_path is None:
            raise ValueError("Provide either hdf5_path or manifest_path.")

        self.augment    = augment
        self.hdf5_path  = str(hdf5_path) if hdf5_path else None
        self._h5        = None   # opened lazily per DataLoader worker

        if hdf5_path:
            # Read metadata only — clips stay on disk until __getitem__
            with h5py.File(hdf5_path, "r") as f:
                maneuvers   = [m.decode() if isinstance(m, bytes) else m
                               for m in f["maneuvers"][:]]
                session_ids = [s.decode() if isinstance(s, bytes) else s
                               for s in f["session_ids"][:]]
                labels      = f["labels"][:].tolist()

            self.df = pd.DataFrame({
                "maneuver":   maneuvers,
                "session_id": session_ids,
                "label":      labels,
                "h5_index":   list(range(len(maneuvers))),
            })
            # Filter to known labels (label == -1 means unknown)
            self.df = self.df[self.df["label"] >= 0].reset_index(drop=True)

        else:
            if isinstance(manifest_path, (list, tuple)):
                df = pd.concat([pd.read_csv(p) for p in manifest_path],
                               ignore_index=True)
            else:
                df = pd.read_csv(manifest_path)
            df = df[df["maneuver"].isin(LABEL_MAP)].reset_index(drop=True)
            df["label"] = df["maneuver"].map(LABEL_MAP)
            self.df = df

        if sessions is not None:
            self.df = self.df[self.df["session_id"].isin(sessions)].reset_index(drop=True)

    def _open_h5(self):
        # Opened lazily so each DataLoader worker gets its own file handle.
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        maneuver = row["maneuver"]
        label    = int(row["label"])

        if self.hdf5_path:
            self._open_h5()
            clip = self._h5["clips"][int(row["h5_index"])]   # (32, 256, 256, 3) uint8
        else:
            clip = np.load(row["npy_path"])

        clip = torch.from_numpy(clip.copy()).permute(0, 3, 1, 2).float() / 255.0

        if self.augment and torch.rand(1).item() < 0.5:
            clip = clip.flip(-1)
            if maneuver == "left":
                label = LABEL_MAP["right"]
            elif maneuver == "right":
                label = LABEL_MAP["left"]

        clip = (clip - MEAN) / STD
        return clip, label

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for weighted cross-entropy loss."""
        counts = self.df["maneuver"].value_counts()
        weights = torch.zeros(len(LABEL_MAP))
        for name, i in LABEL_MAP.items():
            n = counts.get(name, 0)
            weights[i] = 1.0 / n if n > 0 else 0.0
        weights = weights / weights.sum()
        return weights

    def summary(self):
        print(f"  Samples: {len(self.df)}")
        counts = self.df["maneuver"].value_counts()
        for name in LABEL_MAP:
            print(f"    {name:10s}: {counts.get(name, 0)}")


class FeaturesDataset(Dataset):
    """
    Reads pre-encoded features from features.h5 (produced by encode_features.py).
    No encoder needed at training time — each item is already a token feature tensor.

    Args:
        hdf5_path: path to features.h5
        sessions:  optional list of session_ids for train/val split
    """

    def __init__(self, hdf5_path, sessions: list[str] | None = None,
                 include_flipped: bool = True):
        self.hdf5_path = str(hdf5_path)
        self._h5 = None

        with h5py.File(hdf5_path, "r") as f:
            maneuvers   = [m.decode() if isinstance(m, bytes) else m
                           for m in f["maneuvers"][:]]
            session_ids = [s.decode() if isinstance(s, bytes) else s
                           for s in f["session_ids"][:]]
            labels      = f["labels"][:].tolist()
            is_flipped  = (f["is_flipped"][:].tolist()
                           if "is_flipped" in f else [False] * len(labels))

        self.df = pd.DataFrame({
            "maneuver":   maneuvers,
            "session_id": session_ids,
            "label":      labels,
            "is_flipped": is_flipped,
            "h5_index":   list(range(len(maneuvers))),
        })
        self.df = self.df[self.df["label"] >= 0].reset_index(drop=True)

        if not include_flipped:
            self.df = self.df[~self.df["is_flipped"].astype(bool)].reset_index(drop=True)

        if sessions is not None:
            self.df = self.df[self.df["session_id"].isin(sessions)].reset_index(drop=True)

    def _open_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self._open_h5()
        row = self.df.iloc[idx]
        features = torch.from_numpy(
            self._h5["features"][int(row["h5_index"])].astype(np.float32)
        )  # (num_tokens, 1408)
        return features, int(row["label"])

    def class_weights(self) -> torch.Tensor:
        counts = self.df["maneuver"].value_counts()
        weights = torch.zeros(len(LABEL_MAP))
        for name, i in LABEL_MAP.items():
            n = counts.get(name, 0)
            weights[i] = 1.0 / n if n > 0 else 0.0
        weights = weights / weights.sum()
        return weights

    def summary(self):
        print(f"  Samples: {len(self.df)}")
        counts = self.df["maneuver"].value_counts()
        for name in LABEL_MAP:
            print(f"    {name:10s}: {counts.get(name, 0)}")
