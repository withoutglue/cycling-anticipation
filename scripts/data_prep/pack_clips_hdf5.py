"""
Pack per-clip npy files into a single HDF5 file for efficient training
on shared filesystems (e.g. CSC Roihu Lustre), which perform poorly
with thousands of small random file reads.

Output HDF5 schema:
    /clips        (N, 32, 256, 256, 3)  uint8
    /labels       (N,)                  int32  (-1 = unknown/excluded)
    /maneuvers    (N,)                  str
    /session_ids  (N,)                  str

Usage:
    # Labeled probe clips
    python pack_clips_hdf5.py --manifest manifest.csv --output clips.h5

    # Multiple manifests (e.g. across sessions)
    python pack_clips_hdf5.py --manifest s1/manifest.csv s2/manifest.csv --output clips.h5

    # Sliding window clips (no label column needed)
    python pack_clips_hdf5.py --manifest sliding_manifest.csv --output sliding.h5 --unlabeled
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

LABEL_MAP = {"left": 0, "right": 1, "straight": 2, "stop": 3}
CHUNK_SHAPE = (1, 32, 256, 256, 3)   # one clip per chunk — random access friendly


def pack(manifests: list[Path], output: Path, unlabeled: bool):
    if len(manifests) == 1:
        df = pd.read_csv(manifests[0])
    else:
        df = pd.concat([pd.read_csv(p) for p in manifests], ignore_index=True)

    if not unlabeled:
        df = df[df["maneuver"].isin(LABEL_MAP)].reset_index(drop=True)

    N = len(df)
    print(f"Packing {N} clips → {output}")

    with h5py.File(output, "w") as f:
        clips_ds = f.create_dataset(
            "clips",
            shape=(N, 32, 256, 256, 3),
            dtype=np.uint8,
            chunks=CHUNK_SHAPE,
        )

        str_dt = h5py.special_dtype(vlen=str)
        maneuvers_ds  = f.create_dataset("maneuvers",   shape=(N,), dtype=str_dt)
        session_ds    = f.create_dataset("session_ids", shape=(N,), dtype=str_dt)

        if not unlabeled:
            labels_ds = f.create_dataset("labels", shape=(N,), dtype=np.int32)

        for i, row in df.iterrows():
            if i % 500 == 0:
                print(f"  {i}/{N}")

            clip = np.load(row["npy_path"])   # (32, 256, 256, 3) uint8
            clips_ds[i]    = clip
            session_ds[i]  = str(row["session_id"])
            maneuvers_ds[i] = str(row.get("maneuver", ""))

            if not unlabeled:
                labels_ds[i] = LABEL_MAP.get(row["maneuver"], -1)

    print(f"Done. {output} ({output.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",  nargs="+", required=True,
                        help="One or more manifest.csv files")
    parser.add_argument("--output",    required=True,
                        help="Output .h5 file path")
    parser.add_argument("--unlabeled", action="store_true",
                        help="Set for sliding window clips with no maneuver labels")
    args = parser.parse_args()

    pack([Path(p) for p in args.manifest], Path(args.output), args.unlabeled)
