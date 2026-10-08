"""
Convert Gyroflow IMU CSV export to per-frame action vectors.

Gyroflow exports one CSV per .insv file at 1000 Hz. This script aggregates
those readings into one row per video frame (30 fps = 33.3 ms per frame).

Output columns per frame:
  frame           — 0-based video frame index
  timestamp_ms    — start of frame window (ms from recording start)
  delta_gyro_x/y/z — gyroscope integrated over the frame window (degrees)
  acc_x/y/z       — mean accelerometer over the frame window (m/s²)
  quat_w/x/y/z    — mean quaternion over the frame window (absolute orientation)

The 7D action vector for the model is: (delta_gyro_x, delta_gyro_y, delta_gyro_z,
acc_x, acc_y, acc_z, speed_kmh). Quaternions are stored for reference and debugging.
speed_kmh is sourced from the GPS CSV (1 Hz) and forward-filled to match frame rate.

NOTE: gyro axis mapping — which of gyro_x/y/z corresponds to heading (yaw)
depends on how the camera is mounted. With a helmet-mounted Insta360 X3 facing
forward, gyro_z is typically yaw. Verify by checking delta_gyro_z during a
known left/right turn in the data.

Usage — single file:
    python extract_imu.py --input session.csv --output session_imu.csv

Usage — batch (all CSVs in a directory):
    python extract_imu.py --csv_dir /path/to/gyroflow_exports --output_dir /path/to/imu
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 30.0
FRAME_MS = 1000.0 / FPS  # 33.333 ms per frame
GYRO_DT = 0.001           # Gyroflow timestamps are 1 ms apart → dt = 0.001 s


def process_csv(input_path: Path, output_path: Path,
                gps_path: Path | None = None, fps: float = FPS) -> int:
    """
    Convert one Gyroflow CSV to a per-frame action vector CSV.
    Returns the number of frames written.
    """
    frame_ms = 1000.0 / fps

    df = pd.read_csv(input_path)

    required = {"timestamp_ms", "org_gyro_x", "org_gyro_y", "org_gyro_z",
                "org_acc_x", "org_acc_y", "org_acc_z",
                "org_quat_w", "org_quat_x", "org_quat_y", "org_quat_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {input_path.name}: {missing}")

    # Positive timestamps = actual recording; negative = camera pre-roll
    df = df[df["timestamp_ms"] >= 0.0].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No positive timestamps found in {input_path.name}")

    t0 = df["timestamp_ms"].iloc[0]
    t_last = df["timestamp_ms"].iloc[-1]
    n_frames = int((t_last - t0) / frame_ms)

    records = []

    for i in range(n_frames):
        t_start = t0 + i * frame_ms
        t_end = t0 + (i + 1) * frame_ms

        window = df[(df["timestamp_ms"] >= t_start) & (df["timestamp_ms"] < t_end)]

        if window.empty:
            continue

        # Integrate gyroscope over window: Σ(deg/s × 0.001 s) = total degrees rotated
        delta_gyro_x = (window["org_gyro_x"] * GYRO_DT).sum()
        delta_gyro_y = (window["org_gyro_y"] * GYRO_DT).sum()
        delta_gyro_z = (window["org_gyro_z"] * GYRO_DT).sum()

        records.append({
            "frame":         i,
            "timestamp_ms":  round(t_start, 3),
            "delta_gyro_x":  round(delta_gyro_x, 6),
            "delta_gyro_y":  round(delta_gyro_y, 6),
            "delta_gyro_z":  round(delta_gyro_z, 6),
            "acc_x":         round(window["org_acc_x"].mean(), 6),
            "acc_y":         round(window["org_acc_y"].mean(), 6),
            "acc_z":         round(window["org_acc_z"].mean(), 6),
            "quat_w":        round(window["org_quat_w"].mean(), 6),
            "quat_x":        round(window["org_quat_x"].mean(), 6),
            "quat_y":        round(window["org_quat_y"].mean(), 6),
            "quat_z":        round(window["org_quat_z"].mean(), 6),
        })

    out_df = pd.DataFrame(records)

    if gps_path is not None:
        gps = pd.read_csv(gps_path)[["frame", "speed_kmh"]]
        out_df = out_df.merge(gps, on="frame", how="left")
        out_df["speed_kmh"] = out_df["speed_kmh"].ffill().bfill().round(3)
    else:
        out_df["speed_kmh"] = float("nan")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    return len(out_df)


def process_all(csv_dir: Path, output_dir: Path, fps: float = FPS):
    """Process all Gyroflow CSVs in a directory."""
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return

    print(f"Found {len(csv_files)} Gyroflow CSV files\n")

    for csv_path in csv_files:
        out_path = output_dir / (csv_path.stem + "_imu.csv")
        try:
            n = process_csv(csv_path, out_path, fps=fps)
            print(f"  {csv_path.name:40s}  →  {n:5d} frames  →  {out_path.name}")
        except Exception as e:
            print(f"  {csv_path.name:40s}  ERROR: {e}")

    print(f"\nDone. Output in: {output_dir}")


def print_summary(imu_csv_path: Path):
    """Print basic statistics for a processed IMU CSV — useful for sanity checks."""
    df = pd.read_csv(imu_csv_path)
    duration_s = len(df) / FPS
    print(f"\n{imu_csv_path.name}")
    print(f"  Frames: {len(df)}  ({duration_s:.1f} s at {FPS} fps)")
    print(f"  delta_gyro_z  — mean: {df['delta_gyro_z'].mean():+.3f}°  "
          f"std: {df['delta_gyro_z'].std():.3f}°  "
          f"max: {df['delta_gyro_z'].abs().max():.3f}°")
    print(f"  acc_y         — mean: {df['acc_y'].mean():+.3f}  "
          f"(should be ~9.8 if Y is gravity axis)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Gyroflow IMU CSV exports to per-frame action vectors."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",   type=Path,
                       help="Single Gyroflow CSV file")
    group.add_argument("--csv_dir", type=Path,
                       help="Directory of Gyroflow CSV files (batch mode)")

    parser.add_argument("--output",     type=Path,
                        help="Output CSV path (required with --input)")
    parser.add_argument("--output_dir", type=Path,
                        help="Output directory (required with --csv_dir)")
    parser.add_argument("--fps", type=float, default=FPS,
                        help=f"Video frame rate (default: {FPS})")
    parser.add_argument("--gps_csv", type=Path, default=None,
                        help="GPS CSV from extract_gps.py — adds speed_kmh as 7th action dimension")
    parser.add_argument("--summary", action="store_true",
                        help="Print statistics after processing (single file mode)")

    args = parser.parse_args()

    if args.input:
        if not args.output:
            parser.error("--output is required with --input")
        n = process_csv(args.input, args.output, gps_path=args.gps_csv, fps=args.fps)
        print(f"Wrote {n} frames to {args.output}")
        if args.summary:
            print_summary(args.output)
    else:
        if not args.output_dir:
            parser.error("--output_dir is required with --csv_dir")
        process_all(args.csv_dir, args.output_dir, args.fps)
