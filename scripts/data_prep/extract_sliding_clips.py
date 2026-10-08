"""
Extract sliding-window clips from raw cycling video for self-supervised
IMU-conditioned predictor training (V-JEPA 2-AC analogue).

Slides a 4s window across the full video with a configurable stride.
Clips where the mean GPS speed is below MIN_SPEED_KMH are skipped —
near-static footage (traffic lights, stops) adds little motion signal.

No maneuver labels are used. The predictor trains self-supervised:
predict the next frame representation conditioned on past frames + IMU.

Output:
  npy/  — (32, 256, 256, 3) RGB uint8 numpy array per clip

A manifest.csv is written listing clip metadata and file paths.

Usage:
    python extract_sliding_clips.py --video session.mp4 --gps session_gps.csv \\
        --output_dir predictor_clips/

    # Custom stride and speed filter:
    python extract_sliding_clips.py --video session.mp4 --gps session_gps.csv \\
        --output_dir predictor_clips/ --stride_s 2.0 --min_speed_kmh 5.0
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FPS           = 30.0   # source video frame rate
CLIP_S        =  4.0   # clip duration in seconds
STRIDE_S      =  2.0   # stride between clip start frames in seconds
TARGET_FRAMES = 32     # output frames per clip (4 s × 8 fps)
TARGET_SIZE   = 256    # output spatial resolution (square)
MIN_SPEED_KMH =  5.0   # skip clips with mean GPS speed below this


def _video_meta(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total, fps


def _read_clip(cap: cv2.VideoCapture, frame_indices: np.ndarray,
               size: int) -> np.ndarray | None:
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.resize(frame, (size, size))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def _build_speed_lookup(gps_df: pd.DataFrame, total_frames: int) -> np.ndarray:
    """Return a per-source-frame speed array (forward-filled from 1 Hz GPS)."""
    speeds = np.full(total_frames, np.nan)
    acquired = gps_df[gps_df["is_acquired"] & gps_df["frame"].between(0, total_frames - 1)]
    for _, row in acquired.iterrows():
        speeds[int(row["frame"])] = row["speed_kmh"]
    # forward-fill then backward-fill
    df = pd.Series(speeds)
    speeds = df.ffill().bfill().to_numpy()
    return speeds


def process(video_path: Path, gps_path: Path, output_dir: Path,
            clip_s: float = CLIP_S, stride_s: float = STRIDE_S,
            fps: float = FPS, min_speed_kmh: float = MIN_SPEED_KMH) -> pd.DataFrame:

    clip_len   = round(clip_s * fps)
    stride_len = round(stride_s * fps)
    session_id = video_path.stem

    total_frames, actual_fps = _video_meta(video_path)
    gps_df = pd.read_csv(gps_path)
    speeds = _build_speed_lookup(gps_df, total_frames)

    print(f"  Video    : {video_path.name}")
    print(f"  Frames   : {total_frames}  ({total_frames / actual_fps:.0f} s at {actual_fps:.1f} fps)")
    print(f"  Stride   : {stride_s:.1f} s  |  Min speed: {min_speed_kmh} km/h")

    npy_dir = output_dir / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    manifest_rows = []
    n_skip_bounds = 0
    n_skip_speed  = 0

    clip_start = 0
    while clip_start + clip_len <= total_frames:
        clip_end = clip_start + clip_len

        mean_speed = float(np.nanmean(speeds[clip_start:clip_end]))
        if mean_speed < min_speed_kmh:
            n_skip_speed += 1
            clip_start += stride_len
            continue

        frame_indices = np.linspace(clip_start, clip_end - 1, TARGET_FRAMES).round().astype(int)

        stem     = f"{session_id}_{clip_start:07d}"
        npy_path = npy_dir / f"{stem}.npy"

        clip = _read_clip(cap, frame_indices, TARGET_SIZE)
        if clip is None:
            n_skip_bounds += 1
            clip_start += stride_len
            continue

        np.save(npy_path, clip)

        manifest_rows.append({
            "clip_id":        stem,
            "session_id":     session_id,
            "clip_start_frame": clip_start,
            "clip_end_frame":   clip_end,
            "mean_speed_kmh": round(mean_speed, 2),
            "clip_s":         clip_s,
            "stride_s":       stride_s,
            "target_frames":  TARGET_FRAMES,
            "target_size":    TARGET_SIZE,
            "npy_path":       str(npy_path.resolve()),
        })

        clip_start += stride_len

    cap.release()

    manifest = pd.DataFrame(manifest_rows)
    print(f"\n  Extracted : {len(manifest)}")
    print(f"  Skipped (low speed) : {n_skip_speed}")
    if n_skip_bounds:
        print(f"  Skipped (read error): {n_skip_bounds}")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract sliding-window clips for predictor self-supervised training."
    )
    parser.add_argument("--video",         type=Path, required=True)
    parser.add_argument("--gps",           type=Path, required=True,
                        help="GPS CSV from extract_gps.py")
    parser.add_argument("--output_dir",    type=Path, required=True)
    parser.add_argument("--clip_s",        type=float, default=CLIP_S)
    parser.add_argument("--stride_s",      type=float, default=STRIDE_S)
    parser.add_argument("--fps",           type=float, default=FPS)
    parser.add_argument("--min_speed_kmh", type=float, default=MIN_SPEED_KMH)

    args = parser.parse_args()

    manifest = process(
        args.video, args.gps, args.output_dir,
        clip_s=args.clip_s,
        stride_s=args.stride_s,
        fps=args.fps,
        min_speed_kmh=args.min_speed_kmh,
    )

    if not manifest.empty:
        manifest_path = args.output_dir / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"\nManifest saved to: {manifest_path}")
