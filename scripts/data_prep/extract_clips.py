"""
Extract video clips around intersection crossing events.

For each event in the events CSV (from detect_intersections.py), cuts a 4s
approach clip ending at crossing_frame — the moment the cyclist enters the
intersection radius. The model sees only the approach and must predict the
maneuver that follows.

  clip_end   = crossing_frame               (entry into intersection)
  clip_start = crossing_frame - 4s × fps   (4 s of approach context)

  anticipation_gap = onset_frame - crossing_frame  (varies per event)

Output clips are resampled to TARGET_FRAMES evenly-spaced source frames,
resized to TARGET_SIZE×TARGET_SIZE, matching the V-JEPA 2 probe format
(8 fps, 32 frames, 256×256).

Each clip is saved as:
  npy/  — (32, 256, 256, 3) RGB uint8 numpy array    → used for training

A manifest.csv is written to output_dir listing all extracted clips with their
metadata, frame ranges, and anticipation gap. The Dataset class reads this manifest.

Usage:
    python extract_clips.py --video session.mp4 --events session_events.csv --output_dir clips/

    # Custom timing:
    python extract_clips.py --video session.mp4 --events session_events.csv \\
        --output_dir clips/ --clip_s 4.0
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FPS           = 30.0   # source video frame rate
CLIP_S        =  4.0   # clip duration in seconds (approach context)
TARGET_FRAMES = 32     # output frames per clip (4 s × 8 fps)
TARGET_SIZE   = 256    # output spatial resolution (square)


def _video_meta(path: Path) -> tuple[int, float, int, int]:
    """Return (total_frames, fps, height, width)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return total, fps, h, w


def _read_clip(cap: cv2.VideoCapture, frame_indices: np.ndarray,
               size: int) -> np.ndarray | None:
    """
    Seek once to the first frame, decode the window sequentially, then
    subsample. Sequential decoding is ~10-20x faster than seeking per frame
    on compressed video (H.264 must decode from keyframe on every seek).
    Returns (T, size, size, 3) RGB uint8 array, or None on read failure.
    """
    first = int(frame_indices[0])
    last  = int(frame_indices[-1])
    n_source = last - first + 1
    frame_indices_set = set(frame_indices.tolist())

    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    buffer = {}
    for i in range(n_source):
        ret, frame = cap.read()
        if not ret:
            break
        src_idx = first + i
        if src_idx in frame_indices_set:
            buffer[src_idx] = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (size, size)
            )

    frames = [buffer.get(idx) for idx in frame_indices]
    if any(f is None for f in frames):
        return None
    return np.stack(frames, axis=0)


def process(video_path: Path, events_df: pd.DataFrame, output_dir: Path,
            clip_s: float = CLIP_S, fps: float = FPS) -> pd.DataFrame:
    """
    Extract clips for all events. Returns a manifest DataFrame.
    """
    clip_len = round(clip_s * fps)

    total_frames, actual_fps, H, W = _video_meta(video_path)
    print(f"  Video      : {video_path.name}")
    print(f"  Resolution : {W}×{H}  {actual_fps:.1f} fps  {total_frames} frames "
          f"({total_frames / actual_fps:.0f} s)")
    print(f"  Clip window: {clip_s:.0f} s approach ending at crossing_frame  "
          f"→ {TARGET_FRAMES} frames at {TARGET_SIZE}×{TARGET_SIZE}")

    npy_dir = output_dir / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    manifest_rows = []
    n_skip = 0

    # Sort by crossing_frame so seeks move forward through the video —
    # the decoder doesn't have to rewind, cutting seek overhead significantly.
    events_df = events_df.sort_values("crossing_frame").reset_index(drop=True)

    for _, event in events_df.iterrows():
        crossing_frame = int(event["crossing_frame"])
        maneuver       = str(event["maneuver"])
        session_id     = str(event["session_id"])

        onset_val   = event.get("onset_frame")
        onset_frame = int(onset_val) if pd.notna(onset_val) else crossing_frame

        clip_end   = crossing_frame
        clip_start = clip_end - clip_len

        if clip_start < 0:
            print(f"    SKIP crossing {crossing_frame:7d} ({maneuver}): clip_start {clip_start} < 0")
            n_skip += 1
            continue

        if clip_end > total_frames:
            print(f"    SKIP crossing {crossing_frame:7d} ({maneuver}): clip_end {clip_end} > {total_frames}")
            n_skip += 1
            continue

        # Evenly sample TARGET_FRAMES source frames over [clip_start, clip_end)
        frame_indices = np.linspace(clip_start, clip_end - 1, TARGET_FRAMES).round().astype(int)

        stem     = f"{session_id}_{crossing_frame:07d}_{maneuver}"
        npy_path = npy_dir / f"{stem}.npy"

        clip = _read_clip(cap, frame_indices, TARGET_SIZE)
        if clip is None:
            print(f"    SKIP crossing {crossing_frame:7d} ({maneuver}): incomplete read")
            n_skip += 1
            continue

        np.save(npy_path, clip)

        anticipation_gap_s = round((onset_frame - crossing_frame) / fps, 3)

        manifest_rows.append({
            "clip_id":              stem,
            "session_id":           session_id,
            "maneuver":             maneuver,
            "crossing_frame":       crossing_frame,
            "onset_frame":          onset_frame,
            "clip_start_frame":     clip_start,
            "clip_end_frame":       clip_end,
            "anticipation_gap_s":   anticipation_gap_s,
            "clip_s":               clip_s,
            "target_frames":        TARGET_FRAMES,
            "target_size":          TARGET_SIZE,
            "npy_path":             str(npy_path.resolve()),
        })

    cap.release()

    manifest = pd.DataFrame(manifest_rows)
    print(f"\n  Extracted : {len(manifest)}")
    if n_skip:
        print(f"  Skipped   : {n_skip}")
    if not manifest.empty:
        counts = manifest["maneuver"].value_counts().to_dict()
        for label in ("left", "right", "straight", "stop", "unknown"):
            n = counts.get(label, 0)
            if n:
                print(f"    {label:10s}: {n}")
        print(f"\n  Anticipation gap — mean: {manifest['anticipation_gap_s'].mean():.1f} s  "
              f"min: {manifest['anticipation_gap_s'].min():.1f} s  "
              f"max: {manifest['anticipation_gap_s'].max():.1f} s")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract 4s approach clips ending at crossing_frame."
    )
    parser.add_argument("--video",      type=Path, required=True,
                        help="Front-facing .mp4 for this session")
    parser.add_argument("--events",     type=Path, required=True,
                        help="Events CSV from detect_intersections.py")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Root output directory (npy/ created inside)")
    parser.add_argument("--clip_s",     type=float, default=CLIP_S,
                        help=f"Clip duration in seconds (default: {CLIP_S})")
    parser.add_argument("--fps",        type=float, default=FPS,
                        help=f"Source video frame rate (default: {FPS})")

    args = parser.parse_args()

    events_df = pd.read_csv(args.events)
    if events_df.empty:
        raise SystemExit("Events CSV is empty — nothing to extract.")

    print(f"\nEvents: {len(events_df)} crossings from {args.events.name}")

    manifest = process(
        args.video, events_df, args.output_dir,
        clip_s=args.clip_s,
        fps=args.fps,
    )

    if not manifest.empty:
        # Per-session manifest — safe to overwrite on re-run
        manifests_dir = args.output_dir / "manifests"
        manifests_dir.mkdir(exist_ok=True)
        session_id = manifest["session_id"].iloc[0]
        session_manifest = manifests_dir / f"{session_id}.csv"
        manifest.to_csv(session_manifest, index=False)
        print(f"\nSession manifest: {session_manifest}")

        # Combined manifest — append, skip header if file already exists
        combined = args.output_dir / "manifest.csv"
        write_header = not combined.exists()
        manifest.to_csv(combined, mode="a", header=write_header, index=False)
        print(f"Combined manifest {'created' if write_header else 'appended'}: {combined}")
