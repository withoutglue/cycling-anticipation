"""
Extract GPS data from Gyroflow JSON export and convert to frame-indexed CSV.

Gyroflow embeds GPS data (from Insta360 GPS Remote) in its JSON export at 1 Hz.
This script converts those fixes to frame-indexed rows using the recording start
time extracted from the .insv file via exiftool.

Output columns:
  frame          — 0-based video frame index (nearest frame to the GPS fix)
  unix_timestamp — GPS fix time (seconds since Unix epoch, UTC)
  timestamp_ms   — ms from recording start
  lat, lon       — WGS-84 coordinates
  speed_kmh      — speed in km/h (as reported by GPS remote)
  speed_ms       — speed in m/s (converted from km/h)
  heading        — direction of travel in degrees (0=N, 90=E, 180=S, 270=W)
  altitude       — metres above sea level
  is_acquired    — True if GPS had a valid fix at this point

Usage — single file:
    python extract_gps.py --json session.json --insv session.insv --output session_gps.csv

Usage — batch (matching stems: session.json + session.insv → session_gps.csv):
    python extract_gps.py --json_dir /path/to/json --insv_dir /path/to/insv --output_dir /path/to/gps

Usage — if you already know the recording start time (UTC):
    python extract_gps.py --json session.json --start_time "2023:10:16 13:16:10" --output session_gps.csv
"""

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import ijson
import pandas as pd

FPS = 30.0
GPS_COLD_START_S = 13.0  # typical GPS Remote cold start delay (s)


def get_recording_start(insv_path: Path) -> float:
    """
    Return the recording start time as a Unix timestamp (UTC seconds)
    by reading Create Date from .insv metadata via exiftool.
    """
    try:
        result = subprocess.run(
            ["exiftool", "-CreateDate", "-s3", str(insv_path)],
            capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        raise RuntimeError("exiftool not found — install it with: brew install exiftool")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"exiftool failed on {insv_path.name}: {e.stderr.strip()}")

    raw = result.stdout.strip()
    if not raw:
        raise ValueError(f"No CreateDate found in {insv_path.name}")

    dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc).timestamp()


def parse_start_time(s: str) -> float:
    """Parse a manual start time string (UTC) to Unix timestamp."""
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse start_time '{s}' — use 'YYYY:MM:DD HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS'")


def load_gps(json_path: Path) -> list[dict]:
    """
    Stream-parse Gyroflow JSON and return deduplicated GPS points.

    Uses ijson to avoid loading the full file (600MB+) into memory — the bulk
    of the JSON is 1000Hz IMU data we don't need here.
    Gyroflow emits ~10 identical rows per unix_timestamp; deduplicate on that.
    """
    seen: set[float] = set()
    points: list[dict] = []

    with open(json_path, "rb") as f:
        for pt in ijson.items(f, "GPS.Data.item"):
            ts = pt["unix_timestamp"]
            if ts in seen:
                continue
            seen.add(ts)
            points.append(pt)

    if not points:
        raise ValueError(
            f"No GPS data found in {json_path.name} — "
            "was the GPS Remote used for this session?"
        )

    return points


def build_dataframe(points: list[dict], recording_start: float, fps: float = FPS) -> pd.DataFrame:
    rows = []
    for pt in points:
        ts = pt["unix_timestamp"]
        relative_s = float(ts) - recording_start
        frame = round(relative_s * fps)

        rows.append({
            "frame":         frame,
            "unix_timestamp": ts,
            "timestamp_ms":  round(relative_s * 1000, 1),
            "lat":           pt["lat"],
            "lon":           pt["lon"],
            "speed_kmh":     round(pt.get("speed", float("nan")), 3),
            "speed_ms":      round(float(pt.get("speed", float("nan"))) / 3.6, 4),
            "heading":       round(pt.get("track", float("nan")), 3),
            "altitude":      round(pt.get("altitude", float("nan")), 2),
            "is_acquired":   pt.get("is_acquired", False),
        })

    df = pd.DataFrame(rows)

    # Drop points that fall before the recording started
    df = df[df["frame"] >= 0].reset_index(drop=True)
    return df


def derive_recording_start(points: list[dict], cold_start_s: float = GPS_COLD_START_S) -> float:
    """
    Derive recording_start from the first acquired GPS fix minus the cold start delay.
    Use when the camera clock is wrong and exiftool / --start_time is unreliable.
    cold_start_s is the time between pressing Record and the GPS Remote getting its first fix.
    Empirically ~12.8 s for session 14; 10–20 s is typical for a warm GPS Remote.
    """
    acquired = [p for p in points if p.get("is_acquired", False)]
    if not acquired:
        raise ValueError("No acquired GPS fixes — cannot derive recording_start.")
    first_ts = float(acquired[0]["unix_timestamp"])
    return first_ts - cold_start_s


def process_file(json_path: Path, output_path: Path,
                 insv_path: Path | None = None,
                 start_time: str | None = None,
                 cold_start_s: float | None = None,
                 fps: float = FPS) -> int:
    points = load_gps(json_path)

    if cold_start_s is not None:
        recording_start = derive_recording_start(points, cold_start_s)
        print(f"  recording_start (GPS-derived, cold_start={cold_start_s}s): "
              f"{recording_start:.3f}  ({datetime.fromtimestamp(recording_start, tz=timezone.utc).isoformat()})")
    elif start_time is not None:
        recording_start = parse_start_time(start_time)
    elif insv_path is not None:
        recording_start = get_recording_start(insv_path)
    else:
        raise ValueError("Provide --insv, --start_time, or --cold_start_s")

    # Sanity check: first acquired GPS fix should be within 0–120 s of recording_start
    acquired = [p for p in points if p.get("is_acquired", False)]
    if acquired:
        offset_s = float(acquired[0]["unix_timestamp"]) - recording_start
        if offset_s < -60 or offset_s > 300:
            raise ValueError(
                f"recording_start looks wrong: first GPS fix is {offset_s:+.0f}s from it. "
                f"recording_start={recording_start:.0f}, first_gps_ts={acquired[0]['unix_timestamp']:.0f}. "
                "If the camera clock is wrong, use --cold_start_s instead of --insv/--start_time."
            )

    df = build_dataframe(points, recording_start, fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return len(df)


def process_batch(json_dir: Path, insv_dir: Path, output_dir: Path, fps: float = FPS):
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    print(f"Found {len(json_files)} JSON files\n")
    for json_path in json_files:
        insv_path = insv_dir / (json_path.stem + ".insv")
        out_path = output_dir / (json_path.stem + "_gps.csv")

        if not insv_path.exists():
            print(f"  {json_path.name:40s}  SKIP: {insv_path.name} not found")
            continue

        try:
            n = process_file(json_path, out_path, insv_path=insv_path, fps=fps)
            print(f"  {json_path.name:40s}  →  {n:4d} fixes  →  {out_path.name}")
        except Exception as e:
            print(f"  {json_path.name:40s}  ERROR: {e}")

    print(f"\nDone. Output in: {output_dir}")


def print_summary(gps_csv_path: Path):
    df = pd.read_csv(gps_csv_path)
    acquired = df[df["is_acquired"]]
    duration_s = (df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]) / 1000

    print(f"\n{gps_csv_path.name}")
    print(f"  Fixes       : {len(df)} total, {len(acquired)} acquired")
    print(f"  Duration    : {duration_s:.0f} s")
    print(f"  Speed (kmh) : mean {acquired['speed_kmh'].mean():.1f}  "
          f"max {acquired['speed_kmh'].max():.1f}")
    print(f"  Lat range   : {acquired['lat'].min():.6f} – {acquired['lat'].max():.6f}")
    print(f"  Lon range   : {acquired['lon'].min():.6f} – {acquired['lon'].max():.6f}")
    print(f"  Frame range : {df['frame'].min()} – {df['frame'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract GPS from Gyroflow JSON to frame-indexed CSV."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--json",     type=Path, help="Single Gyroflow JSON file")
    mode.add_argument("--json_dir", type=Path, help="Directory of Gyroflow JSON files (batch)")

    anchor = parser.add_mutually_exclusive_group()
    anchor.add_argument("--insv",         type=Path,  help="Matching .insv file (exiftool reads start time)")
    anchor.add_argument("--start_time",   type=str,   help="Recording start in UTC: 'YYYY:MM:DD HH:MM:SS'")
    anchor.add_argument("--cold_start_s", type=float,
                        help=f"Derive recording_start from first GPS fix minus this delay (s). "
                             f"Use when camera clock is wrong. Empirical default: {GPS_COLD_START_S}")

    parser.add_argument("--insv_dir",   type=Path, help="Directory of .insv files (batch mode)")
    parser.add_argument("--output",     type=Path, help="Output CSV path (single mode)")
    parser.add_argument("--output_dir", type=Path, help="Output directory (batch mode)")
    parser.add_argument("--fps",        type=float, default=FPS)
    parser.add_argument("--summary",    action="store_true")

    args = parser.parse_args()

    if args.json:
        if not args.output:
            parser.error("--output is required with --json")
        if not args.insv and not args.start_time and args.cold_start_s is None:
            parser.error("Provide --insv, --start_time, or --cold_start_s")

        n = process_file(args.json, args.output,
                         insv_path=args.insv,
                         start_time=args.start_time,
                         cold_start_s=args.cold_start_s,
                         fps=args.fps)
        print(f"Wrote {n} GPS fixes to {args.output}")
        if args.summary:
            print_summary(args.output)
    else:
        if not args.insv_dir or not args.output_dir:
            parser.error("--insv_dir and --output_dir are required with --json_dir")
        process_batch(args.json_dir, args.insv_dir, args.output_dir, args.fps)
