"""
Count maneuver class distribution across one or more events CSV files.

Usage:
    python count_events.py --events session_events.csv [session2_events.csv ...]
    python count_events.py --events_dir data/events/
"""

import argparse
from pathlib import Path

import pandas as pd

LABELS = ("left", "right", "straight", "stop", "unknown")


def main():
    parser = argparse.ArgumentParser(description="Count maneuver classes across events CSVs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--events",     type=Path, nargs="+", help="One or more events CSV files")
    group.add_argument("--events_dir", type=Path,             help="Directory containing events CSVs")
    args = parser.parse_args()

    if args.events_dir:
        files = sorted(args.events_dir.glob("*events*.csv"))
        if not files:
            raise SystemExit(f"No events CSVs found in {args.events_dir}")
    else:
        files = args.events

    all_rows = []
    for f in files:
        df = pd.read_csv(f)
        df["_source"] = f.name
        all_rows.append(df)
        counts = df["maneuver"].value_counts()
        print(f"{f.name}  ({len(df)} events)")
        for label in LABELS:
            n = counts.get(label, 0)
            if n:
                print(f"  {label:10s}: {n}")

    if len(files) > 1:
        combined = pd.concat(all_rows, ignore_index=True)
        counts = combined["maneuver"].value_counts()
        print(f"\nTOTAL  ({len(combined)} events across {len(files)} files)")
        for label in LABELS:
            n = counts.get(label, 0)
            print(f"  {label:10s}: {n}")


if __name__ == "__main__":
    main()
