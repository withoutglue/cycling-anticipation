"""
Remove an event row from an intersection events CSV.

Usage:
    python remove_event.py --events session_events.csv --frame 14782 --label stop
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Remove an event row from an events CSV.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--frame",  type=int,  required=True, help="crossing_frame of the event to remove")
    parser.add_argument("--label",  type=str,  required=True, help="maneuver label of the event to remove")
    args = parser.parse_args()

    df = pd.read_csv(args.events)

    mask = (df["crossing_frame"] == args.frame) & (df["maneuver"] == args.label)
    matches = df[mask]

    if matches.empty:
        raise SystemExit(f"No event with crossing_frame={args.frame} and maneuver='{args.label}'")

    print(f"Removing {len(matches)} row(s):")
    print(matches[["crossing_frame", "onset_frame", "maneuver", "session_id"]].to_string(index=False))

    df = df[~mask].reset_index(drop=True)
    df.to_csv(args.events, index=False)
    print(f"Saved: {args.events}  ({len(df)} events remaining)")


if __name__ == "__main__":
    main()
