"""
Manually patch one event row in an intersection events CSV.

Identify the row by --frame (crossing_frame). When multiple rows share the same
crossing_frame, --label narrows by current maneuver value and deletes duplicates.
Use --new_label to change the label in that case (--label is then the identifier only).

Usage:
    python relabel.py --events session_events.csv --frame 22380 --label right
    python relabel.py --events session_events.csv --frame 22380 --crossing_frame 22200
    python relabel.py --events session_events.csv --frame 22380 --onset_frame 22450
    python relabel.py --events session_events.csv --frame 22380 --label left --onset_frame 22450
    # duplicate frames — identify by current label, change to a new label:
    python relabel.py --events session_events.csv --frame 26010 --label straight --new_label right
"""

import argparse
from pathlib import Path

import pandas as pd

VALID_LABELS = {"left", "right", "straight", "stop", "unknown"}


def main():
    parser = argparse.ArgumentParser(description="Patch a single event row in an events CSV.")
    parser.add_argument("--events",         type=Path, required=True,
                        help="Events CSV from detect_intersections.py")
    parser.add_argument("--frame",          type=int,  required=True,
                        help="crossing_frame value identifying the event to patch")
    parser.add_argument("--label",          type=str,  default=None,
                        help="Current maneuver label — used as identifier when frames are ambiguous, "
                             "and sets the new label unless --new_label is also given")
    parser.add_argument("--new_label",      type=str,  default=None,
                        help=f"New maneuver label (use when --label is needed to identify the row): {VALID_LABELS}")
    parser.add_argument("--crossing_frame", type=int,  default=None,
                        help="New crossing_frame value")
    parser.add_argument("--onset_frame",    type=int,  default=None,
                        help="New onset_frame value")
    args = parser.parse_args()

    if args.label is None and args.new_label is None and args.crossing_frame is None and args.onset_frame is None:
        parser.error("Provide at least one of --label, --new_label, --crossing_frame, --onset_frame")

    for val in [args.label, args.new_label]:
        if val and val not in VALID_LABELS:
            parser.error(f"Invalid label '{val}'. Must be one of: {VALID_LABELS}")

    df = pd.read_csv(args.events)

    matches = df.index[df["crossing_frame"] == args.frame].tolist()
    if not matches:
        raise SystemExit(f"No event with crossing_frame={args.frame} in {args.events.name}")

    if len(matches) > 1:
        if args.label is None:
            raise SystemExit(
                f"Multiple events with crossing_frame={args.frame} — pass --label to identify which row to keep:\n"
                + "\n".join(f"  maneuver={df.loc[i, 'maneuver']}  osm_node_id={df.loc[i, 'osm_node_id']}"
                            for i in matches)
            )
        keep = [i for i in matches if df.loc[i, "maneuver"] == args.label]
        drop = [i for i in matches if i not in keep]
        if not keep:
            raise SystemExit(
                f"No event with crossing_frame={args.frame} and maneuver='{args.label}'"
            )
        if len(keep) > 1:
            raise SystemExit(
                f"Still ambiguous: multiple rows with crossing_frame={args.frame} and maneuver='{args.label}'"
            )
        print(f"Dropping {len(drop)} duplicate row(s) with crossing_frame={args.frame}")
        df = df.drop(index=drop).reset_index(drop=True)
        matches = df.index[df["crossing_frame"] == args.frame].tolist()

    idx = matches[0]
    row = df.loc[idx]
    print(f"Found: frame={row['crossing_frame']}  onset={row.get('onset_frame', '—')}  "
          f"maneuver={row['maneuver']}  session={row['session_id']}")

    new_label = args.new_label or args.label
    if new_label:
        df.at[idx, "maneuver"] = new_label
        print(f"  maneuver  →  {new_label}")

    if args.crossing_frame is not None:
        df.at[idx, "crossing_frame"] = args.crossing_frame
        print(f"  crossing_frame  {args.frame}  →  {args.crossing_frame}")

    if args.onset_frame is not None:
        if "onset_frame" not in df.columns:
            raise SystemExit("onset_frame column not present in this CSV")
        df.at[idx, "onset_frame"] = args.onset_frame
        print(f"  onset_frame  →  {args.onset_frame}")

    df.to_csv(args.events, index=False)
    print(f"Saved: {args.events}")


if __name__ == "__main__":
    main()
