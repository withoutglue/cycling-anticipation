"""
Extract training metrics from a SLURM output log and write metrics.csv.

Usage:
    python parse_slurm_log.py cyclist_probe_12345.out
    python parse_slurm_log.py cyclist_probe_12345.out --output my_metrics.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

EPOCH_RE  = re.compile(
    r"Epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+val_acc=([\d.]+)"
)
CLASS_RE  = re.compile(r"(left|right|straight|stop)\s*:\s*([\d.]+|—)")
COLUMNS   = ["epoch", "train_loss", "val_loss", "val_acc",
             "acc_left", "acc_right", "acc_straight", "acc_stop"]


def parse(log_path: Path) -> list[dict]:
    rows = []
    current = None

    for line in log_path.read_text().splitlines():
        m = EPOCH_RE.search(line)
        if m:
            if current is not None:
                rows.append(current)
            current = {
                "epoch":      int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss":   float(m.group(3)),
                "val_acc":    float(m.group(4)),
            }
            continue

        if current is not None:
            m = CLASS_RE.search(line)
            if m:
                name, val = m.group(1), m.group(2)
                current[f"acc_{name}"] = float(val) if val != "—" else float("nan")

    if current is not None:
        rows.append(current)

    return rows


def main(args):
    log_path = Path(args.log)
    if not log_path.exists():
        sys.exit(f"File not found: {log_path}")

    rows = parse(log_path)
    if not rows:
        sys.exit("No epoch lines found — check that the log path is correct.")

    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} epochs → {out_path}")
    print(f"  Epoch range: {rows[0]['epoch']} – {rows[-1]['epoch']}")
    print(f"  Best val_acc: {max(r['val_acc'] for r in rows):.3f} "
          f"(epoch {max(rows, key=lambda r: r['val_acc'])['epoch']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log",            help="SLURM .out log file")
    parser.add_argument("--output", "-o", default="metrics.csv")
    main(parser.parse_args())
