#!/bin/zsh
set -e

: "${RAW_DIR:?Set RAW_DIR to the folder holding JSON/, GPX/, GPS-CSV/, EVENTS-CSV/, mp4/}"

PYTHON=${PYTHON:-python}

EVENTS_DIR="$RAW_DIR/EVENTS-CSV"
MP4_DIR="$RAW_DIR/mp4"
OUTPUT_DIR="${OUTPUT_DIR:-./clips}"
SCRIPT="$(dirname "$0")/extract_clips.py"

for events in "$EVENTS_DIR"/VID_*_events.csv; do
    session=$(basename "$events" _events.csv)
    video="$MP4_DIR/${session}.mp4"

    if [ ! -f "$video" ]; then
        echo "SKIP $session — no mp4 found"
        continue
    fi

    echo "Processing $session..."
    $PYTHON "$SCRIPT" \
        --video "$video" \
        --events "$events" \
        --output_dir "$OUTPUT_DIR"
done
