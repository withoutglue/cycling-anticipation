#!/bin/zsh
set -e

: "${RAW_DIR:?Set RAW_DIR to the folder holding JSON/, GPX/, GPS-CSV/, EVENTS-CSV/, mp4/}"

if [ -z "$3" ]; then
    echo "Usage: ./process_session_calibrated.sh <session_name> <unix_timestamp> <frame_number>"
    exit 1
fi

SESSION="$1"
UNIX_TIMESTAMP="$2"
FRAME_NUMBER="$3"
PYTHON=${PYTHON:-python}

START_TIME=$($PYTHON unix_to_utc.py "$UNIX_TIMESTAMP" "$FRAME_NUMBER")

$PYTHON extract_gps.py --json $RAW_DIR/JSON/"${SESSION}.json" --start_time "$START_TIME" --output $RAW_DIR/GPS-CSV/"${SESSION}_gps.csv"
$PYTHON detect_intersections.py --gps $RAW_DIR/GPS-CSV/"${SESSION}_gps.csv" --output $RAW_DIR/EVENTS-CSV/"${SESSION}_events.csv"
$PYTHON visualize_gps.py --input $RAW_DIR/GPX/"${SESSION}.gpx" --events $RAW_DIR/EVENTS-CSV/"${SESSION}_events.csv" --output $RAW_DIR/VISUALIZATIONS/"${SESSION}.html"
