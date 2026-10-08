#!/bin/zsh
set -e

: "${RAW_DIR:?Set RAW_DIR to the folder holding JSON/, GPX/, GPS-CSV/, EVENTS-CSV/, mp4/}"

if [ -z "$1" ]; then
    echo "Usage: bash process_session.sh <session_name>"
    exit 1
fi


SESSION="$1"
PYTHON=${PYTHON:-python}
JSON="$RAW_DIR/JSON/${SESSION}.json"

# cut the metadata json, because gyroflow export json that is in square brackets
cp "$JSON" "${JSON}.tmp"
sed '1d; $d' "${JSON}.tmp" > "$JSON"
rm -f "${JSON}.tmp"

# extract preliminary gps, and visualize the track and timestamps for calibration
$PYTHON extract_gps.py --json $RAW_DIR/JSON/"${SESSION}.json" --cold_start_s 13 --output $RAW_DIR/GPS-CSV/"${SESSION}_gps.csv"
$PYTHON visualize_gps.py --input $RAW_DIR/GPX/"${SESSION}.gpx" --gps_csv $RAW_DIR/GPS-CSV/"${SESSION}_gps.csv" --output $RAW_DIR/TEMP_VIS/"${SESSION}.html"
