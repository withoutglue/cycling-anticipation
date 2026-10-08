#!/bin/bash
set -e

: "${RAW_DIR:?Set RAW_DIR to the folder holding JSON/, GPX/, GPS-CSV/, EVENTS-CSV/, mp4/}"

if [ -z "$1" ]; then
    echo "Usage: bash strip_json.sh <session_name>"
    exit 1
fi

SESSION="$1"
JSON="$RAW_DIR/JSON/${SESSION}.json"

cp "$JSON" "${JSON}.tmp"
sed '1d; $d' "${JSON}.tmp" > "$JSON"
rm -f "${JSON}.tmp"
