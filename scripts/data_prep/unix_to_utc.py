import datetime
import sys

def unix_to_utc(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp, datetime.UTC)
    return dt.strftime("%Y:%m:%d %H:%M:%S")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python unix_to_utc.py <unix_timestamp> <frame_number>")
        sys.exit(1)
    ts = float(sys.argv[1])
    frame_number = int(sys.argv[2])
    ts -= frame_number / 30
    print(unix_to_utc(ts))
