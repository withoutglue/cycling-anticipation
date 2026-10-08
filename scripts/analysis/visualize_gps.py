"""
Visualize a GPS track from a GPX file on an OpenStreetMap background.

Outputs an interactive HTML map (Leaflet.js via folium) with the track
drawn as a polyline, plus start/end markers and basic statistics.

Usage:
    python visualize_gps.py --input track.gpx
    python visualize_gps.py --input track.gpx --output map.html
    python visualize_gps.py --input track.gpx --output map.html --speed
"""

import argparse
from pathlib import Path

import folium
import gpxpy
import gpxpy.gpx
import numpy as np
import pandas as pd


def load_gpx(path: Path) -> list[dict]:
    """Parse GPX file and return a list of {lat, lon, time, elevation} dicts."""
    with open(path) as f:
        gpx = gpxpy.parse(f)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append({
                    "lat":  pt.latitude,
                    "lon":  pt.longitude,
                    "time": pt.time,
                    "ele":  pt.elevation,
                })

    if not points:
        for wpt in gpx.waypoints:
            points.append({
                "lat":  wpt.latitude,
                "lon":  wpt.longitude,
                "time": wpt.time,
                "ele":  wpt.elevation,
            })

    return points


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance in metres between two lat/lon points."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_speeds(points: list[dict]) -> list[float | None]:
    """Return speed in km/h between consecutive points (None if no timestamps)."""
    speeds = [None]
    for i in range(1, len(points)):
        p0, p1 = points[i - 1], points[i]
        if p0["time"] is None or p1["time"] is None:
            speeds.append(None)
            continue
        dt = (p1["time"] - p0["time"]).total_seconds()
        if dt <= 0:
            speeds.append(None)
            continue
        dist = haversine_m(p0["lat"], p0["lon"], p1["lat"], p1["lon"])
        speeds.append(dist / dt * 3.6)
    return speeds


def speed_color(kmh: float | None) -> str:
    if kmh is None:
        return "#3388ff"
    if kmh < 5:
        return "#d73027"   # near-stop: red
    if kmh < 15:
        return "#fc8d59"   # slow: orange
    if kmh < 25:
        return "#91cf60"   # normal cycling: green
    return "#1a9850"       # fast: dark green


_MANEUVER_COLOR = {
    "left":     "blue",
    "right":    "orange",
    "straight": "green",
    "stop":     "red",
    "unknown":  "gray",
}


def add_gps_points(m: folium.Map, gps_csv_path: Path) -> None:
    """Render each acquired GPS fix as a small clickable dot showing unix_timestamp."""
    df = pd.read_csv(gps_csv_path)
    for _, row in df[df["is_acquired"]].iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color="#333333",
            fill=True,
            fill_color="#ffffff",
            fill_opacity=0.8,
            weight=1,
            popup=folium.Popup(f"{row['unix_timestamp']:.3f}", max_width=160),
            tooltip=f"{row['unix_timestamp']:.0f}",
        ).add_to(m)


def add_intersection_events(m: folium.Map, events_path: Path) -> None:
    events = pd.read_csv(events_path)
    for _, row in events.iterrows():
        maneuver = str(row.get("maneuver", "unknown"))
        color = _MANEUVER_COLOR.get(maneuver, "gray")

        heading_change = row.get("heading_change")
        hc_str = f"{heading_change:+.1f}°" if pd.notna(heading_change) else "—"

        onset_frame = row.get("onset_frame")
        onset_str = str(int(onset_frame)) if pd.notna(onset_frame) else "—"

        popup_html = (
            f"<b>{maneuver.upper()}</b><br>"
            f"Crossing frame: {int(row['crossing_frame'])}<br>"
            f"Onset frame: {onset_str}<br>"
            f"Heading Δ: {hc_str}<br>"
            f"Speed before: {row.get('speed_before_kmh', '—'):.1f} km/h<br>"
            f"Speed after: {row.get('speed_after_kmh', '—'):.1f} km/h"
        )

        folium.CircleMarker(
            location=[row["intersection_lat"], row["intersection_lon"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=maneuver,
            popup=folium.Popup(popup_html, max_width=200),
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;right:30px;z-index:1000;
                background:white;padding:10px;border-radius:6px;
                border:1px solid #ccc;font-size:13px;">
      <b>Maneuver</b><br>
      <span style="color:green">&#9679;</span> Straight<br>
      <span style="color:blue">&#9679;</span> Left<br>
      <span style="color:orange">&#9679;</span> Right<br>
      <span style="color:red">&#9679;</span> Stop<br>
      <span style="color:gray">&#9679;</span> Unknown
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def build_map(points: list[dict], show_speed: bool = False) -> folium.Map:
    center_lat = np.mean([p["lat"] for p in points])
    center_lon = np.mean([p["lon"] for p in points])

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15,
                   tiles="OpenStreetMap")

    coords = [(p["lat"], p["lon"]) for p in points]

    if show_speed:
        speeds = compute_speeds(points)
        for i in range(1, len(coords)):
            folium.PolyLine(
                [coords[i - 1], coords[i]],
                color=speed_color(speeds[i]),
                weight=4,
                opacity=0.85,
            ).add_to(m)

        # Legend
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:white;padding:10px;border-radius:6px;
                    border:1px solid #ccc;font-size:13px;">
          <b>Speed</b><br>
          <span style="color:#d73027">&#9646;</span> &lt;5 km/h<br>
          <span style="color:#fc8d59">&#9646;</span> 5–15 km/h<br>
          <span style="color:#91cf60">&#9646;</span> 15–25 km/h<br>
          <span style="color:#1a9850">&#9646;</span> &gt;25 km/h
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
    else:
        folium.PolyLine(coords, color="#3388ff", weight=4, opacity=0.85).add_to(m)

    # Start marker
    folium.Marker(
        coords[0],
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    # End marker
    folium.Marker(
        coords[-1],
        tooltip="End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    return m


def print_stats(points: list[dict]):
    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    total_dist = sum(
        haversine_m(points[i - 1]["lat"], points[i - 1]["lon"],
                    points[i]["lat"],     points[i]["lon"])
        for i in range(1, len(points))
    )
    has_time = points[0]["time"] is not None and points[-1]["time"] is not None
    duration = (
        (points[-1]["time"] - points[0]["time"]).total_seconds()
        if has_time else None
    )

    print(f"  Points   : {len(points)}")
    print(f"  Distance : {total_dist / 1000:.2f} km")
    if duration:
        print(f"  Duration : {duration / 60:.1f} min")
        print(f"  Avg speed: {total_dist / duration * 3.6:.1f} km/h")
    print(f"  Lat range: {min(lats):.6f} – {max(lats):.6f}")
    print(f"  Lon range: {min(lons):.6f} – {max(lons):.6f}")
    if points[0]["time"]:
        print(f"  Start    : {points[0]['time'].isoformat()}")
    if points[-1]["time"]:
        print(f"  End      : {points[-1]['time'].isoformat()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a GPX track on OpenStreetMap."
    )
    parser.add_argument("--input",  type=Path, required=True, help="GPX file")
    parser.add_argument("--output", type=Path, help="Output HTML (default: <input>.html)")
    parser.add_argument("--speed",  action="store_true",
                        help="Color track by speed (requires timestamps)")
    parser.add_argument("--events", type=Path, default=None,
                        help="Intersection events CSV from detect_intersections.py")
    parser.add_argument("--gps_csv", type=Path, default=None,
                        help="GPS CSV from extract_gps.py — shows each fix as a clickable dot with unix_timestamp")
    args = parser.parse_args()

    output = args.output or args.input.with_suffix(".html")

    points = load_gpx(args.input)
    if not points:
        raise SystemExit(f"No GPS points found in {args.input}")

    print(f"\n{args.input.name}")
    print_stats(points)

    m = build_map(points, show_speed=args.speed)

    if args.gps_csv:
        add_gps_points(m, args.gps_csv)
        print(f"GPS fixes overlaid from: {args.gps_csv.name}")

    if args.events:
        add_intersection_events(m, args.events)
        print(f"Intersection events overlaid from: {args.events.name}")

    m.save(str(output))
    print(f"\nMap saved to: {output}")
    print("Open in a browser to inspect the track.")
