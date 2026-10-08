"""
Detect intersection crossings in a GPS track and classify maneuvers.

Uses OpenStreetMap (via osmnx) to find where the GPS track passes through
road intersections, then classifies each crossing as:
  left, right, straight, stop, unknown

The OSM network is downloaded once and cached as a GraphML file. On
subsequent runs the cache is reloaded without hitting the network.

Output: one row per crossing event with frame index and maneuver label.

Usage:
    python detect_intersections.py --gps session_gps.csv --output session_events.csv

    # Custom OSM cache (shared across sessions in the same area):
    python detect_intersections.py --gps session_gps.csv --output session_events.csv \\
        --osm_cache data/osm/helsinki.graphml
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

# ── defaults ─────────────────────────────────────────────────────────────────
RADIUS_M        = 15.0   # detection radius around intersection node (metres)
DEBOUNCE_S      = 30.0   # per-node cooldown after it fires (seconds)
GLOBAL_COOL_S   =  5.0   # minimum gap between any two events (seconds)
WINDOW_S        =  3     # GPS points before/after crossing for heading
STOP_WINDOW_S   = 10     # wider look-ahead for stop classification
MIN_SPEED_KMH   =  5.0   # heading is unreliable below this speed
STOP_SPEED_KMH     =  0.01 # speed threshold for consecutive-point stop check
STOP_CONSEC_PTS    =  3    # min consecutive points below STOP_SPEED_KMH → stop
STOP_ONSET_KMH     =  1.5  # speed threshold for stop onset detection
ONSET_PAST_CENTER_M = 5.0  # metres past intersection center for turn/straight onset
OSM_BUFFER_DEG  =  0.005 # bbox padding around GPS track (~500 m at 60°N)
DEFAULT_CACHE   = Path(__file__).parent / "osm_cache" / "network.graphml"



# ── OSM loading ───────────────────────────────────────────────────────────────

def _build_spatial_index(G, transformer: Transformer):
    """Extract intersection nodes and build a UTM KDTree.

    Works with unsimplified graphs (simplify=False): uses unique neighbour count
    instead of street_count, which is only set on simplified graphs.
    A node is an intersection if 3+ distinct ways meet there.
    """
    nodes = {}
    for nid, d in G.nodes(data=True):
        unique_neighbors = set(G.successors(nid)) | set(G.predecessors(nid))
        if len(unique_neighbors) >= 3:
            nodes[nid] = d

    if not nodes:
        raise ValueError("No intersection nodes found in OSM network.")

    node_ids  = list(nodes.keys())
    node_lons = [nodes[n]["x"] for n in node_ids]
    node_lats = [nodes[n]["y"] for n in node_ids]
    node_x, node_y = transformer.transform(node_lons, node_lats)
    node_utm = np.column_stack([node_x, node_y])

    return node_ids, nodes, node_utm, cKDTree(node_utm)


def load_osm(gps_df: pd.DataFrame, cache_path: Path,
             transformer: Transformer, network_type: str = "all"):
    """
    Load (or download + cache) the OSM road network for the GPS track's area.
    Returns (node_ids, nodes_dict, node_utm, tree).
    """
    if cache_path.exists():
        print(f"  OSM cache found: {cache_path.name}")
        G = ox.load_graphml(cache_path)
    else:
        acquired = gps_df[gps_df["is_acquired"]]
        north = acquired["lat"].max() + OSM_BUFFER_DEG
        south = acquired["lat"].min() - OSM_BUFFER_DEG
        east  = acquired["lon"].max() + OSM_BUFFER_DEG
        west  = acquired["lon"].min() - OSM_BUFFER_DEG
        print(f"  Downloading OSM ({network_type}) for "
              f"({south:.4f},{west:.4f})–({north:.4f},{east:.4f}) …")
        G = ox.graph_from_bbox((west, south, east, north),
                               network_type=network_type,
                               simplify=False)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(G, cache_path)
        print(f"  Saved to {cache_path}")

    node_ids, nodes, node_utm, tree = _build_spatial_index(G, transformer)
    print(f"  Intersection nodes: {len(node_ids)}")
    return node_ids, nodes, node_utm, tree


# ── maneuver classification ───────────────────────────────────────────────────

def _circular_mean(angles_deg: np.ndarray) -> float:
    s = np.mean(np.sin(np.radians(angles_deg)))
    c = np.mean(np.cos(np.radians(angles_deg)))
    return float(np.degrees(np.arctan2(s, c)) % 360)


def _classify(gps: pd.DataFrame, i: int,
              window: int, min_speed: float,
              stop_speed: float) -> tuple:
    before     = gps.iloc[max(0, i - window) : i]
    after      = gps.iloc[i + 1 : i + 1 + window]
    after_wide = gps.iloc[i + 1 : i + 1 + STOP_WINDOW_S]

    # Stop: at least STOP_CONSEC_PTS consecutive points below STOP_SPEED_KMH
    speeds = after_wide["speed_kmh"].values
    consec = max((sum(1 for _ in g) for v, g in itertools.groupby(speeds < STOP_SPEED_KMH) if v), default=0)
    if consec >= STOP_CONSEC_PTS:
        return "stop", None, None, None

    before_h = before.loc[before["speed_kmh"] >= min_speed, "heading"].dropna()
    after_h  = after.loc[after["speed_kmh"]  >= min_speed, "heading"].dropna()

    if len(before_h) < 2 or len(after_h) < 2:
        return "unknown", None, None, None

    approach = _circular_mean(before_h.values)
    exit_h   = _circular_mean(after_h.values)
    delta    = float((exit_h - approach + 180) % 360 - 180)

    if delta < -30:
        label = "left"
    elif delta > 30:
        label = "right"
    else:
        label = "straight"

    return label, round(approach, 2), round(exit_h, 2), round(delta, 2)


# ── onset detection ──────────────────────────────────────────────────────────

def _find_onset_idx(gps: pd.DataFrame, gps_utm: np.ndarray,
                    node_utm_point: np.ndarray, j: int,
                    maneuver: str, radius: float) -> int:
    """Return the GPS index of the maneuver onset within the intersection radius.

    stop           — first point where speed_kmh < STOP_ONSET_KMH; fallback: apex
    left/right/
    straight       — first point after apex where distance > ONSET_PAST_CENTER_M; fallback: last in-radius
    unknown        — apex (closest approach)
    """
    in_radius = []
    for k in range(j, len(gps)):
        if np.linalg.norm(node_utm_point - gps_utm[k]) <= radius:
            in_radius.append(k)
        elif in_radius:
            break

    if not in_radius:
        return j

    dists = np.array([np.linalg.norm(node_utm_point - gps_utm[k]) for k in in_radius])
    apex_list_idx = int(np.argmin(dists))
    apex_idx = in_radius[apex_list_idx]

    if maneuver == "stop":
        for k in in_radius:
            if gps.iloc[k]["speed_kmh"] < STOP_ONSET_KMH:
                return k
        return apex_idx

    if maneuver in ("left", "right", "straight"):
        for k in in_radius[apex_list_idx + 1:]:
            if np.linalg.norm(node_utm_point - gps_utm[k]) > ONSET_PAST_CENTER_M:
                return k
        return in_radius[-1]

    return apex_idx


# ── main detection ────────────────────────────────────────────────────────────

def detect(gps_df: pd.DataFrame, node_ids: list, nodes: dict,
           node_utm: np.ndarray, tree: cKDTree,
           transformer: Transformer, session_id: str,
           radius: float, debounce: float, global_cool: float,
           window: int, min_speed: float, stop_speed: float) -> pd.DataFrame:

    gps = gps_df[gps_df["is_acquired"]].reset_index(drop=True)

    gps_x, gps_y = transformer.transform(gps["lon"].values, gps["lat"].values)
    gps_utm = np.column_stack([gps_x, gps_y])

    node_cooldown: dict = {}
    last_event_ts = -np.inf
    events = []

    for i in range(len(gps)):
        row = gps.iloc[i]
        if row["speed_kmh"] < min_speed:
            continue

        hits = tree.query_ball_point(gps_utm[i], radius)
        if not hits:
            continue

        dists   = np.linalg.norm(node_utm[hits] - gps_utm[i], axis=1)
        closest = hits[int(np.argmin(dists))]
        node_id = node_ids[closest]
        ts      = float(row["unix_timestamp"])

        if ts - node_cooldown.get(node_id, -np.inf) < debounce:
            continue
        if ts - last_event_ts < global_cool:
            continue

        node_cooldown[node_id] = ts
        last_event_ts = ts

        # Walk back to the first GPS point continuously within radius —
        # the actual crossing moment, not the cooldown-delayed trigger.
        j = i
        while j > 0 and np.linalg.norm(node_utm[closest] - gps_utm[j - 1]) <= radius:
            j -= 1
        crossing_row = gps.iloc[j]

        # Classify using i so the after-window captures the exit heading,
        # not the inside-intersection heading that j's after-window would see.
        maneuver, approach, exit_h, delta = _classify(
            gps, i, window, min_speed, stop_speed
        )

        onset_idx = _find_onset_idx(
            gps, gps_utm, node_utm[closest], j, maneuver, radius
        )
        onset_row = gps.iloc[onset_idx]

        after_slice = gps.iloc[i + 1 : i + 1 + window]
        events.append({
            "session_id":           session_id,
            "osm_node_id":          node_id,
            "crossing_frame":       int(crossing_row["frame"]),
            "unix_timestamp":       float(crossing_row["unix_timestamp"]),
            "onset_frame":          int(onset_row["frame"]),
            "onset_unix_timestamp": float(onset_row["unix_timestamp"]),
            "maneuver":             maneuver,
            "approach_heading":     approach,
            "exit_heading":         exit_h,
            "heading_change":       delta,
            "speed_before_kmh":     round(gps.iloc[max(0, j - window):j]["speed_kmh"].mean(), 2),
            "speed_after_kmh":      round(after_slice["speed_kmh"].mean(), 2) if len(after_slice) else None,
            "intersection_lat":     round(nodes[node_id]["y"], 7),
            "intersection_lon":     round(nodes[node_id]["x"], 7),
        })

    return pd.DataFrame(events)


def print_summary(events: pd.DataFrame):
    if events.empty:
        print("  No crossings detected.")
        return
    counts = events["maneuver"].value_counts().to_dict()
    print(f"  Crossings detected: {len(events)}")
    for label in ("left", "right", "straight", "stop", "unknown"):
        n = counts.get(label, 0)
        if n:
            print(f"    {label:10s}: {n}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect intersection crossings and classify maneuvers from GPS CSV."
    )
    parser.add_argument("--gps",        type=Path, required=True,
                        help="GPS CSV from extract_gps.py")
    parser.add_argument("--output",     type=Path, required=True,
                        help="Output events CSV")
    parser.add_argument("--session_id", type=str, default=None,
                        help="Session identifier (default: GPS CSV stem)")
    parser.add_argument("--osm_cache",  type=Path, default=DEFAULT_CACHE,
                        help=f"GraphML cache path (default: {DEFAULT_CACHE})")
    parser.add_argument("--network",    type=str, default="all",
                        help="OSM network type: all, drive, bike (default: all)")
    parser.add_argument("--radius",     type=float, default=RADIUS_M)
    parser.add_argument("--debounce",   type=float, default=DEBOUNCE_S)
    parser.add_argument("--window",     type=int,   default=WINDOW_S)
    parser.add_argument("--min_speed",  type=float, default=MIN_SPEED_KMH)
    parser.add_argument("--stop_speed", type=float, default=STOP_SPEED_KMH,
                        help=f"Speed threshold (km/h) for consecutive-point stop check (default: {STOP_SPEED_KMH})")

    args = parser.parse_args()

    session_id = args.session_id or args.gps.stem

    print(f"\n{args.gps.name}")
    gps_df = pd.read_csv(args.gps)
    print(f"  GPS fixes: {len(gps_df)}  ({gps_df['is_acquired'].sum()} acquired)")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True)

    node_ids, nodes, node_utm, tree = load_osm(
        gps_df, args.osm_cache, transformer, network_type=args.network
    )

    events = detect(
        gps_df, node_ids, nodes, node_utm, tree, transformer,
        session_id=session_id,
        radius=args.radius,
        debounce=args.debounce,
        global_cool=GLOBAL_COOL_S,
        window=args.window,
        min_speed=args.min_speed,
        stop_speed=args.stop_speed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output, index=False)
    print_summary(events)
    print(f"\nEvents saved to: {args.output}")

    args.osm_cache.unlink(missing_ok=True)
    print(f"OSM cache deleted: {args.osm_cache}")
