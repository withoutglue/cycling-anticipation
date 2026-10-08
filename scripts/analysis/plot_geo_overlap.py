"""
Visualise geographic overlap between train and validation intersections.

OSM intersection nodes within --merge_m metres are merged into one intersection
(a large junction is often several OSM nodes). Produces:
  geo_overlap_map.png     — static map, intersections coloured by split membership
  geo_overlap_map.html    — interactive version (folium), per-intersection popups
  geo_overlap_bars.png    — per class, share of val events at locations seen in training
  geo_overlap_events.csv  — per-event table (intersection id, split, approach, seen flags)

Usage:
    python plot_geo_overlap.py \\
        --events_dir     ~/Desktop/EVENTS-CSV \\
        --gps_dir        ~/Desktop/GPS-CSV \\
        --manifest       clips/manifest.csv \\
        --train_sessions clips/train_sessions.txt \\
        --val_sessions   clips/val_sessions.txt \\
        --output_dir     plots/geo
"""

import argparse
from collections import defaultdict
from pathlib import Path

import contextily as ctx
import folium
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

CLASS_ORDER = ["left", "right", "straight", "stop"]

CATEGORIES = ["train only", "val only", "both"]
CAT_COLORS = {"train only": "#2a78d6", "val only": "#eb6834", "both": "#1baf7a"}
CAT_LABELS = {"train only": "Train only", "val only": "Validation only",
              "both": "Train and validation"}

MEASURES = ["seen_int", "seen_app", "seen_app_man"]
MEASURE_COLORS = {"seen_int": "#86b6ef", "seen_app": "#3987e5", "seen_app_man": "#1c5cab"}
MEASURE_LABELS = {
    "seen_int":     "Same intersection",
    "seen_app":     "Same intersection and approach",
    "seen_app_man": "Same intersection, approach and maneuver",
}

SURFACE  = "#fcfcfb"
GRIDLINE = "#e1e0d9"
AXIS_INK = "#c3c2b7"
TEXT_PRI = "#0b0b0b"
TEXT_SEC = "#52514e"
TEXT_MUT = "#898781"

REGIONS = {"Helsinki": (60.19, 24.95), "London": (51.51, -0.17),
           "Las Palmas": (28.12, -15.43)}
BASEMAP = ctx.providers.Esri.WorldGrayCanvas
FOLIUM_TILES = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")

APPROACH_PTS   = 3   # GPS points (1 Hz) before crossing used for approach heading
MIN_SPEED_KMH  = 5.0  # GPS heading unreliable below this speed

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:3067", always_xy=True)  # ETRS-TM35FIN


def circmean(deg) -> float:
    r = np.radians(np.asarray(deg, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def nearest_region(lat: float, lon: float) -> str:
    names = list(REGIONS)
    return names[int(np.argmin([(lat - a) ** 2 + (lon - b) ** 2 for a, b in REGIONS.values()]))]


def angdiff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def approach_heading(gps: pd.DataFrame, crossing_frame: int) -> float:
    frames = gps["frame"].values
    i = int(np.searchsorted(frames, crossing_frame, side="right")) - 1
    if i <= 0:
        return np.nan
    before = gps.iloc[max(0, i - APPROACH_PTS):i]
    h = before.loc[before["speed_kmh"] >= MIN_SPEED_KMH, "heading"].dropna()
    if len(h):
        return circmean(h.values)
    # Slow approach (common for stops): fall back to displacement bearing
    j = max(0, i - APPROACH_PTS)
    x0, y0 = to_utm.transform(gps["lon"].iloc[j], gps["lat"].iloc[j])
    x1, y1 = to_utm.transform(gps["lon"].iloc[i], gps["lat"].iloc[i])
    if np.hypot(x1 - x0, y1 - y0) < 2.0:
        return np.nan
    return float(np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 360)


def load_events(args) -> pd.DataFrame:
    ev = pd.concat([pd.read_csv(f) for f in sorted(args.events_dir.glob("*_events.csv"))],
                   ignore_index=True)
    man = pd.read_csv(args.manifest)
    man = man[man["maneuver"].isin(CLASS_ORDER)]
    # Event files repeat some (session, crossing_frame) keys — adjacent OSM nodes
    # entered at the same GPS point; keep one so the join stays one row per clip.
    ev = ev.drop_duplicates(["session_id", "crossing_frame"])
    df = man[["session_id", "crossing_frame", "maneuver"]].merge(
        ev[["session_id", "crossing_frame", "osm_node_id",
            "intersection_lat", "intersection_lon"]],
        on=["session_id", "crossing_frame"], how="left", validate="many_to_one")
    missing = df["osm_node_id"].isna().sum()
    print(f"Manifest clips (labeled): {len(df)}   without matching event: {missing}")
    df = df.dropna(subset=["osm_node_id"])

    train = set(args.train_sessions.read_text().split())
    val   = set(args.val_sessions.read_text().split())
    df["split"] = np.where(df["session_id"].isin(train), "train",
                  np.where(df["session_id"].isin(val), "val", None))
    unassigned = df["split"].isna().sum()
    if unassigned:
        print(f"Clips in sessions not listed in either split: {unassigned} (dropped)")
    df = df.dropna(subset=["split"]).reset_index(drop=True)
    df["region"] = [nearest_region(a, b) for a, b in
                    zip(df["intersection_lat"], df["intersection_lon"])]

    approaches = np.full(len(df), np.nan)
    for sid, idx in df.groupby("session_id").groups.items():
        gps = pd.read_csv(args.gps_dir / f"{sid}.csv").sort_values("frame")
        for k in idx:
            approaches[k] = approach_heading(gps, int(df.at[k, "crossing_frame"]))
    df["approach"] = approaches
    print(f"Events without approach heading: {int(np.isnan(approaches).sum())}")
    return df


def cluster_nodes(df: pd.DataFrame, merge_m: float) -> pd.DataFrame:
    nodes = (df.groupby("osm_node_id")[["intersection_lat", "intersection_lon"]]
               .first().reset_index())
    x, y = to_utm.transform(nodes["intersection_lon"].values, nodes["intersection_lat"].values)
    xy = np.column_stack([x, y])
    # Complete linkage: no two nodes in one intersection are more than merge_m apart
    # (single linkage chains along dense street grids)
    labels = fcluster(linkage(xy, method="complete"), t=merge_m, criterion="distance") - 1
    nodes["intersection_id"] = labels

    spans = [pdist(xy[labels == c]).max()
             for c in np.unique(labels) if (labels == c).sum() > 1]
    print(f"OSM nodes: {len(nodes)}  →  intersections: {labels.max() + 1}  "
          f"(widest merged span: {max(spans, default=0):.1f} m)")
    return df.merge(nodes[["osm_node_id", "intersection_id"]], on="osm_node_id")


def add_seen_flags(df: pd.DataFrame, tol_deg: float) -> pd.DataFrame:
    train_by_int = defaultdict(list)
    for r in df[df["split"] == "train"].itertuples():
        train_by_int[r.intersection_id].append((r.approach, r.maneuver))

    flags = []
    for r in df.itertuples():
        if r.split != "val":
            flags.append((np.nan, np.nan, np.nan))
            continue
        tr = train_by_int.get(r.intersection_id, [])
        seen_int = float(len(tr) > 0)
        if np.isnan(r.approach):
            flags.append((seen_int, np.nan, np.nan))
            continue
        same_app = [m for a, m in tr if not np.isnan(a) and angdiff(a, r.approach) <= tol_deg]
        flags.append((seen_int, float(len(same_app) > 0), float(r.maneuver in same_app)))
    df[MEASURES] = pd.DataFrame(flags, index=df.index)
    return df


def summarise_intersections(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("intersection_id")
    s = pd.DataFrame({
        "lat": g["intersection_lat"].mean(),
        "lon": g["intersection_lon"].mean(),
        "region": g["region"].first(),
        "n_osm_nodes": g["osm_node_id"].nunique(),
        "n_train": g["split"].apply(lambda x: (x == "train").sum()),
        "n_val":   g["split"].apply(lambda x: (x == "val").sum()),
    })
    for split in ["train", "val"]:
        for c in CLASS_ORDER:
            s[f"{split}_{c}"] = g.apply(
                lambda x, sp=split, cl=c: ((x["split"] == sp) & (x["maneuver"] == cl)).sum())
    s["n_total"] = s["n_train"] + s["n_val"]
    s["category"] = np.where((s["n_train"] > 0) & (s["n_val"] > 0), "both",
                    np.where(s["n_train"] > 0, "train only", "val only"))
    return s.reset_index()


# ── static map ────────────────────────────────────────────────────────────────

def draw_region(ax, gdf: gpd.GeoDataFrame, region: str, min_span_m: float = 2000):
    sub_r = gdf[gdf["region"] == region]
    for cat in CATEGORIES:  # "both" drawn last, on top
        sub = sub_r[sub_r["category"] == cat]
        ax.scatter(sub.geometry.x, sub.geometry.y, s=24,
                   color=CAT_COLORS[cat], edgecolor="white", linewidth=0.6,
                   alpha=0.9, zorder=3,
                   label=f"{CAT_LABELS[cat]} ({(gdf['category'] == cat).sum()})")

    # Equal-aspect extent with padding and a minimum span for small regions
    x0, x1 = sub_r.geometry.x.min(), sub_r.geometry.x.max()
    y0, y1 = sub_r.geometry.y.min(), sub_r.geometry.y.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, min_span_m) * 0.55
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ctx.add_basemap(ax, source=BASEMAP, crs="EPSG:3857", attribution_size=5)
    ax.set_axis_off()

    ax.set_title(region, fontsize=10, color=TEXT_SEC, loc="left")


def plot_static_map(ints: pd.DataFrame, output_dir: Path):
    gdf = gpd.GeoDataFrame(ints, geometry=gpd.points_from_xy(ints["lon"], ints["lat"]),
                           crs="EPSG:4326").to_crs("EPSG:3857")

    fig = plt.figure(figsize=(12, 7.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], wspace=0.05, hspace=0.15)
    ax_main = fig.add_subplot(gs[:, 0])
    draw_region(ax_main, gdf, "Helsinki")
    draw_region(fig.add_subplot(gs[0, 1]), gdf, "London")
    draw_region(fig.add_subplot(gs[1, 1]), gdf, "Las Palmas")

    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=11,
                      markerfacecolor=CAT_COLORS[cat], markeredgecolor="white",
                      label=f"{CAT_LABELS[cat]} ({(gdf['category'] == cat).sum()})")
               for cat in CATEGORIES]
    ax_main.legend(handles=handles, fontsize=9.5, frameon=True, facecolor=SURFACE,
                   edgecolor=AXIS_INK, labelcolor=TEXT_SEC, loc="upper left",
                   title="Intersections (all cities)", title_fontsize=10, labelspacing=0.8)

    out = output_dir / "geo_overlap_map.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


# ── interactive map ───────────────────────────────────────────────────────────

def plot_interactive_map(ints: pd.DataFrame, output_dir: Path):
    m = folium.Map(location=list(REGIONS["Helsinki"]), zoom_start=13,
                   tiles=FOLIUM_TILES, attr="Tiles © Esri — Esri, DeLorme, NAVTEQ")

    for cat in CATEGORIES:
        layer = folium.FeatureGroup(name=f"{CAT_LABELS[cat]} ({(ints['category'] == cat).sum()})")
        for r in ints[ints["category"] == cat].itertuples():
            rows = "".join(
                f"<tr><td>{c}</td><td align='right'>{getattr(r, 'train_' + c)}</td>"
                f"<td align='right'>{getattr(r, 'val_' + c)}</td></tr>"
                for c in CLASS_ORDER)
            popup = (f"<b>Intersection {r.intersection_id}</b> "
                     f"({r.n_osm_nodes} OSM node{'s' if r.n_osm_nodes > 1 else ''})<br>"
                     f"<table><tr><th></th><th>Train</th><th>Val</th></tr>{rows}"
                     f"<tr><td><b>total</b></td><td align='right'><b>{r.n_train}</b></td>"
                     f"<td align='right'><b>{r.n_val}</b></td></tr></table>")
            folium.CircleMarker(
                location=[r.lat, r.lon], radius=3 + 1.2 * np.sqrt(r.n_total),
                color="white", weight=1, fill=True, fill_color=CAT_COLORS[cat],
                fill_opacity=0.85, popup=folium.Popup(popup, max_width=240),
                tooltip=f"{CAT_LABELS[cat]}: {r.n_train} train / {r.n_val} val",
            ).add_to(layer)
        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out = output_dir / "geo_overlap_map.html"
    m.save(str(out))
    print(f"Saved {out}")


# ── overlap bars ──────────────────────────────────────────────────────────────

def plot_overlap_bars(df: pd.DataFrame, output_dir: Path):
    val = df[df["split"] == "val"]
    groups = CLASS_ORDER + ["all"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS_INK)
    ax.tick_params(colors=TEXT_MUT, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    bar_w = 0.26
    xs = np.arange(len(groups))
    print("\nShare of validation events at locations seen in training:")
    print(f"  {'class':9s} {'n':>5s}  " + "  ".join(f"{m:>13s}" for m in MEASURES))
    for gi, g in enumerate(groups):
        sub = val if g == "all" else val[val["maneuver"] == g]
        vals = [sub[m].mean() for m in MEASURES]  # NaN (no approach) excluded by mean()
        print(f"  {g:9s} {len(sub):5d}  " + "  ".join(f"{v:13.2f}" for v in vals))
        for mi, (meas, v) in enumerate(zip(MEASURES, vals)):
            x = xs[gi] + (mi - 1) * bar_w
            ax.bar(x, v, width=bar_w * 0.9, color=MEASURE_COLORS[meas], zorder=3,
                   linewidth=0, label=MEASURE_LABELS[meas] if gi == 0 else None)
            ax.text(x, v + 0.01, f"{v:.0%}", ha="center", va="bottom",
                    fontsize=7, color=TEXT_MUT)

    n_per = [len(val) if g == "all" else (val["maneuver"] == g).sum() for g in groups]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g.capitalize()}\n(n={n})" for g, n in zip(groups, n_per)],
                       fontsize=9, color=TEXT_SEC)
    ax.set_ylabel("Share of validation events", fontsize=10, color=TEXT_SEC)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=8.5, frameon=False, labelcolor=TEXT_SEC,
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    fig.tight_layout()
    out = output_dir / "geo_overlap_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events_dir", type=Path,
                        default=Path("~/Desktop/EVENTS-CSV").expanduser())
    parser.add_argument("--gps_dir", type=Path,
                        default=Path("~/Desktop/GPS-CSV").expanduser())
    parser.add_argument("--manifest", type=Path, default=Path("clips/manifest.csv"))
    parser.add_argument("--train_sessions", type=Path, default=Path("clips/train_sessions.txt"))
    parser.add_argument("--val_sessions", type=Path, default=Path("clips/val_sessions.txt"))
    parser.add_argument("--output_dir", type=Path, default=Path("plots/geo"))
    parser.add_argument("--merge_m", type=float, default=20.0,
                        help="Merge OSM nodes closer than this into one intersection")
    parser.add_argument("--approach_tol_deg", type=float, default=45.0,
                        help="Approach headings within this angle count as the same approach")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_events(args)
    print(df.groupby(["split", "maneuver"]).size().unstack(fill_value=0)[CLASS_ORDER])
    df = cluster_nodes(df, args.merge_m)
    df = add_seen_flags(df, args.approach_tol_deg)
    ints = summarise_intersections(df)
    print("\nEvents by region:")
    print(df.groupby(["region", "split"]).size().unstack(fill_value=0))
    val = df[df["split"] == "val"]
    print("\nValidation overlap by region:")
    print(val.groupby("region")[MEASURES].mean().round(2))
    print("\nIntersections by split membership:")
    for cat in CATEGORIES:
        sub = ints[ints["category"] == cat]
        print(f"  {cat:11s} {len(sub):4d} intersections   "
              f"{sub['n_train'].sum():5d} train events   {sub['n_val'].sum():5d} val events")

    df.to_csv(args.output_dir / "geo_overlap_events.csv", index=False)
    plot_static_map(ints, args.output_dir)
    plot_interactive_map(ints, args.output_dir)
    plot_overlap_bars(df, args.output_dir)
    print("Done.")
