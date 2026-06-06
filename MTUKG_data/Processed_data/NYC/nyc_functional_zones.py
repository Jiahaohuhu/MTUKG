#!/usr/bin/env python3
"""
Build quarterly functional zones for NYC from road blocks, commercial POI events,
land development, road repair, and road network change events.

Version v4 update:
- All four temporal inputs are modeled as interval events by quarter overlap.
- Boundary features are additionally retained through per-quarter start/end signals.

This script follows a block-first approach:
1) Polygonize road centerlines into stable road blocks.
2) Build static block features from block morphology, road types and node types.
3) For each quarter, compute distance-decayed dynamic features from:
   - active commercial POIs
   - land development events
   - road repair events
   - road change events
4) Cluster blocks into quarterly functional states.
5) Merge adjacent blocks with the same state into quarterly functional zones.
6) Build inter-quarter transition links between zones.

Important note
--------------
This implementation supports an optional static background POI table through
--background-poi-csv. It is designed to work directly with the uploaded
poi_filter.csv schema (lng, lat, cate) and can also read several common POI schemas.

To reduce double counting with dynamic commercial POI events, the default behavior is
--background-poi-mode stable, which keeps relatively stable background categories
(such as medical, education, government, public services, worship, transportation,
residential, and parking) and excludes high-churn commercial categories.
You may switch to --background-poi-mode all to use all POIs as static background.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import wkt
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    from shapely import set_precision
except Exception:  # pragma: no cover
    set_precision = None

warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")
warnings.filterwarnings("ignore", message=".*keep_geom_type=True.*")

WGS84 = "EPSG:4326"
NYC_CRS = "EPSG:2263"

# -----------------------------
# Semantic grouping parameters
# -----------------------------
BUSINESS_GROUP_RULES: Dict[str, str] = {
    # hospitality / food-adjacent
    "Hotel": "hospitality_food",
    "Third Party Food Delivery Service": "hospitality_food",
    # mobility / parking / transport
    "Garage & Parking Lot": "mobility_parking",
    "Pedicab Business": "mobility_parking",
    "Pedicab Driver": "mobility_parking",
    "Horse Drawn Cab Owner": "mobility_parking",
    "Sightseeing Bus": "mobility_parking",
    "Tow Truck Company": "mobility_parking",
    "Tow Truck Driver": "mobility_parking",
    "Car Wash": "mobility_parking",
    "Booting Company": "mobility_parking",
    # entertainment / gaming
    "Bingo Game Operator": "gaming_entertainment",
    "Commercial Lessor - Bingo": "gaming_entertainment",
    "Games of Chance - Bell Jar": "gaming_entertainment",
    "Games of Chance - Las Vegas / Casino Nights": "gaming_entertainment",
    "Games of Chance - Raffle with Net Proceeds Over $30,000": "gaming_entertainment",
    "Games of Chance - Raffle with Net Proceeds Under $30,000": "gaming_entertainment",
    # retail / trade
    "Electronics Store": "retail_trade",
    "Electronic Cigarette Dealer": "retail_trade",
    "Tobacco Retail Dealer": "retail_trade",
    "Secondhand Dealer - General": "retail_trade",
    "Secondhand Dealer - Auto": "retail_trade",
    "Pawnbroker": "retail_trade",
    "Dealer In Products For The Disabled": "retail_trade",
    "Newsstand": "retail_trade",
    "General Vendor Distributor": "retail_trade",
    "Stoop Line Stand": "retail_trade",
    "Ticket Seller Business": "retail_trade",
    "Scale Dealer/Repairer": "retail_trade",
    # business services
    "Home Improvement Contractor": "business_services",
    "Electronic & Home Appliance Service Dealer": "business_services",
    "Employment Agency": "business_services",
    "Process Server Individual": "business_services",
    "Process Serving Agency": "business_services",
    "Debt Collection Agency": "business_services",
    "Construction Labor Provider": "business_services",
    "Laundries": "business_services",
    "Industrial Laundry": "business_services",
    "Industrial Laundry Delivery": "business_services",
    "Locksmith": "business_services",
    "Storage Warehouse": "business_services",
    # industrial / misc
    "Scrap Metal Processor": "industrial_misc",
}

# Distance decay parameters in meters.
POI_RADIUS = {
    "retail_trade": 900.0,
    "business_services": 1200.0,
    "hospitality_food": 1200.0,
    "mobility_parking": 1500.0,
    "gaming_entertainment": 1500.0,
    "industrial_misc": 1800.0,
    "other_commercial": 1000.0,
}
POI_BW = {
    "retail_trade": 300.0,
    "business_services": 450.0,
    "hospitality_food": 450.0,
    "mobility_parking": 550.0,
    "gaming_entertainment": 600.0,
    "industrial_misc": 700.0,
    "other_commercial": 350.0,
}

STATIC_ROAD_RADIUS = {
    "motorway": 1200.0,
    "trunk": 1000.0,
    "primary": 800.0,
    "secondary": 700.0,
    "tertiary": 600.0,
    "residential": 350.0,
    "other_road": 500.0,
}
STATIC_ROAD_BW = {
    "motorway": 450.0,
    "trunk": 400.0,
    "primary": 300.0,
    "secondary": 260.0,
    "tertiary": 220.0,
    "residential": 120.0,
    "other_road": 180.0,
}

NODE_RADIUS = defaultdict(lambda: 250.0, {
    "crossing": 160.0,
    "traffic_signals": 180.0,
    "motorway_junction": 500.0,
    "turning_circle": 200.0,
    "stop": 180.0,
})
NODE_BW = defaultdict(lambda: 80.0, {
    "crossing": 60.0,
    "traffic_signals": 70.0,
    "motorway_junction": 150.0,
    "turning_circle": 70.0,
    "stop": 70.0,
})

EVENT_RADIUS = {
    "land_dev": 1800.0,
    "road_repair": 800.0,
    "road_change": 1000.0,
}
EVENT_BW = {
    "land_dev": 700.0,
    "road_repair": 250.0,
    "road_change": 350.0,
}

CLUSTER_LABEL_PRIORITY = [
    "bg_poi_kde_medical_and_health",
    "bg_poi_kde_culture_and_education",
    "bg_poi_kde_public_services",
    "bg_poi_kde_governments_and_organizations",
    "bg_poi_kde_transportation",
    "bg_poi_kde_residential_area",
    "bg_poi_kde_place_of_worship",
    "bg_poi_kde_parking_area",
    "poi_kde_retail_trade",
    "poi_kde_business_services",
    "poi_kde_hospitality_food",
    "poi_kde_mobility_parking",
    "poi_kde_gaming_entertainment",
    "poi_kde_industrial_misc",
    "land_dev_kde",
    "road_change_kde",
    "road_repair_kde",
]

STABLE_BG_CATEGORIES = {
    "medical_and_health",
    "culture_and_education",
    "public_services",
    "governments_and_organizations",
    "place_of_worship",
    "transportation",
    "residential_area",
    "parking_area",
}

STATIC_POI_RADIUS = defaultdict(lambda: 1000.0, {
    "medical_and_health": 2200.0,
    "culture_and_education": 1800.0,
    "public_services": 1500.0,
    "governments_and_organizations": 1500.0,
    "place_of_worship": 1200.0,
    "transportation": 1800.0,
    "residential_area": 600.0,
    "parking_area": 500.0,
    "sports_and_leisure": 1400.0,
    "scenic_spots": 1600.0,
    "finance": 900.0,
    "corporations": 1000.0,
    "shopping": 900.0,
    "catering": 800.0,
    "domestic_services": 800.0,
})

STATIC_POI_BW = defaultdict(lambda: 350.0, {
    "medical_and_health": 700.0,
    "culture_and_education": 550.0,
    "public_services": 450.0,
    "governments_and_organizations": 450.0,
    "place_of_worship": 400.0,
    "transportation": 650.0,
    "residential_area": 180.0,
    "parking_area": 160.0,
    "sports_and_leisure": 450.0,
    "scenic_spots": 550.0,
    "finance": 250.0,
    "corporations": 300.0,
    "shopping": 250.0,
    "catering": 220.0,
    "domestic_services": 220.0,
})


# -----------------------------
# Utility helpers
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, path: Path) -> Path:
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def quarter_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=((ts.month - 1) // 3) * 3 + 1, day=1)


def quarter_label(ts: pd.Timestamp) -> str:
    q = ((ts.month - 1) // 3) + 1
    return f"{ts.year}Q{q}"


def quarterly_periods(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    start_q = quarter_floor(start)
    end_q = quarter_floor(end)
    quarters = []
    current = start_q
    while current <= end_q:
        q_end = (current + pd.offsets.QuarterEnd(0)).normalize()
        quarters.append((current, q_end, quarter_label(current)))
        current = current + pd.offsets.QuarterBegin(startingMonth=1)
    return quarters


def gaussian_kernel(dist: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-0.5 * (dist / bandwidth) ** 2)


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def detect_swapped_lat_lng(df: pd.DataFrame, lat_col: str, lng_col: str) -> bool:
    sample = df[[lat_col, lng_col]].dropna().head(1000)
    if sample.empty:
        return False
    # NYC: latitude ~ 40.x, longitude ~ -73.x
    lat_med = sample[lat_col].median()
    lng_med = sample[lng_col].median()
    return (lat_med < 0) and (lng_med > 0)


def group_business_category(cat: str) -> str:
    if pd.isna(cat):
        return "other_commercial"
    return BUSINESS_GROUP_RULES.get(str(cat), "other_commercial")


def road_group(link_type_name: str) -> str:
    valid = {"motorway", "trunk", "primary", "secondary", "tertiary", "residential"}
    val = str(link_type_name).lower() if pd.notna(link_type_name) else "other_road"
    return val if val in valid else "other_road"


def maybe_set_precision(geom_series: gpd.GeoSeries, grid: float = 0.5) -> gpd.GeoSeries:
    if set_precision is None:
        return geom_series
    return geom_series.apply(lambda g: set_precision(g, grid_size=grid) if g is not None else g)


# -----------------------------
# Data loaders
# -----------------------------
def load_roads(path: Path, max_rows: Optional[int] = None) -> gpd.GeoDataFrame:
    usecols = [
        "link_id", "from_node_id", "to_node_id", "length", "link_type_name",
        "is_link", "geometry"
    ]
    df = pd.read_csv(path, usecols=usecols, nrows=max_rows)
    df = df.dropna(subset=["geometry"]).copy()
    df["geometry"] = df["geometry"].map(wkt.loads)
    df = df.drop_duplicates(subset=["geometry"]).copy()
    # Exclude ramps / connector links by default; keep true street skeleton.
    if "is_link" in df.columns:
        df = df[df["is_link"].fillna(0).astype(int) == 0].copy()
    df["road_group"] = df["link_type_name"].map(road_group)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=WGS84).to_crs(NYC_CRS)
    gdf["geometry"] = maybe_set_precision(gdf.geometry)
    return gdf.reset_index(drop=True)


def load_nodes(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, usecols=["node_id", "osm_highway", "lat", "lng"])
    if detect_swapped_lat_lng(df, "lat", "lng"):
        x, y = df["lat"].copy(), df["lng"].copy()
    else:
        x, y = df["lng"].copy(), df["lat"].copy()
    gdf = gpd.GeoDataFrame(
        df.rename(columns={"osm_highway": "node_type"}),
        geometry=gpd.points_from_xy(x, y),
        crs=WGS84,
    ).to_crs(NYC_CRS)
    return gdf


def load_background_poi(path: Optional[Path], mode: str = "stable") -> Optional[gpd.GeoDataFrame]:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)

    # Common schemas, including uploaded poi_filter.csv: lng, lat, cate
    schemas = [
        ("lng", "lat", "cate"),
        ("longitude", "latitude", "poi_category"),
        ("lng", "lat", "poi_category"),
        ("longitude", "latitude", "category"),
        ("lng", "lat", "category"),
    ]
    matched = None
    for lng_col, lat_col, cat_col in schemas:
        if lng_col in df.columns and lat_col in df.columns and cat_col in df.columns:
            matched = (lng_col, lat_col, cat_col)
            break
    if matched is None:
        raise ValueError(
            "Background POI table schema not recognized. Expected columns like (lng, lat, cate) or (longitude, latitude, category)."
        )

    lng_col, lat_col, cat_col = matched
    df = df[df[lng_col].notna() & df[lat_col].notna()].copy()
    df["bg_category"] = df[cat_col].astype(str).str.strip().str.lower()
    df["bg_category"] = df["bg_category"].replace({
        "governments_and_organization": "governments_and_organizations",
        "government_and_organizations": "governments_and_organizations",
        "public_service": "public_services",
        "scenic_spot": "scenic_spots",
        "domestic_service": "domestic_services",
        "corporation": "corporations",
    })
    if mode == "stable":
        df = df[df["bg_category"].isin(STABLE_BG_CATEGORIES)].copy()
    elif mode != "all":
        raise ValueError("background POI mode must be either 'stable' or 'all'.")
    if df.empty:
        return None
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lng_col], df[lat_col]),
        crs=WGS84,
    ).to_crs(NYC_CRS)
    if "poi_id" in gdf.columns:
        gdf = gdf.drop_duplicates(subset=["poi_id"]).copy()
    else:
        gdf = gdf.drop_duplicates(subset=[lng_col, lat_col, "bg_category"]).copy()
    return gdf.reset_index(drop=True)


def load_poi_events(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = ["start_event_time", "end_event_time", "longitude", "latitude", "business_category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"POI event file missing columns: {missing}")
    df["start_time"] = parse_dates(df["start_event_time"])
    df["end_time"] = parse_dates(df["end_event_time"])
    df["business_group"] = df["business_category"].map(group_business_category)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=WGS84,
    ).to_crs(NYC_CRS)
    return gdf


def load_land_dev(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["longitude", "latitude"]).copy()
    df["start_time"] = parse_dates(df["first_project_start"])
    df["end_time"] = parse_dates(df.get("first_project_end", pd.Series(index=df.index, dtype=object)))
    for fallback in ["final_co_time", "signoff_time", "non_temporary_co_time", "first_tco_time"]:
        if fallback in df.columns:
            df["end_time"] = df["end_time"].fillna(parse_dates(df[fallback]))
    df["end_time"] = df["end_time"].fillna(df["start_time"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=WGS84,
    ).to_crs(NYC_CRS)
    return gdf


def load_road_repair(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["start_time"] = parse_dates(df["start_event_time"])
    df["end_time"] = parse_dates(df["end_event_time"])

    def choose_xy(frame: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        if {"centroid_longitude", "centroid_latitude"}.issubset(frame.columns):
            x = frame["centroid_longitude"]
            y = frame["centroid_latitude"]
            if x.notna().sum() > 0:
                return x.fillna(frame.get("longitude")), y.fillna(frame.get("latitude"))
        if {"longitude", "latitude"}.issubset(frame.columns):
            return frame["longitude"], frame["latitude"]
        raise ValueError("Road repair file lacks usable point coordinates.")

    x, y = choose_xy(df)
    df = df[x.notna() & y.notna()].copy()
    x, y = x.loc[df.index], y.loc[df.index]
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y), crs=WGS84).to_crs(NYC_CRS)
    return gdf


def load_road_change(path: Path) -> gpd.GeoDataFrame:
    import fiona
    layers = fiona.listlayers(path)
    layer = layers[0]
    gdf = gpd.read_file(path, layer=layer)
    if gdf.crs is None:
        gdf = gdf.set_crs(NYC_CRS)
    else:
        gdf = gdf.to_crs(NYC_CRS)
    gdf["start_time"] = parse_dates(gdf.get("event_time_start", pd.Series(index=gdf.index, dtype=object)))
    gdf["end_time"] = parse_dates(gdf.get("event_time_end", pd.Series(index=gdf.index, dtype=object)))
    gdf["end_time"] = gdf["end_time"].fillna(parse_dates(gdf.get("effective_time", pd.Series(index=gdf.index, dtype=object))))
    gdf["start_time"] = gdf["start_time"].fillna(gdf["end_time"])
    gdf["event_group"] = gdf.get("event_type", "road_change").fillna("road_change")
    gdf["geometry"] = gdf.geometry.centroid
    return gdf


# -----------------------------
# Block construction
# -----------------------------
def build_blocks_from_roads(
    roads: gpd.GeoDataFrame,
    min_area_m2: float = 600.0,
    max_area_m2: float = 8_000_000.0,
) -> gpd.GeoDataFrame:
    """Polygonize road centerlines into stable road blocks."""
    merged = unary_union(list(roads.geometry.values))
    polys = list(polygonize(merged))
    blocks = gpd.GeoDataFrame({"geometry": polys}, crs=roads.crs)
    blocks = blocks[~blocks.geometry.is_empty & blocks.geometry.is_valid].copy()
    blocks["block_area_m2"] = blocks.geometry.area
    blocks = blocks[(blocks["block_area_m2"] >= min_area_m2) & (blocks["block_area_m2"] <= max_area_m2)].copy()
    blocks = blocks.reset_index(drop=True)
    blocks["block_id"] = np.arange(1, len(blocks) + 1)
    blocks["block_perimeter_m"] = blocks.geometry.length
    blocks["compactness"] = 4 * math.pi * blocks["block_area_m2"] / (blocks["block_perimeter_m"] ** 2 + 1e-9)
    blocks["block_point"] = blocks.geometry.representative_point()
    return blocks


def build_block_adjacency(blocks: gpd.GeoDataFrame) -> pd.DataFrame:
    joins = gpd.sjoin(
        blocks[["block_id", "geometry"]],
        blocks[["block_id", "geometry"]],
        how="inner",
        predicate="touches",
    )
    joins = joins.loc[joins["block_id_left"] < joins["block_id_right"], ["block_id_left", "block_id_right"]].copy()
    joins.columns = ["src_block_id", "dst_block_id"]
    return joins.reset_index(drop=True)


# -----------------------------
# Feature construction
# -----------------------------
def block_points(blocks: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray]:
    pts = gpd.GeoSeries(blocks["block_point"], crs=blocks.crs)
    xy = np.column_stack([pts.x.values, pts.y.values])
    return blocks["block_id"].values, xy


def add_decay_feature_group(
    out: pd.DataFrame,
    block_ids: np.ndarray,
    block_xy: np.ndarray,
    event_gdf: Optional[gpd.GeoDataFrame],
    category_col: str,
    prefix: str,
    radius_map: Dict[str, float],
    bw_map: Dict[str, float],
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    if event_gdf is None or len(event_gdf) == 0:
        return out

    gdf = event_gdf.copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return out
    gdf[category_col] = gdf[category_col].fillna("unknown")
    geom_xy = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    weights = gdf[weight_col].fillna(1.0).astype(float).values if weight_col and weight_col in gdf.columns else np.ones(len(gdf))
    cats = gdf[category_col].astype(str).values

    for cat in sorted(pd.unique(cats)):
        mask = cats == cat
        if mask.sum() == 0:
            continue
        radius = float(radius_map.get(cat, radius_map.get("default", 1000.0)))
        bw = float(bw_map.get(cat, bw_map.get("default", max(radius / 3.0, 1.0))))
        tree = cKDTree(geom_xy[mask])
        point_weights = weights[mask]
        neigh = tree.query_ball_point(block_xy, r=radius)
        vals = np.zeros(len(block_xy), dtype=float)
        coords_cat = geom_xy[mask]
        for i, idxs in enumerate(neigh):
            if not idxs:
                continue
            local = coords_cat[idxs]
            d = np.sqrt(((local - block_xy[i]) ** 2).sum(axis=1))
            vals[i] = np.sum(gaussian_kernel(d, bw) * point_weights[idxs])
        col = f"{prefix}_{cat}"
        out[col] = vals
    return out


def _keep_linear_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    out = gdf.explode(index_parts=False).copy()
    mask = out.geometry.geom_type.isin(["LineString", "MultiLineString"])
    return out.loc[mask].copy()


def _keep_polygon_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    out = gdf.explode(index_parts=False).copy()
    mask = out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    return out.loc[mask].copy()


def overlay_length_sum(lines: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame({"block_id": polygons["block_id"]})
    inter = gpd.overlay(
        lines[[group_col, "geometry"]],
        polygons[["block_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    inter = _keep_linear_geometries(inter)
    if inter.empty:
        return pd.DataFrame({"block_id": polygons["block_id"]})
    inter["seg_len_m"] = inter.geometry.length
    piv = inter.pivot_table(index="block_id", columns=group_col, values="seg_len_m", aggfunc="sum", fill_value=0.0)
    piv.columns = [f"{prefix}_{c}" for c in piv.columns]
    return piv.reset_index()


def node_count_features(nodes: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame({"block_id": blocks["block_id"]})
    joined = gpd.sjoin(nodes[["node_type", "geometry"]], blocks[["block_id", "geometry"]], predicate="within", how="inner")
    if joined.empty:
        return pd.DataFrame({"block_id": blocks["block_id"]})
    piv = joined.pivot_table(index="block_id", columns="node_type", values="index_right", aggfunc="count", fill_value=0)
    piv.columns = [f"node_count_{c}" for c in piv.columns]
    return piv.reset_index()


def build_static_features(
    blocks: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    bg_poi: Optional[gpd.GeoDataFrame] = None,
) -> pd.DataFrame:
    out = blocks[["block_id", "block_area_m2", "block_perimeter_m", "compactness"]].copy()

    # Road boundary / internal lengths by road type
    road_len = overlay_length_sum(roads[["road_group", "geometry"]].copy(), blocks, "road_group", "road_len_m")
    out = out.merge(road_len, on="block_id", how="left")

    # Node counts within block
    node_counts = node_count_features(nodes, blocks)
    out = out.merge(node_counts, on="block_id", how="left")

    # Distance-decayed road and node influence at block representative points
    block_ids, block_xy = block_points(blocks)
    road_pts = roads.copy()
    road_pts["geometry"] = road_pts.geometry.representative_point()
    out = add_decay_feature_group(out, block_ids, block_xy, road_pts, "road_group", "road_kde", STATIC_ROAD_RADIUS, STATIC_ROAD_BW)
    out = add_decay_feature_group(out, block_ids, block_xy, nodes.copy(), "node_type", "node_kde", NODE_RADIUS, NODE_BW)

    if bg_poi is not None and len(bg_poi) > 0:
        out = add_decay_feature_group(
            out, block_ids, block_xy, bg_poi.copy(),
            "bg_category", "bg_poi_kde", STATIC_POI_RADIUS, STATIC_POI_BW
        )
        bg_join = gpd.sjoin(bg_poi[["bg_category", "geometry"]], blocks[["block_id", "geometry"]], how="inner", predicate="within")
        if not bg_join.empty:
            bg_counts = bg_join.groupby(["block_id", "bg_category"]).size().unstack(fill_value=0)
            bg_counts.columns = [f"bg_poi_count_{c}" for c in bg_counts.columns]
            bg_counts = bg_counts.reset_index()
            out = out.merge(bg_counts, on="block_id", how="left")

    out = out.fillna(0.0)
    return out


def active_overlap(gdf: gpd.GeoDataFrame, q_start: pd.Timestamp, q_end: pd.Timestamp) -> gpd.GeoDataFrame:
    """Rows whose [start_time, end_time] overlaps the quarter interval."""
    if gdf.empty:
        return gdf.copy()
    st = gdf["start_time"].fillna(q_start)
    et = gdf["end_time"].fillna(st)
    mask = (st <= q_end) & (et >= q_start)
    return gdf.loc[mask].copy()


def start_events_in_quarter(gdf: gpd.GeoDataFrame, q_start: pd.Timestamp, q_end: pd.Timestamp) -> gpd.GeoDataFrame:
    """Rows whose start boundary falls within the quarter."""
    if gdf.empty:
        return gdf.copy()
    st = gdf["start_time"]
    mask = st.notna() & (st >= q_start) & (st <= q_end)
    return gdf.loc[mask].copy()


def end_events_in_quarter(gdf: gpd.GeoDataFrame, q_start: pd.Timestamp, q_end: pd.Timestamp) -> gpd.GeoDataFrame:
    """Rows whose end boundary falls within the quarter."""
    if gdf.empty:
        return gdf.copy()
    et = gdf["end_time"]
    mask = et.notna() & (et >= q_start) & (et <= q_end)
    return gdf.loc[mask].copy()


def _merge_event_counts(out: pd.DataFrame, blocks: gpd.GeoDataFrame, gdf: gpd.GeoDataFrame, colname: str) -> pd.DataFrame:
    if gdf is None or len(gdf) == 0:
        out[colname] = 0.0
        return out
    counts = gpd.sjoin(gdf[["geometry"]], blocks[["block_id", "geometry"]], predicate="within", how="inner")
    if counts.empty:
        out[colname] = 0.0
        return out
    cnt = counts.groupby("block_id").size().rename(colname).reset_index()
    out = out.merge(cnt, on="block_id", how="left")
    out[colname] = out[colname].fillna(0.0)
    return out


def _add_single_event_feature(out: pd.DataFrame, blocks: gpd.GeoDataFrame, block_ids: np.ndarray, block_xy: np.ndarray,
                              gdf: gpd.GeoDataFrame, event_name: str, prefix: str) -> pd.DataFrame:
    """Add one scalar count + KDE feature for an event subset."""
    count_col = f"{prefix}_count"
    kde_col = f"{prefix}_kde"
    if gdf is None or len(gdf) == 0:
        out[count_col] = 0.0
        out[kde_col] = 0.0
        return out
    tmp = gdf.copy()
    tmp["event_group"] = event_name
    raw_prefix = f"__tmp_{prefix}"
    out = add_decay_feature_group(
        out,
        block_ids,
        block_xy,
        tmp,
        "event_group",
        raw_prefix,
        {event_name: EVENT_RADIUS[event_name]},
        {event_name: EVENT_BW[event_name]},
    )
    raw_kde_col = f"{raw_prefix}_{event_name}"
    out[kde_col] = out.get(raw_kde_col, 0.0)
    if raw_kde_col in out.columns:
        out = out.drop(columns=[raw_kde_col])
    out = _merge_event_counts(out, blocks, tmp, count_col)
    return out


def build_dynamic_features_for_quarter(
    blocks: gpd.GeoDataFrame,
    poi_events: gpd.GeoDataFrame,
    land_dev: gpd.GeoDataFrame,
    road_repair: gpd.GeoDataFrame,
    road_change: gpd.GeoDataFrame,
    q_start: pd.Timestamp,
    q_end: pd.Timestamp,
    static_features: pd.DataFrame,
) -> pd.DataFrame:
    block_ids, block_xy = block_points(blocks)
    out = static_features.copy()
    out["quarter_start"] = q_start
    out["quarter_end"] = q_end
    out["quarter"] = quarter_label(q_start)

    # Commercial POI intervals + start/end boundaries
    active_poi = active_overlap(poi_events, q_start, q_end)
    started_poi = start_events_in_quarter(poi_events, q_start, q_end)
    ended_poi = end_events_in_quarter(poi_events, q_start, q_end)

    out = add_decay_feature_group(out, block_ids, block_xy, active_poi, "business_group", "poi_kde", POI_RADIUS, POI_BW)
    out = add_decay_feature_group(out, block_ids, block_xy, started_poi, "business_group", "poi_open_kde", POI_RADIUS, POI_BW)
    out = add_decay_feature_group(out, block_ids, block_xy, ended_poi, "business_group", "poi_close_kde", POI_RADIUS, POI_BW)
    out = _merge_event_counts(out, blocks, active_poi, "poi_active_count")
    out = _merge_event_counts(out, blocks, started_poi, "poi_open_count")
    out = _merge_event_counts(out, blocks, ended_poi, "poi_close_count")

    # All non-POI events are now handled consistently as interval-overlap + start/end boundaries
    for name, base_gdf in [("land_dev", land_dev), ("road_repair", road_repair), ("road_change", road_change)]:
        active_gdf = active_overlap(base_gdf, q_start, q_end)
        started_gdf = start_events_in_quarter(base_gdf, q_start, q_end)
        ended_gdf = end_events_in_quarter(base_gdf, q_start, q_end)

        out = _add_single_event_feature(out, blocks, block_ids, block_xy, active_gdf, name, f"{name}")
        out = _add_single_event_feature(out, blocks, block_ids, block_xy, started_gdf, name, f"{name}_start")
        out = _add_single_event_feature(out, blocks, block_ids, block_xy, ended_gdf, name, f"{name}_end")

    # Net commercial growth signal
    open_cols = [c for c in out.columns if c.startswith("poi_open_kde_")]
    close_cols = [c for c in out.columns if c.startswith("poi_close_kde_")]
    out["poi_open_total"] = out[open_cols].sum(axis=1) if open_cols else 0.0
    out["poi_close_total"] = out[close_cols].sum(axis=1) if close_cols else 0.0
    out["poi_net_growth"] = out["poi_open_total"] - out["poi_close_total"]

    out = out.fillna(0.0)
    return out


# -----------------------------
# State recognition and zoning
# -----------------------------
def cluster_quarter_states(
    quarter_df: pd.DataFrame,
    n_clusters: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    id_cols = ["block_id", "quarter", "quarter_start", "quarter_end"]
    feature_cols = [c for c in quarter_df.columns if c not in id_cols and np.issubdtype(quarter_df[c].dtype, np.number)]
    x = quarter_df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imp = imputer.fit_transform(x)
    x_std = scaler.fit_transform(x_imp)
    k = min(n_clusters, max(2, len(quarter_df) // 20 if len(quarter_df) >= 40 else len(quarter_df)))
    if k < 2:
        labels = np.zeros(len(quarter_df), dtype=int)
    else:
        model = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=4096, n_init="auto")
        labels = model.fit_predict(x_std)
    out = quarter_df[["block_id", "quarter", "quarter_start", "quarter_end"]].copy()
    out["state_code"] = labels

    # Human-readable names by dominant features in cluster centers.
    cluster_means = quarter_df.assign(state_code=labels).groupby("state_code").mean(numeric_only=True)
    name_map = {}
    for cid, row in cluster_means.iterrows():
        dominant = None
        best = -np.inf
        for col in CLUSTER_LABEL_PRIORITY:
            if col in row.index and row[col] > best:
                best = row[col]
                dominant = col
        if dominant is None or best <= 0:
            label = "low_activity"
        elif dominant.startswith("land_dev"):
            label = "development_transition"
        elif dominant.startswith("road_change"):
            label = "traffic_reconfigured"
        elif dominant.startswith("road_repair"):
            label = "traffic_disturbed"
        elif dominant.startswith("poi_kde_"):
            label = dominant.replace("poi_kde_", "")
        elif dominant.startswith("bg_poi_kde_"):
            label = dominant.replace("bg_poi_kde_", "")
        else:
            label = dominant
        # Growth-aware suffix
        if dominant and dominant.startswith("poi_kde_") and "poi_net_growth" in row.index and row["poi_net_growth"] > 0.5 * max(1e-6, row.get(dominant, 1.0)):
            label = f"{label}_growth"
        name_map[cid] = label
    out["state_name"] = out["state_code"].map(name_map)
    return out


def build_zone_components(
    block_states: pd.DataFrame,
    adjacency: pd.DataFrame,
) -> pd.DataFrame:
    """Connected components of adjacent blocks sharing the same state."""
    g = nx.Graph()
    for _, row in block_states.iterrows():
        g.add_node(int(row["block_id"]), state=row["state_code"], state_name=row["state_name"])
    state_lookup = block_states.set_index("block_id")["state_code"].to_dict()
    for _, row in adjacency.iterrows():
        a, b = int(row["src_block_id"]), int(row["dst_block_id"])
        if state_lookup.get(a) == state_lookup.get(b):
            g.add_edge(a, b)
    components = []
    for zone_idx, comp in enumerate(nx.connected_components(g), start=1):
        members = sorted(comp)
        state_code = state_lookup[members[0]]
        state_name = g.nodes[members[0]]["state_name"]
        for bid in members:
            components.append({
                "block_id": bid,
                "zone_local_id": zone_idx,
                "state_code": state_code,
                "state_name": state_name,
            })
    return pd.DataFrame(components)


def dissolve_quarter_zones(
    blocks: gpd.GeoDataFrame,
    zone_membership: pd.DataFrame,
    quarter: str,
) -> gpd.GeoDataFrame:
    merged = blocks[["block_id", "geometry"]].merge(zone_membership, on="block_id", how="inner")
    merged["zone_id"] = merged["zone_local_id"].map(lambda x: f"{quarter}_FZ_{int(x):05d}")
    gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=blocks.crs)
    zones = gdf.dissolve(by="zone_id", as_index=False, aggfunc={"state_code": "first", "state_name": "first", "zone_local_id": "first"})
    zones["quarter"] = quarter
    zones["zone_area_m2"] = zones.geometry.area
    return zones[["zone_id", "quarter", "state_code", "state_name", "zone_local_id", "zone_area_m2", "geometry"]]


def zone_transitions(prev_zones: gpd.GeoDataFrame, curr_zones: gpd.GeoDataFrame, overlap_threshold: float = 0.15) -> pd.DataFrame:
    if prev_zones.empty or curr_zones.empty:
        return pd.DataFrame(columns=["prev_zone_id", "curr_zone_id", "overlap_prev", "overlap_curr", "transition_type"])
    ov = gpd.overlay(
        prev_zones[["zone_id", "geometry"]],
        curr_zones[["zone_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    ov = _keep_polygon_geometries(ov)
    if ov.empty:
        return pd.DataFrame(columns=["prev_zone_id", "curr_zone_id", "overlap_prev", "overlap_curr", "transition_type"])
    ov["overlap_area"] = ov.geometry.area
    prev_area = prev_zones.set_index("zone_id")["zone_area_m2"].to_dict()
    curr_area = curr_zones.set_index("zone_id")["zone_area_m2"].to_dict()
    ov = ov.rename(columns={"zone_id_1": "prev_zone_id", "zone_id_2": "curr_zone_id"})
    ov["overlap_prev"] = ov.apply(lambda r: r["overlap_area"] / max(prev_area.get(r["prev_zone_id"], 1.0), 1.0), axis=1)
    ov["overlap_curr"] = ov.apply(lambda r: r["overlap_area"] / max(curr_area.get(r["curr_zone_id"], 1.0), 1.0), axis=1)
    ov = ov[(ov["overlap_prev"] >= overlap_threshold) | (ov["overlap_curr"] >= overlap_threshold)].copy()
    if ov.empty:
        return pd.DataFrame(columns=["prev_zone_id", "curr_zone_id", "overlap_prev", "overlap_curr", "transition_type"])
    counts_prev = ov.groupby("prev_zone_id")["curr_zone_id"].nunique().to_dict()
    counts_curr = ov.groupby("curr_zone_id")["prev_zone_id"].nunique().to_dict()
    def trans(row):
        p, c = counts_prev[row["prev_zone_id"]], counts_curr[row["curr_zone_id"]]
        if p == 1 and c == 1:
            return "stable_or_transform"
        if p > 1 and c == 1:
            return "merge"
        if p == 1 and c > 1:
            return "split"
        return "complex"
    ov["transition_type"] = ov.apply(trans, axis=1)
    return ov[["prev_zone_id", "curr_zone_id", "overlap_prev", "overlap_curr", "transition_type"]].drop_duplicates()


# -----------------------------
# Main pipeline
# -----------------------------
@dataclass
class PipelineInputs:
    roads_csv: Path
    nodes_csv: Path
    poi_events_csv: Path
    land_dev_csv: Path
    road_repair_csv: Path
    road_change_gpkg: Path
    output_dir: Path
    background_poi_csv: Optional[Path] = None
    background_poi_mode: str = "stable"
    start_quarter: Optional[str] = None
    end_quarter: Optional[str] = None
    n_clusters: int = 8
    max_roads: Optional[int] = None


def parse_quarter_arg(q: Optional[str]) -> Optional[pd.Timestamp]:
    if not q:
        return None
    year = int(q[:4])
    quarter = int(q[-1])
    month = (quarter - 1) * 3 + 1
    return pd.Timestamp(year=year, month=month, day=1)


def run_pipeline(inp: PipelineInputs) -> None:
    ensure_dir(inp.output_dir)
    ensure_dir(inp.output_dir / "zones")

    print("[1/8] Loading data ...")
    roads = load_roads(inp.roads_csv, max_rows=inp.max_roads)
    nodes = load_nodes(inp.nodes_csv)
    poi_events = load_poi_events(inp.poi_events_csv)
    land_dev = load_land_dev(inp.land_dev_csv)
    road_repair = load_road_repair(inp.road_repair_csv)
    road_change = load_road_change(inp.road_change_gpkg)
    bg_poi = load_background_poi(inp.background_poi_csv, mode=inp.background_poi_mode)

    print("[2/8] Building road blocks ...")
    blocks = build_blocks_from_roads(roads)
    if len(blocks) == 0:
        raise RuntimeError("No valid blocks were polygonized from the selected roads. Try using the full road network or increase --max-roads.")
    adjacency = build_block_adjacency(blocks)
    gpd.GeoDataFrame(blocks.drop(columns=["block_point"]), geometry="geometry", crs=blocks.crs).to_file(
        inp.output_dir / "blocks.gpkg", layer="blocks", driver="GPKG"
    )
    adjacency.to_csv(inp.output_dir / "block_adjacency.csv", index=False)

    print("[3/8] Building static block features ...")
    static_features = build_static_features(blocks, roads, nodes, bg_poi)
    write_table(static_features, inp.output_dir / "block_static_features.parquet")

    print("[4/8] Determining quarterly timeline ...")
    min_ts = min(
        s for s in [
            poi_events["start_time"].min(),
            land_dev["start_time"].min(),
            road_repair["start_time"].min(),
            road_change["start_time"].min(),
        ] if pd.notna(s)
    )
    max_ts = max(
        s for s in [
            poi_events["end_time"].max(),
            land_dev["end_time"].max(),
            road_repair["end_time"].max(),
            road_change["end_time"].max(),
        ] if pd.notna(s)
    )
    q_start_override = parse_quarter_arg(inp.start_quarter)
    q_end_override = parse_quarter_arg(inp.end_quarter)
    if q_start_override is not None:
        min_ts = q_start_override
    if q_end_override is not None:
        max_ts = q_end_override
    quarters = quarterly_periods(min_ts, max_ts)

    print(f"[5/8] Quarterly feature construction for {len(quarters)} quarters ...")
    all_block_features: List[pd.DataFrame] = []
    all_states: List[pd.DataFrame] = []
    prev_zones = None
    all_transitions: List[pd.DataFrame] = []

    for i, (q_start, q_end, q_label) in enumerate(quarters, start=1):
        print(f"  - quarter {i}/{len(quarters)}: {q_label}")
        q_feat = build_dynamic_features_for_quarter(
            blocks=blocks,
            poi_events=poi_events,
            land_dev=land_dev,
            road_repair=road_repair,
            road_change=road_change,
            q_start=q_start,
            q_end=q_end,
            static_features=static_features,
        )
        all_block_features.append(q_feat)

        q_states = cluster_quarter_states(q_feat, n_clusters=inp.n_clusters)
        all_states.append(q_states)

        zone_membership = build_zone_components(q_states, adjacency)
        q_zones = dissolve_quarter_zones(blocks, zone_membership, q_label)
        q_zones.to_file(inp.output_dir / "zones" / f"functional_zones_{q_label}.gpkg", layer=q_label, driver="GPKG")
        zone_membership.to_csv(inp.output_dir / "zones" / f"block_zone_membership_{q_label}.csv", index=False)

        if prev_zones is not None:
            trans = zone_transitions(prev_zones, q_zones)
            trans["prev_quarter"] = prev_zones["quarter"].iloc[0]
            trans["curr_quarter"] = q_label
            all_transitions.append(trans)
        prev_zones = q_zones

    print("[6/8] Writing quarterly block features / states ...")
    block_features = pd.concat(all_block_features, ignore_index=True)
    block_states = pd.concat(all_states, ignore_index=True)
    write_table(block_features, inp.output_dir / "block_features_quarterly.parquet")
    write_table(block_states, inp.output_dir / "block_states_quarterly.parquet")

    print("[7/8] Writing zone transitions ...")
    transitions = pd.concat(all_transitions, ignore_index=True) if all_transitions else pd.DataFrame()
    write_table(transitions, inp.output_dir / "functional_zone_transitions.parquet")

    print("[8/8] Done.")
    print(f"Outputs written to: {inp.output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build quarterly functional zones for NYC.")
    p.add_argument("--roads-csv", type=Path, required=True)
    p.add_argument("--nodes-csv", type=Path, required=True)
    p.add_argument("--poi-events-csv", type=Path, required=True)
    p.add_argument("--land-dev-csv", type=Path, required=True)
    p.add_argument("--road-repair-csv", type=Path, required=True)
    p.add_argument("--road-change-gpkg", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--background-poi-csv", type=Path, default=None, help="Optional static background POI table. Supports poi_filter.csv.")
    p.add_argument("--background-poi-mode", type=str, default="stable", choices=["stable", "all"], help="Use only relatively stable POI categories or all background POIs.")
    p.add_argument("--start-quarter", type=str, default=None, help="Example: 2019Q1")
    p.add_argument("--end-quarter", type=str, default=None, help="Example: 2024Q4")
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--max-roads", type=int, default=None, help="Debug option to limit roads when testing.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(PipelineInputs(
        roads_csv=args.roads_csv,
        nodes_csv=args.nodes_csv,
        poi_events_csv=args.poi_events_csv,
        land_dev_csv=args.land_dev_csv,
        road_repair_csv=args.road_repair_csv,
        road_change_gpkg=args.road_change_gpkg,
        output_dir=args.output_dir,
        background_poi_csv=args.background_poi_csv,
        background_poi_mode=args.background_poi_mode,
        start_quarter=args.start_quarter,
        end_quarter=args.end_quarter,
        n_clusters=args.n_clusters,
        max_roads=args.max_roads,
    ))


if __name__ == "__main__":
    main()
