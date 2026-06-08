#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an explicit interval Temporal UrbanKG dataset for NYC using user-defined relations.

Output format
-------------
Raw quadruples:    head, relation, tail, start_time, end_time
ID quadruples:     head_id, relation_id, tail_id, start_time, end_time

Time semantics
--------------
1) Static structural/background facts use empty time interval: start_time="", end_time="".
2) Dynamic event facts use their real validity intervals.
3) Quarterly FunctionalZone facts use [quarter_start, quarter_end].
4) Commercial POI facts use [start_event_time, end_event_time] when available.

Relations used (exact abbreviations)
------------------------------------
PLA, RLA, JLA, PLB, RLB, JLB, ALB, JBR, BNB, ANA, PLR,
PHPC, RHRC, JHJC, FHPC, PLF, FLA, BBF, FNF,
RCIR, RRIR, RCIB, RPIB, LDIB, RCIF, RPIF, LDIF
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point

WGS84 = "EPSG:4326"
NYC_CRS = "EPSG:2263"

STATIC_BG_CATEGORIES = {
    "medical_and_health",
    "culture_and_education",
    "public_services",
    "governments_and_organizations",
    "place_of_worship",
    "transportation",
    "residential_area",
    "parking_area",
}

RELATIONS = [
    "PLA", "RLA", "JLA", "PLB", "RLB", "JLB", "ALB", "JBR", "BNB", "ANA", "PLR",
    "PHPC", "RHRC", "JHJC", "FHPC", "PLF", "FLA", "BBF", "FNF",
    "RCIR", "RRIR", "RCIB", "RPIB", "LDIB", "RCIF", "RPIF", "LDIF",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build explicit interval TKG quads with user-defined relations.")
    p.add_argument("--fz-output-dir", required=True)
    p.add_argument("--poi-filter-csv", required=True)
    p.add_argument("--poi-events-csv", required=True)
    p.add_argument("--land-dev-csv", required=True)
    p.add_argument("--road-repair-csv", required=True)
    p.add_argument("--road-change-gpkg", required=True)
    p.add_argument("--road-filter-csv", required=True)
    p.add_argument("--node-filter-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--background-poi-mode", choices=["stable", "all"], default="stable")
    p.add_argument("--area-file", default=None, help="Optional area polygon file with area_id and borough_id.")
    p.add_argument("--borough-file", default=None, help="Optional borough polygon file with borough_id.")
    p.add_argument("--area-id-col", default=None, help="Optional explicit area id column name. If omitted, auto-detects NYC fields such as LocationID.")
    p.add_argument("--borough-id-col", default=None, help="Optional explicit borough id column name. If omitted, auto-detects NYC fields such as BoroCode.")
    p.add_argument("--borough-name-col", default=None, help="Optional explicit borough name column name. If omitted, auto-detects NYC fields such as BoroName.")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quarters", nargs="*", default=None)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_dates(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def quarter_to_bounds(q: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    m = re.fullmatch(r"(\d{4})Q([1-4])", q)
    if not m:
        raise ValueError(f"Bad quarter label: {q}")
    year = int(m.group(1))
    qq = int(m.group(2))
    start_month = (qq - 1) * 3 + 1
    start = pd.Timestamp(year=year, month=start_month, day=1)
    end = (start + pd.offsets.QuarterEnd()).normalize()
    return start, end


def extract_quarter_from_zone_filename(path: Path) -> str:
    m = re.match(r"functional_zones_(\d{4}Q[1-4])\.gpkg$", path.name)
    if not m:
        raise ValueError(f"Unexpected zone file name: {path.name}")
    return m.group(1)


def load_auto_table(base_dir: Path, stem: str) -> pd.DataFrame:
    pq = base_dir / f"{stem}.parquet"
    csv = base_dir / f"{stem}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Could not find {stem}.parquet or {stem}.csv under {base_dir}")


def detect_swapped_lat_lng(df: pd.DataFrame, lat_col: str, lng_col: str) -> bool:
    sample = df[[lat_col, lng_col]].dropna().head(1000)
    if sample.empty:
        return False
    lat_med = sample[lat_col].median()
    lng_med = sample[lng_col].median()
    return (lat_med < 0) and (lng_med > 0)


def geometry_from_lonlat(df: pd.DataFrame, lon_col: str, lat_col: str) -> gpd.GeoDataFrame:
    tmp = df.copy()
    if detect_swapped_lat_lng(tmp, lat_col, lon_col):
        tmp[[lat_col, lon_col]] = tmp[[lon_col, lat_col]]
    tmp = tmp[pd.notna(tmp[lon_col]) & pd.notna(tmp[lat_col])].copy()
    geom = gpd.points_from_xy(tmp[lon_col], tmp[lat_col], crs=WGS84)
    return gpd.GeoDataFrame(tmp, geometry=geom, crs=WGS84)


def normalize_text(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s and s.lower() != "nan" else None


def make_entity(kind: str, raw_id: str) -> str:
    return f"{kind}::{raw_id}"


def format_ts(x) -> str:
    if x is None or pd.isna(x) or str(x) == "":
        return ""
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def add_quad(store: List[Tuple[str, str, str, str, str]], h, r, t, start=None, end=None):
    if h is None or t is None or r is None:
        return
    store.append((str(h), str(r), str(t), format_ts(start), format_ts(end)))


def interval_overlap(start: pd.Series, end: pd.Series, q_start: pd.Timestamp, q_end: pd.Timestamp) -> pd.Series:
    s = pd.to_datetime(start, errors="coerce")
    e = pd.to_datetime(end, errors="coerce")
    e = e.fillna(s)
    return (s <= q_end) & (e >= q_start)


def overlap_bounds(start, end, q_start, q_end):
    s = pd.to_datetime(start, errors="coerce")
    e = pd.to_datetime(end, errors="coerce")
    if pd.isna(s):
        return None, None
    if pd.isna(e):
        e = s
    os = max(s, q_start)
    oe = min(e, q_end)
    if os > oe:
        return None, None
    return os, oe


def load_linestring_csv(path: Path, geometry_col: str = "geometry") -> gpd.GeoDataFrame:
    df = pd.read_csv(path, low_memory=False)
    geoms = []
    for val in df[geometry_col]:
        geom = None
        if pd.notna(val):
            try:
                geom = wkt.loads(val)
            except Exception:
                geom = None
        geoms.append(geom)
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs=NYC_CRS)
    if not gdf.empty:
        bounds = gdf.total_bounds
        if bounds[0] > -180 and bounds[2] < 180 and bounds[1] > -90 and bounds[3] < 90:
            gdf = gdf.set_crs(WGS84, allow_override=True).to_crs(NYC_CRS)
    return gdf[gdf.geometry.notna()].copy()


def load_roads(path: Path) -> gpd.GeoDataFrame:
    usecols = ["link_id", "from_node_id", "to_node_id", "link_type_name", "geometry"]
    gdf = load_linestring_csv(path)[usecols].copy()
    gdf["road_e"] = gdf["link_id"].map(lambda x: make_entity("road", str(int(x))))
    gdf["road_cat_e"] = gdf["link_type_name"].fillna("other_road").astype(str).str.lower().map(lambda x: make_entity("road_category", x))
    gdf["from_j_e"] = gdf["from_node_id"].map(lambda x: make_entity("junction", str(int(x))) if pd.notna(x) else None)
    gdf["to_j_e"] = gdf["to_node_id"].map(lambda x: make_entity("junction", str(int(x))) if pd.notna(x) else None)
    return gdf


def load_nodes(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, usecols=["node_id", "osm_highway", "lng", "lat"], low_memory=False)
    gdf = geometry_from_lonlat(df, "lng", "lat").to_crs(NYC_CRS)
    gdf["junction_e"] = gdf["node_id"].map(lambda x: make_entity("junction", str(int(x))))
    gdf["junction_cat_e"] = gdf["osm_highway"].fillna("other_junction").astype(str).str.lower().map(lambda x: make_entity("junction_category", x))
    return gdf


def load_static_poi(path: Path, mode: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, usecols=["poi_id", "lng", "lat", "cate"], low_memory=False)
    if mode == "stable":
        df = df[df["cate"].isin(STATIC_BG_CATEGORIES)].copy()
    gdf = geometry_from_lonlat(df, "lng", "lat").to_crs(NYC_CRS)
    gdf["point_e"] = gdf["poi_id"].map(lambda x: make_entity("point", f"static_{int(x)}"))
    gdf["cat_e"] = gdf["cate"].astype(str).map(lambda x: make_entity("poi_category", x))
    return gdf


def load_commercial_poi(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, low_memory=False)
    gdf = geometry_from_lonlat(df, "longitude", "latitude").to_crs(NYC_CRS)
    raw_key = gdf["license_nbr"] if "license_nbr" in gdf.columns else pd.Series(np.arange(len(gdf)), index=gdf.index).map(lambda x: f"row_{x}")
    raw_key = raw_key.fillna(pd.Series(np.arange(len(gdf)), index=gdf.index).map(lambda x: f"row_{x}"))
    gdf["point_key"] = raw_key.astype(str)
    gdf["point_e"] = gdf["point_key"].map(lambda x: make_entity("point", f"commercial_{x}"))
    gdf["cat_name"] = gdf["business_category"].fillna("other_commercial").astype(str)
    gdf["cat_e"] = gdf["cat_name"].map(lambda x: make_entity("poi_category", x))
    gdf["start_time"] = parse_dates(gdf["start_event_time"])
    gdf["end_time"] = parse_dates(gdf["end_event_time"])
    return gdf


def load_land_dev(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[pd.notna(df.get("longitude")) & pd.notna(df.get("latitude"))].copy()
    if df.empty:
        return gpd.GeoDataFrame(df.copy(), geometry=[], crs=NYC_CRS)
    df["start_time"] = parse_dates(df.get("first_project_start"))
    end = pd.Series(pd.NaT, index=df.index)
    for c in ["first_project_end", "final_co_time", "non_temporary_co_time", "signoff_time", "first_tco_time"]:
        if c in df.columns:
            cand = parse_dates(df[c])
            end = end.where(end.notna(), cand)
    df["end_time"] = end
    key = df.get("parcel_key")
    if key is None:
        key = pd.Series(np.arange(len(df)), index=df.index).map(lambda x: f"row_{x}")
    df["event_key"] = key.fillna(pd.Series(np.arange(len(df)), index=df.index).map(lambda x: f"row_{x}")).astype(str)
    gdf = geometry_from_lonlat(df, "longitude", "latitude").to_crs(NYC_CRS)
    gdf["event_e"] = gdf["event_key"].map(lambda x: make_entity("land_dev_event", x))
    return gdf


def load_road_repair(path: Path) -> gpd.GeoDataFrame:
    usecols = [
        "event_id", "start_event_time", "end_event_time", "wkt",
        "centroid_longitude", "centroid_latitude", "longitude", "latitude",
        "borough", "borough_code", "area"
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    geoms = []
    for _, row in df.iterrows():
        geom = None
        w = row.get("wkt")
        if pd.notna(w):
            try:
                geom = wkt.loads(w)
            except Exception:
                geom = None
        if geom is None:
            lon = row.get("centroid_longitude", row.get("longitude"))
            lat = row.get("centroid_latitude", row.get("latitude"))
            if pd.notna(lon) and pd.notna(lat):
                geom = Point(float(lon), float(lat))
        geoms.append(geom)
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs=NYC_CRS)
    if not gdf.empty:
        bounds = gdf.total_bounds
        if bounds[0] > -180 and bounds[2] < 180 and bounds[1] > -90 and bounds[3] < 90:
            gdf = gdf.set_crs(WGS84, allow_override=True).to_crs(NYC_CRS)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["start_time"] = parse_dates(gdf["start_event_time"])
    gdf["end_time"] = parse_dates(gdf["end_event_time"])
    gdf["event_key"] = gdf["event_id"].fillna(pd.Series(np.arange(len(gdf)), index=gdf.index).map(lambda x: f"row_{x}")).astype(str)
    gdf["event_e"] = gdf["event_key"].map(lambda x: make_entity("road_repair_event", x))
    return gdf


def load_road_change(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(NYC_CRS)
    else:
        gdf = gdf.to_crs(NYC_CRS)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["start_time"] = parse_dates(gdf["event_time_start"])
    gdf["end_time"] = parse_dates(gdf["event_time_end"]).fillna(parse_dates(gdf["effective_time"]))
    gdf["event_key"] = gdf["event_id"].fillna(pd.Series(np.arange(len(gdf)), index=gdf.index).map(lambda x: f"row_{x}")).astype(str)
    gdf["event_e"] = gdf["event_key"].map(lambda x: make_entity("road_change_event", x))
    return gdf


def load_blocks_and_zones(fz_dir: Path, quarter_subset: Optional[Sequence[str]] = None):
    blocks = gpd.read_file(fz_dir / "blocks.gpkg", layer="blocks")
    if blocks.crs is None:
        blocks = blocks.set_crs(NYC_CRS)
    else:
        blocks = blocks.to_crs(NYC_CRS)
    blocks["block_e"] = blocks["block_id"].map(lambda x: make_entity("block", str(int(x))))
    adjacency = pd.read_csv(fz_dir / "block_adjacency.csv")
    quarter_zones = {}
    zone_dir = fz_dir / "zones"
    for gpkg in sorted(zone_dir.glob("functional_zones_*.gpkg")):
        q = extract_quarter_from_zone_filename(gpkg)
        if quarter_subset and q not in quarter_subset:
            continue
        zones = gpd.read_file(gpkg, layer=q)
        if zones.crs is None:
            zones = zones.set_crs(NYC_CRS)
        else:
            zones = zones.to_crs(NYC_CRS)
        q_start, q_end = quarter_to_bounds(q)
        zones["zone_e"] = zones["zone_id"].map(lambda z: make_entity("functional_zone", str(z)))
        zones["cat_e"] = zones["state_name"].astype(str).map(lambda s: make_entity("functional_zone_category", s))
        zones["q_start"] = q_start
        zones["q_end"] = q_end
        mem = pd.read_csv(zone_dir / f"block_zone_membership_{q}.csv")
        mem["zone_id"] = mem["zone_local_id"].map(lambda x: f"{q}_FZ_{int(x):05d}")
        mem["zone_e"] = mem["zone_id"].map(lambda z: make_entity("functional_zone", str(z)))
        mem["block_e"] = mem["block_id"].map(lambda b: make_entity("block", str(int(b))))
        mem["q_start"] = q_start
        mem["q_end"] = q_end
        quarter_zones[q] = (zones, mem)
    return blocks, adjacency, quarter_zones


def load_area_borough_files(area_file: Optional[str], borough_file: Optional[str], area_id_col: Optional[str], borough_id_col: Optional[str], borough_name_col: Optional[str]):
    areas = boroughs = None
    borough_name_to_id: Dict[str, str] = {}
    if borough_file:
        boroughs = gpd.read_file(borough_file)
        if boroughs.crs is None:
            boroughs = boroughs.set_crs(WGS84)
        if boroughs.crs != NYC_CRS:
            boroughs = boroughs.to_crs(NYC_CRS)
        b_id_col = borough_id_col or guess_borough_id_col(boroughs)
        b_name_col = borough_name_col or guess_borough_name_col(boroughs)
        if b_id_col is None:
            raise ValueError("Could not detect borough id column from borough boundary file. Please pass --borough-id-col.")
        boroughs = boroughs.copy()
        boroughs["borough_id"] = boroughs[b_id_col].map(normalize_code)
        boroughs["borough_name"] = boroughs[b_name_col].map(normalize_name) if b_name_col else None
        boroughs = boroughs[boroughs["borough_id"].notna()].copy()
        boroughs["borough_e"] = boroughs["borough_id"].map(lambda x: make_entity("borough", x))
        if "borough_name" in boroughs.columns:
            borough_name_to_id = {normalize_name(n): i for n, i in boroughs[["borough_name", "borough_id"]].dropna().drop_duplicates().itertuples(index=False)}
    if area_file:
        areas = gpd.read_file(area_file)
        if areas.crs is None:
            areas = areas.set_crs(WGS84)
        if areas.crs != NYC_CRS:
            areas = areas.to_crs(NYC_CRS)
        a_id_col = area_id_col or guess_area_id_col(areas)
        if a_id_col is None:
            raise ValueError("Could not detect area id column from area boundary file. Please pass --area-id-col.")
        a_name_col = guess_area_name_col(areas)
        a_boro_name_col = guess_area_borough_name_col(areas)
        areas = areas.copy()
        areas["area_id"] = areas[a_id_col].map(normalize_code)
        if a_name_col:
            areas["area_name"] = areas[a_name_col].astype(str)
        if a_boro_name_col:
            areas["borough_name_from_area"] = areas[a_boro_name_col].map(normalize_name)
        # derive borough_id from explicit column if present, otherwise via borough name join
        b_id_from_area_col = borough_id_col if (borough_id_col and borough_id_col in areas.columns) else first_existing_col(areas, ["borough_id", "BoroCode", "borocode"])
        if b_id_from_area_col and b_id_from_area_col in areas.columns:
            areas["borough_id"] = areas[b_id_from_area_col].map(normalize_code)
        elif "borough_name_from_area" in areas.columns and borough_name_to_id:
            areas["borough_id"] = areas["borough_name_from_area"].map(borough_name_to_id)
        else:
            areas["borough_id"] = None
        areas = areas[areas["area_id"].notna()].copy()
        areas["area_e"] = areas["area_id"].map(lambda x: make_entity("area", x))
        areas["borough_e"] = areas["borough_id"].map(lambda x: make_entity("borough", x) if x is not None else None)
    return areas, boroughs


def representative_point_join(gdf: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, polygon_cols: Sequence[str], predicate: str = "within") -> pd.DataFrame:
    if gdf.empty or polygons is None or polygons.empty:
        return pd.DataFrame(columns=list(gdf.columns) + list(polygon_cols))
    tmp = gdf.copy()
    tmp["geometry"] = tmp.geometry.representative_point()
    joined = gpd.sjoin(tmp, polygons[list(polygon_cols) + ["geometry"]], how="inner", predicate=predicate)
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    return joined


def nearest_road_join(points: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.DataFrame:
    if points.empty or roads.empty:
        return pd.DataFrame(columns=list(points.columns) + ["road_e"])
    base = roads[["road_e", "geometry"]].copy()
    joined = gpd.sjoin_nearest(points, base, how="inner", distance_col="_dist_to_road")
    return joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])


def build_area_borough_lookup(commercial_poi, land_dev, road_change):
    pairs = []
    for df in [commercial_poi, land_dev, road_change]:
        cols = set(df.columns)
        if {"area_id", "borough_id"}.issubset(cols):
            sub = df[["area_id", "borough_id"]].dropna().drop_duplicates()
            pairs.append(sub)
    if not pairs:
        return pd.DataFrame(columns=["area_id", "borough_id", "area_e", "borough_e"])
    pair_df = pd.concat(pairs, ignore_index=True).drop_duplicates()
    pair_df["area_e"] = pair_df["area_id"].map(lambda x: make_entity("area", str(int(x)) if str(x).isdigit() else str(x)))
    pair_df["borough_e"] = pair_df["borough_id"].map(lambda x: make_entity("borough", str(int(x)) if str(x).isdigit() else str(x)))
    return pair_df


def normalize_code(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        if float(s).is_integer():
            return str(int(float(s)))
    except Exception:
        pass
    return s

def normalize_name(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    s = re.sub(r"\s+", " ", s)
    return s


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def guess_area_id_col(df: gpd.GeoDataFrame) -> Optional[str]:
    return first_existing_col(df, ["area_id", "LocationID", "locationid", "OBJECTID", "objectid", "zone_id", "id"])


def guess_area_name_col(df: gpd.GeoDataFrame) -> Optional[str]:
    return first_existing_col(df, ["area_name", "zone", "name", "Name"])


def guess_area_borough_name_col(df: gpd.GeoDataFrame) -> Optional[str]:
    return first_existing_col(df, ["borough", "borough_name", "BoroName", "boro_name"])


def guess_borough_id_col(df: gpd.GeoDataFrame) -> Optional[str]:
    return first_existing_col(df, ["borough_id", "BoroCode", "borocode", "code", "id", "OBJECTID"])


def guess_borough_name_col(df: gpd.GeoDataFrame) -> Optional[str]:
    return first_existing_col(df, ["borough_name", "BoroName", "name", "borough"])


def add_static_point_relations(quads, static_poi, roads, areas, boroughs):
    if static_poi.empty:
        return
    # PHPC
    for _, row in static_poi[["point_e", "cat_e"]].drop_duplicates().iterrows():
        add_quad(quads, row["point_e"], "PHPC", row["cat_e"], "", "")
    # PLR nearest road
    sp_road = nearest_road_join(static_poi[["point_e", "geometry"]], roads)
    for _, row in sp_road[["point_e", "road_e"]].drop_duplicates().iterrows():
        add_quad(quads, row["point_e"], "PLR", row["road_e"], "", "")
    # PLA/PLB if polygons available
    if areas is not None:
        sp_area = representative_point_join(static_poi[["point_e", "geometry"]], areas[["area_e", "borough_e", "geometry"]], ["area_e", "borough_e"])
        for _, row in sp_area.drop_duplicates(subset=["point_e", "area_e"]).iterrows():
            add_quad(quads, row["point_e"], "PLA", row["area_e"], "", "")
        for _, row in sp_area.drop_duplicates(subset=["point_e", "borough_e"]).iterrows():
            add_quad(quads, row["point_e"], "PLB", row["borough_e"], "", "")
    elif boroughs is not None:
        sp_b = representative_point_join(static_poi[["point_e", "geometry"]], boroughs[["borough_e", "geometry"]], ["borough_e"])
        for _, row in sp_b.drop_duplicates(subset=["point_e", "borough_e"]).iterrows():
            add_quad(quads, row["point_e"], "PLB", row["borough_e"], "", "")


def add_commercial_point_relations(quads, commercial_poi, roads, areas, borough_lookup, quarter_zones, boroughs=None):
    if commercial_poi.empty:
        return
    # category + road
    base_cols = [c for c in ["point_e", "geometry", "start_time", "end_time", "cat_e"] if c in commercial_poi.columns]
    cp_road = nearest_road_join(commercial_poi[base_cols], roads)
    for _, row in cp_road.iterrows():
        s = row["start_time"]
        e = row["end_time"] if pd.notna(row["end_time"]) else None
        add_quad(quads, row["point_e"], "PHPC", row["cat_e"], s, e)
        add_quad(quads, row["point_e"], "PLR", row["road_e"], s, e)
    # area/borough from polygons first, fallback to lookup cols if unavailable
    if areas is not None and not areas.empty:
        cp_area = representative_point_join(commercial_poi[["point_e", "geometry", "start_time", "end_time"]], areas[["area_e", "borough_e", "geometry"]], ["area_e", "borough_e"])
        for _, row in cp_area.drop_duplicates(subset=["point_e", "area_e", "borough_e"]).iterrows():
            s = row["start_time"]
            e = row["end_time"] if pd.notna(row["end_time"]) else None
            if pd.notna(row.get("area_e")):
                add_quad(quads, row["point_e"], "PLA", row["area_e"], s, e)
            if pd.notna(row.get("borough_e")):
                add_quad(quads, row["point_e"], "PLB", row["borough_e"], s, e)
    elif boroughs is not None and not boroughs.empty:
        cp_b = representative_point_join(commercial_poi[["point_e", "geometry", "start_time", "end_time"]], boroughs[["borough_e", "geometry"]], ["borough_e"])
        for _, row in cp_b.drop_duplicates(subset=["point_e", "borough_e"]).iterrows():
            s = row["start_time"]
            e = row["end_time"] if pd.notna(row["end_time"]) else None
            add_quad(quads, row["point_e"], "PLB", row["borough_e"], s, e)
    else:
        for _, row in commercial_poi.iterrows():
            s = row["start_time"]
            e = row["end_time"] if pd.notna(row["end_time"]) else None
            if pd.notna(row.get("area_id")):
                add_quad(quads, row["point_e"], "PLA", make_entity("area", normalize_code(row["area_id"])), s, e)
            if pd.notna(row.get("borough_id")):
                add_quad(quads, row["point_e"], "PLB", make_entity("borough", normalize_code(row["borough_id"])), s, e)
    # PLF quarter overlap to FZs
    for q, (zones, _) in quarter_zones.items():
        q_start, q_end = quarter_to_bounds(q)
        mask = interval_overlap(commercial_poi["start_time"], commercial_poi["end_time"], q_start, q_end)
        sub = commercial_poi.loc[mask, ["point_e", "start_time", "end_time", "geometry"]].copy()
        if sub.empty:
            continue
        joined = representative_point_join(sub, zones[["zone_e", "geometry"]], ["zone_e"])
        for _, row in joined.iterrows():
            s, e = overlap_bounds(row["start_time"], row["end_time"], q_start, q_end)
            add_quad(quads, row["point_e"], "PLF", row["zone_e"], s, e)


def add_static_road_junction_relations(quads, roads, nodes, areas, boroughs, area_lookup):
    # RHRC + JHJC + JBR
    for _, row in roads[["road_e", "road_cat_e", "from_j_e", "to_j_e"]].drop_duplicates().iterrows():
        add_quad(quads, row["road_e"], "RHRC", row["road_cat_e"], "", "")
        if row["from_j_e"] is not None:
            add_quad(quads, row["from_j_e"], "JBR", row["road_e"], "", "")
        if row["to_j_e"] is not None:
            add_quad(quads, row["to_j_e"], "JBR", row["road_e"], "", "")
    for _, row in nodes[["junction_e", "junction_cat_e"]].drop_duplicates().iterrows():
        add_quad(quads, row["junction_e"], "JHJC", row["junction_cat_e"], "", "")
    # RLA/RLB/JLA/JLB if polygons exist
    if areas is not None:
        r_area = representative_point_join(roads[["road_e", "geometry"]], areas[["area_e", "borough_e", "geometry"]], ["area_e", "borough_e"], predicate="intersects")
        for _, row in r_area.drop_duplicates(subset=["road_e", "area_e"]).iterrows():
            add_quad(quads, row["road_e"], "RLA", row["area_e"], "", "")
        for _, row in r_area.drop_duplicates(subset=["road_e", "borough_e"]).iterrows():
            add_quad(quads, row["road_e"], "RLB", row["borough_e"], "", "")
        j_area = representative_point_join(nodes[["junction_e", "geometry"]], areas[["area_e", "borough_e", "geometry"]], ["area_e", "borough_e"])
        for _, row in j_area.drop_duplicates(subset=["junction_e", "area_e"]).iterrows():
            add_quad(quads, row["junction_e"], "JLA", row["area_e"], "", "")
        for _, row in j_area.drop_duplicates(subset=["junction_e", "borough_e"]).iterrows():
            add_quad(quads, row["junction_e"], "JLB", row["borough_e"], "", "")
    elif boroughs is not None:
        r_b = representative_point_join(roads[["road_e", "geometry"]], boroughs[["borough_e", "geometry"]], ["borough_e"], predicate="intersects")
        for _, row in r_b.drop_duplicates(subset=["road_e", "borough_e"]).iterrows():
            add_quad(quads, row["road_e"], "RLB", row["borough_e"], "", "")
        j_b = representative_point_join(nodes[["junction_e", "geometry"]], boroughs[["borough_e", "geometry"]], ["borough_e"])
        for _, row in j_b.drop_duplicates(subset=["junction_e", "borough_e"]).iterrows():
            add_quad(quads, row["junction_e"], "JLB", row["borough_e"], "", "")


def add_area_borough_relations(quads, areas, boroughs, area_lookup):
    # ALB
    if areas is not None and {"area_e", "borough_e"}.issubset(areas.columns):
        for _, row in areas[["area_e", "borough_e"]].drop_duplicates().iterrows():
            add_quad(quads, row["area_e"], "ALB", row["borough_e"], "", "")
    else:
        for _, row in area_lookup[["area_e", "borough_e"]].drop_duplicates().iterrows():
            add_quad(quads, row["area_e"], "ALB", row["borough_e"], "", "")
    # adjacency if polygons available
    if boroughs is not None and not boroughs.empty:
        ov = gpd.overlay(boroughs[["borough_e", "geometry"]], boroughs[["borough_e", "geometry"]], how="intersection", keep_geom_type=False)
        if not ov.empty:
            ov = ov[ov["borough_e_1"] != ov["borough_e_2"]]
            for _, row in ov[["borough_e_1", "borough_e_2"]].drop_duplicates().iterrows():
                add_quad(quads, row["borough_e_1"], "BNB", row["borough_e_2"], "", "")
    if areas is not None and not areas.empty:
        ov = gpd.overlay(areas[["area_e", "geometry"]], areas[["area_e", "geometry"]], how="intersection", keep_geom_type=False)
        if not ov.empty:
            ov = ov[ov["area_e_1"] != ov["area_e_2"]]
            for _, row in ov[["area_e_1", "area_e_2"]].drop_duplicates().iterrows():
                add_quad(quads, row["area_e_1"], "ANA", row["area_e_2"], "", "")


def majority_area_for_zone(zone_geom, area_polys) -> Optional[str]:
    if area_polys is None or area_polys.empty:
        return None
    inter = gpd.overlay(gpd.GeoDataFrame({"_id": [1]}, geometry=[zone_geom], crs=area_polys.crs), area_polys[["area_e", "geometry"]], how="intersection", keep_geom_type=False)
    if inter.empty:
        return None
    inter["_a"] = inter.geometry.area
    return inter.sort_values("_a", ascending=False)["area_e"].iloc[0]


def add_functional_zone_relations(quads, quarter_zones, blocks, areas):
    # FLA, FHPC, BBF, FNF
    for q, (zones, mem) in quarter_zones.items():
        q_start, q_end = quarter_to_bounds(q)
        for _, row in zones[["zone_e", "cat_e", "geometry"]].iterrows():
            add_quad(quads, row["zone_e"], "FHPC", row["cat_e"], q_start, q_end)
            area_e = majority_area_for_zone(row["geometry"], areas) if areas is not None else None
            if area_e is not None:
                add_quad(quads, row["zone_e"], "FLA", area_e, q_start, q_end)
        for _, row in mem[["block_e", "zone_e"]].drop_duplicates().iterrows():
            add_quad(quads, row["block_e"], "BBF", row["zone_e"], q_start, q_end)
        if not zones.empty:
            ov = gpd.overlay(zones[["zone_e", "geometry"]], zones[["zone_e", "geometry"]], how="intersection", keep_geom_type=False)
            if not ov.empty:
                ov = ov[ov["zone_e_1"] != ov["zone_e_2"]]
                for _, row in ov[["zone_e_1", "zone_e_2"]].drop_duplicates().iterrows():
                    add_quad(quads, row["zone_e_1"], "FNF", row["zone_e_2"], q_start, q_end)


def add_event_relations(quads, roads, quarter_zones, road_change, road_repair, land_dev, boroughs=None):
    # event -> road / borough / FZ
    road_base = roads[["road_e", "geometry"]].copy()

    def event_to_road(events: gpd.GeoDataFrame, rel: str):
        if events.empty:
            return
        joined = gpd.sjoin(events[["event_e", "start_time", "end_time", "geometry"]], road_base, how="inner", predicate="intersects")
        if joined.empty:
            joined = gpd.sjoin_nearest(events[["event_e", "start_time", "end_time", "geometry"]], road_base, how="inner", distance_col="_dist")
        joined = joined.drop(columns=[c for c in ["index_right", "_dist"] if c in joined.columns])
        for _, row in joined.drop_duplicates(subset=["event_e", "road_e"]).iterrows():
            add_quad(quads, row["event_e"], rel, row["road_e"], row["start_time"], row["end_time"])

    event_to_road(road_change, "RCIR")
    event_to_road(road_repair, "RRIR")

    def add_borough_from_cols(events: pd.DataFrame, event_col: str, rel: str, borough_col_candidates: Sequence[str]):
        if events.empty:
            return
        for _, row in events.iterrows():
            bvals = []
            for c in borough_col_candidates:
                if c not in events.columns:
                    continue
                val = row.get(c)
                if pd.isna(val):
                    continue
                if isinstance(val, str) and c.endswith("json"):
                    try:
                        arr = json.loads(val)
                        if isinstance(arr, list):
                            bvals.extend([normalize_code(v) for v in arr if normalize_code(v) is not None])
                    except Exception:
                        pass
                else:
                    norm = normalize_code(val)
                    if norm is not None:
                        bvals.append(norm)
            for b in sorted(set([x for x in bvals if x is not None])):
                add_quad(quads, row[event_col], rel, make_entity("borough", b), row["start_time"], row["end_time"])

    # Borough from polygons first if available
    if boroughs is not None and not boroughs.empty:
        borough_polys = boroughs[["borough_e", "geometry"]].copy()
        for events, rel in [(road_change, "RCIB"), (road_repair, "RPIB"), (land_dev, "LDIB")]:
            if events.empty:
                continue
            joined = gpd.sjoin(events[["event_e", "start_time", "end_time", "geometry"]], borough_polys, how="inner", predicate="intersects")
            if joined.empty:
                joined = gpd.sjoin_nearest(events[["event_e", "start_time", "end_time", "geometry"]], borough_polys, how="inner", distance_col="_dist_b")
            joined = joined.drop(columns=[c for c in ["index_right", "_dist_b"] if c in joined.columns])
            for _, row in joined.drop_duplicates(subset=["event_e", "borough_e"]).iterrows():
                add_quad(quads, row["event_e"], rel, row["borough_e"], row["start_time"], row["end_time"])
    else:
        add_borough_from_cols(road_change, "event_e", "RCIB", ["borough_id", "borough_id_list_json"])
        add_borough_from_cols(land_dev, "event_e", "LDIB", ["borough_id_spatial", "borough_id"])
        if not road_repair.empty:
            rr = road_repair.copy()
            rr["borough_code_norm"] = rr.get("borough_code", pd.Series([None]*len(rr))).map(normalize_code)
            for _, row in rr.iterrows():
                b = row.get("borough_code_norm") or normalize_code(row.get("borough"))
                if b is not None:
                    add_quad(quads, row["event_e"], "RPIB", make_entity("borough", b), row["start_time"], row["end_time"])

    # event -> FZ with overlap intervals
    for q, (zones, _) in quarter_zones.items():
        q_start, q_end = quarter_to_bounds(q)
        zone_polys = zones[["zone_e", "geometry"]].copy()
        for gdf, rel in [(road_change, "RCIF"), (road_repair, "RPIF"), (land_dev, "LDIF")]:
            if gdf.empty:
                continue
            mask = interval_overlap(gdf["start_time"], gdf["end_time"], q_start, q_end)
            sub = gdf.loc[mask, ["event_e", "start_time", "end_time", "geometry"]].copy()
            if sub.empty:
                continue
            joined = gpd.sjoin(sub, zone_polys, how="inner", predicate="intersects")
            joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
            for _, row in joined.drop_duplicates(subset=["event_e", "zone_e"]).iterrows():
                s, e = overlap_bounds(row["start_time"], row["end_time"], q_start, q_end)
                add_quad(quads, row["event_e"], rel, row["zone_e"], s, e)


def build_all_quads(args: argparse.Namespace):
    fz_dir = Path(args.fz_output_dir)
    roads = load_roads(Path(args.road_filter_csv))
    nodes = load_nodes(Path(args.node_filter_csv))
    static_poi = load_static_poi(Path(args.poi_filter_csv), args.background_poi_mode)
    commercial_poi = load_commercial_poi(Path(args.poi_events_csv))
    land_dev = load_land_dev(Path(args.land_dev_csv))
    road_repair = load_road_repair(Path(args.road_repair_csv))
    road_change = load_road_change(Path(args.road_change_gpkg))
    blocks, adjacency, quarter_zones = load_blocks_and_zones(fz_dir, args.quarters)
    areas, boroughs = load_area_borough_files(args.area_file, args.borough_file, args.area_id_col, args.borough_id_col, args.borough_name_col)
    area_lookup = build_area_borough_lookup(commercial_poi, land_dev, road_change)

    quads: List[Tuple[str, str, str, str, str]] = []
    add_static_point_relations(quads, static_poi, roads, areas, boroughs)
    add_commercial_point_relations(quads, commercial_poi, roads, areas, area_lookup, quarter_zones, boroughs)
    add_static_road_junction_relations(quads, roads, nodes, areas, boroughs, area_lookup)
    add_area_borough_relations(quads, areas, boroughs, area_lookup)
    add_functional_zone_relations(quads, quarter_zones, blocks, areas)
    add_event_relations(quads, roads, quarter_zones, road_change, road_repair, land_dev, boroughs)

    # de-duplicate
    qdf = pd.DataFrame(quads, columns=["head", "relation", "tail", "start_time", "end_time"]).drop_duplicates().reset_index(drop=True)
    return qdf


def encode_and_split(qdf: pd.DataFrame, output_dir: Path, val_ratio: float, test_ratio: float, seed: int):
    entities = pd.Index(pd.unique(pd.concat([qdf["head"], qdf["tail"]], ignore_index=True))).sort_values()
    relations = pd.Index(sorted(qdf["relation"].dropna().unique().tolist()))
    ent2id = pd.DataFrame({"entity": entities, "entity_id": np.arange(len(entities), dtype=np.int64)})
    rel2id = pd.DataFrame({"relation": relations, "relation_id": np.arange(len(relations), dtype=np.int64)})
    ent_map = dict(zip(ent2id["entity"], ent2id["entity_id"]))
    rel_map = dict(zip(rel2id["relation"], rel2id["relation_id"]))
    qid = qdf.copy()
    qid["head_id"] = qid["head"].map(ent_map)
    qid["relation_id"] = qid["relation"].map(rel_map)
    qid["tail_id"] = qid["tail"].map(ent_map)
    qid = qid[["head_id", "relation_id", "tail_id", "start_time", "end_time"]]

    rng = np.random.default_rng(seed)
    idx = np.arange(len(qdf))
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test+n_val]
    train_idx = idx[n_test+n_val:]

    def dump_split(name: str, sel: np.ndarray):
        raw = qdf.iloc[sel].reset_index(drop=True)
        iid = qid.iloc[sel].reset_index(drop=True)
        raw.to_csv(output_dir / f"{name}_raw.txt", sep="\t", index=False, header=False)
        iid.to_csv(output_dir / f"{name}_id.txt", sep="\t", index=False, header=False)

    ent2id.to_csv(output_dir / "entity2id.csv", index=False)
    rel2id.to_csv(output_dir / "relation2id.csv", index=False)
    qdf.to_csv(output_dir / "quadruples_raw.csv", index=False)
    qid.to_csv(output_dir / "quadruples_id.csv", index=False)
    dump_split("train", train_idx)
    dump_split("valid", val_idx)
    dump_split("test", test_idx)

    stats = {
        "num_quadruples": int(len(qdf)),
        "num_entities": int(len(ent2id)),
        "num_relations": int(len(rel2id)),
        "num_train": int(len(train_idx)),
        "num_valid": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "static_no_time_quadruples": int(((qdf["start_time"] == "") & (qdf["end_time"] == "")).sum()),
        "temporal_quadruples": int(((qdf["start_time"] != "") | (qdf["end_time"] != "")).sum()),
        "relations": relations.tolist(),
    }
    with open(output_dir / "kg_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    qdf = build_all_quads(args)
    encode_and_split(qdf, out_dir, args.val_ratio, args.test_ratio, args.seed)
    print(f"Saved explicit interval TKG dataset to: {out_dir}")
    print(f"Quadruples: {len(qdf):,}")


if __name__ == "__main__":
    main()
