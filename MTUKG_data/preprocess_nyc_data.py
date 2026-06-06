from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


TASKS = ("static", "poi-events", "land-dev", "road-repair", "road-change")
WGS84 = "EPSG:4326"

gpd = None
np = None
pd = None
wkt = None


def ensure_geo_deps() -> None:
    global gpd, np, pd, wkt
    if gpd is not None:
        return
    try:
        import geopandas as _gpd
        import numpy as _np
        import pandas as _pd
        from shapely import wkt as _wkt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This script needs pandas, geopandas, shapely, and a GeoPandas I/O backend "
            "(fiona or pyogrio). Run it in the same GIS Python environment used by the "
            "original NYC preprocessing scripts."
        ) from exc
    gpd = _gpd
    np = _np
    pd = _pd
    wkt = _wkt


def log(message: str) -> None:
    print(message, flush=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def choose_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def clean_id(value):
    if pd.isna(value):
        return pd.NA
    try:
        value_float = float(value)
        if value_float.is_integer():
            return int(value_float)
    except Exception:
        pass
    return value


def unique_clean(values: Iterable, as_string: bool = False) -> list:
    out = []
    seen = set()
    for value in values:
        if pd.isna(value):
            continue
        value = str(value) if as_string else clean_id(value)
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    try:
        return sorted(out)
    except Exception:
        return out


def load_borough_area(
    borough_shp: Path,
    area_shp: Path,
    target_crs=WGS84,
    exclude_area_ids: list[int] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    require_path(borough_shp, "borough shapefile")
    require_path(area_shp, "area shapefile")
    borough = gpd.read_file(borough_shp).to_crs(target_crs)
    area = gpd.read_file(area_shp).to_crs(target_crs)

    borough = borough[["BoroCode", "BoroName", "geometry"]].copy()
    area = area[["OBJECTID", "zone", "geometry"]].copy()

    if exclude_area_ids:
        area = area[~area["OBJECTID"].isin(exclude_area_ids)].copy()
    return borough, area


def spatial_lookup(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    rename: dict[str, str],
    predicate: str,
    row_col: str = "_row_id",
) -> pd.DataFrame:
    if left.empty:
        return pd.DataFrame(columns=list(rename.values())).rename_axis(row_col)
    left_geom_col = left.geometry.name
    left_for_join = left[[row_col, left_geom_col]].copy()
    if left_geom_col != "geometry":
        left_for_join = left_for_join.set_geometry(left_geom_col)
    joined = gpd.sjoin(
        left_for_join,
        right[list(rename.keys()) + ["geometry"]].rename(columns=rename),
        how="left",
        predicate=predicate,
    )
    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined = joined.sort_values(row_col).drop_duplicates(row_col, keep="first")
    return joined.set_index(row_col)[list(rename.values())]


def point_borough_area_lookup(
    points: gpd.GeoDataFrame,
    borough: gpd.GeoDataFrame,
    area: gpd.GeoDataFrame,
    row_col: str = "_row_id",
) -> pd.DataFrame:
    b_within = spatial_lookup(
        points,
        borough,
        {"BoroCode": "borough_id", "BoroName": "borough_name"},
        "within",
        row_col,
    )
    b_intersects = spatial_lookup(
        points,
        borough,
        {"BoroCode": "borough_id", "BoroName": "borough_name"},
        "intersects",
        row_col,
    )
    a_within = spatial_lookup(
        points,
        area,
        {"OBJECTID": "area_id", "zone": "area_name"},
        "within",
        row_col,
    )
    a_intersects = spatial_lookup(
        points,
        area,
        {"OBJECTID": "area_id", "zone": "area_name"},
        "intersects",
        row_col,
    )
    result = b_within.combine_first(b_intersects)
    result = result.join(a_within.combine_first(a_intersects), how="outer")
    return result


def assign_point_events(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    borough: gpd.GeoDataFrame,
    area: gpd.GeoDataFrame,
    preserve_no_coord: bool,
) -> pd.DataFrame:
    df = df.copy()
    df["_row_id"] = np.arange(len(df))
    has_coord = df[lon_col].notna() & df[lat_col].notna()

    matched_parts = []
    if has_coord.any():
        gdf = gpd.GeoDataFrame(
            df.loc[has_coord].copy(),
            geometry=gpd.points_from_xy(df.loc[has_coord, lon_col], df.loc[has_coord, lat_col]),
            crs=WGS84,
        )
        lookup = point_borough_area_lookup(gdf, borough, area)
        matched = pd.DataFrame(gdf.drop(columns="geometry")).set_index("_row_id")
        matched = matched.join(lookup, how="left").reset_index()
        matched_parts.append(matched)

    if preserve_no_coord and (~has_coord).any():
        no_coord = df.loc[~has_coord].copy()
        for col in ["borough_id", "borough_name", "area_id", "area_name"]:
            if col not in no_coord.columns:
                no_coord[col] = pd.NA
        matched_parts.append(no_coord)

    if not matched_parts:
        return df.iloc[0:0].drop(columns=["_row_id"], errors="ignore")

    out = pd.concat(matched_parts, ignore_index=True)
    out = out.sort_values("_row_id").reset_index(drop=True)
    return out.drop(columns=["_row_id"], errors="ignore")


def process_static(args: argparse.Namespace) -> None:
    log("[static] Loading borough and area")
    borough, area = load_borough_area(
        args.borough_shp,
        args.area_shp,
        WGS84,
        args.exclude_area_ids,
    )
    ensure_parent(args.static_borough_out)
    borough.to_csv(args.static_borough_out, index=False, encoding="utf-8-sig")
    area.to_csv(args.static_area_out, index=False, encoding="utf-8-sig")

    log("[static] Processing POI")
    poi = pd.read_csv(args.static_poi_csv, low_memory=False)
    lon_col, lat_col = ("lng", "lat") if {"lng", "lat"}.issubset(poi.columns) else ("longitude", "latitude")
    poi = poi.dropna(subset=[lon_col, lat_col]).copy()
    poi = poi.drop(columns=[c for c in ["borough_id", "area_id", "borough_name", "area_name"] if c in poi.columns])
    poi_done = assign_point_events(poi, lon_col, lat_col, borough, area, preserve_no_coord=False)
    poi_done = poi_done.dropna(subset=["borough_id", "area_id"])
    ensure_parent(args.static_poi_out)
    poi_done.to_csv(args.static_poi_out, index=False, encoding="utf-8-sig")
    log(f"  wrote {args.static_poi_out} ({len(poi_done):,} rows)")

    log("[static] Processing road")
    road = pd.read_csv(args.static_road_csv, low_memory=False).copy()
    road = road.drop(columns=[c for c in ["borough_id", "area_id", "borough_name", "area_name"] if c in road.columns])
    road["_row_id"] = np.arange(len(road))
    road_geom = road["geometry"].map(lambda value: wkt.loads(str(value)) if pd.notna(value) else None)
    road_gdf = gpd.GeoDataFrame(
        road.copy(),
        geometry=gpd.GeoSeries(road_geom, index=road.index, crs=WGS84, name="_shape"),
        crs=WGS84,
    )
    road_gdf = road_gdf.loc[road_gdf.geometry.notna()].copy()
    lookup_b = spatial_lookup(
        road_gdf,
        borough,
        {"BoroCode": "borough_id", "BoroName": "borough_name"},
        "intersects",
    )
    lookup_a = spatial_lookup(
        road_gdf,
        area,
        {"OBJECTID": "area_id", "zone": "area_name"},
        "intersects",
    )
    road_done = pd.DataFrame(road_gdf.drop(columns="_shape")).set_index("_row_id")
    road_done = road_done.join(lookup_b, how="left").join(lookup_a, how="left").reset_index(drop=True)
    road_done = road_done.dropna(subset=["borough_id", "area_id"])
    ensure_parent(args.static_road_out)
    road_done.to_csv(args.static_road_out, index=False, encoding="utf-8-sig")
    log(f"  wrote {args.static_road_out} ({len(road_done):,} rows)")

    log("[static] Processing junction")
    junction = pd.read_csv(args.static_node_csv, low_memory=False)
    lon_col, lat_col = ("lng", "lat") if {"lng", "lat"}.issubset(junction.columns) else ("longitude", "latitude")
    junction = junction.dropna(subset=[lon_col, lat_col]).copy()
    junction = junction.drop(columns=[c for c in ["borough_id", "area_id", "borough_name", "area_name"] if c in junction.columns])
    junction_done = assign_point_events(junction, lon_col, lat_col, borough, area, preserve_no_coord=False)
    junction_done = junction_done.dropna(subset=["borough_id", "area_id"])
    ensure_parent(args.static_junction_out)
    junction_done.to_csv(args.static_junction_out, index=False, encoding="utf-8-sig")
    log(f"  wrote {args.static_junction_out} ({len(junction_done):,} rows)")


def process_poi_events(args: argparse.Namespace) -> None:
    log("[poi-events] Loading input")
    borough, area = load_borough_area(args.borough_shp, args.area_shp, WGS84, args.exclude_area_ids)
    poi = pd.read_csv(args.poi_events_csv, low_memory=False)
    if {"longitude", "latitude"}.issubset(poi.columns):
        lon_col, lat_col = "longitude", "latitude"
    elif {"lng", "lat"}.issubset(poi.columns):
        lon_col, lat_col = "lng", "lat"
    else:
        raise ValueError("POI events need longitude/latitude or lng/lat columns.")
    poi = poi.drop(
        columns=[c for c in ["borough_id", "borough_name", "area_id", "area_name", "index_right"] if c in poi.columns],
        errors="ignore",
    )
    out = assign_point_events(poi.dropna(subset=[lon_col, lat_col]).copy(), lon_col, lat_col, borough, area, False)
    ensure_parent(args.poi_events_out)
    out.to_csv(args.poi_events_out, index=False, encoding="utf-8-sig")
    log(f"  wrote {args.poi_events_out} ({len(out):,} rows)")
    log(f"  matched borough_id: {out['borough_id'].notna().sum():,}")
    log(f"  matched area_id: {out['area_id'].notna().sum():,}")


def process_land_dev(args: argparse.Namespace) -> None:
    log("[land-dev] Loading input")
    borough, area = load_borough_area(args.borough_shp, args.area_shp, WGS84, args.exclude_area_ids)
    land = pd.read_csv(args.land_dev_csv, low_memory=False)
    land = land.drop(
        columns=[
            c
            for c in [
                "borough_id",
                "borough_name",
                "area_id",
                "area_name",
                "borough_id_spatial",
                "borough_match_check",
                "index_right",
            ]
            if c in land.columns
        ],
        errors="ignore",
    ).copy()
    land["borough_id"] = land["borough_code"] if "borough_code" in land.columns else pd.NA
    land["_row_id"] = np.arange(len(land))

    has_coord = land["longitude"].notna() & land["latitude"].notna()
    parts = []
    if has_coord.any():
        gdf = gpd.GeoDataFrame(
            land.loc[has_coord].copy(),
            geometry=gpd.points_from_xy(land.loc[has_coord, "longitude"], land.loc[has_coord, "latitude"]),
            crs=WGS84,
        )
        area_lookup = spatial_lookup(gdf, area, {"OBJECTID": "area_id", "zone": "area_name"}, "within")
        area_lookup = area_lookup.combine_first(
            spatial_lookup(gdf, area, {"OBJECTID": "area_id", "zone": "area_name"}, "intersects")
        )
        borough_lookup = spatial_lookup(
            gdf,
            borough,
            {"BoroCode": "borough_id_spatial", "BoroName": "borough_name"},
            "within",
        )
        borough_lookup = borough_lookup.combine_first(
            spatial_lookup(gdf, borough, {"BoroCode": "borough_id_spatial", "BoroName": "borough_name"}, "intersects")
        )
        matched = pd.DataFrame(gdf.drop(columns="geometry")).set_index("_row_id")
        matched = matched.join(area_lookup, how="left").join(borough_lookup, how="left").reset_index()
        matched["borough_id"] = matched["borough_id"].fillna(matched["borough_id_spatial"])
        matched["borough_match_check"] = (
            matched["borough_id"].astype("string") == matched["borough_id_spatial"].astype("string")
        )
        parts.append(matched)

    if (~has_coord).any():
        no_coord = land.loc[~has_coord].copy()
        no_coord["area_id"] = pd.NA
        no_coord["area_name"] = pd.NA
        no_coord["borough_name"] = pd.NA
        no_coord["borough_id_spatial"] = pd.NA
        no_coord["borough_match_check"] = pd.NA
        parts.append(no_coord)

    out = pd.concat(parts, ignore_index=True).sort_values("_row_id").reset_index(drop=True)
    out = out.drop(columns=["_row_id"], errors="ignore")
    ensure_parent(args.land_dev_out)
    out.to_csv(args.land_dev_out, index=False, encoding="utf-8-sig")
    log(f"  wrote {args.land_dev_out} ({len(out):,} rows)")
    log(f"  matched borough_id: {out['borough_id'].notna().sum():,}")
    log(f"  matched area_id: {out['area_id'].notna().sum():,}")


def choose_road_repair_alignment(road: pd.DataFrame) -> pd.DataFrame:
    road = road.copy()
    geom_type = road["geometry_type"].astype("string").str.strip().str.lower().fillna("")
    is_point = geom_type.eq("point")
    road["align_longitude"] = np.nan
    road["align_latitude"] = np.nan
    road["align_coord_source"] = pd.NA

    if {"longitude", "latitude"}.issubset(road.columns):
        road.loc[is_point, "align_longitude"] = road.loc[is_point, "longitude"]
        road.loc[is_point, "align_latitude"] = road.loc[is_point, "latitude"]
        road.loc[is_point, "align_coord_source"] = "point_lonlat"

    if {"start_longitude", "start_latitude"}.issubset(road.columns):
        point_missing = is_point & (road["align_longitude"].isna() | road["align_latitude"].isna())
        road.loc[point_missing, "align_longitude"] = road.loc[point_missing, "start_longitude"]
        road.loc[point_missing, "align_latitude"] = road.loc[point_missing, "start_latitude"]
        road.loc[point_missing, "align_coord_source"] = "point_start_coord"

    if {"centroid_longitude", "centroid_latitude"}.issubset(road.columns):
        road.loc[~is_point, "align_longitude"] = road.loc[~is_point, "centroid_longitude"]
        road.loc[~is_point, "align_latitude"] = road.loc[~is_point, "centroid_latitude"]
        road.loc[~is_point, "align_coord_source"] = "nonpoint_centroid_coord"

    if {"longitude", "latitude"}.issubset(road.columns):
        nonpoint_missing = (~is_point) & (road["align_longitude"].isna() | road["align_latitude"].isna())
        road.loc[nonpoint_missing, "align_longitude"] = road.loc[nonpoint_missing, "longitude"]
        road.loc[nonpoint_missing, "align_latitude"] = road.loc[nonpoint_missing, "latitude"]
        road.loc[nonpoint_missing, "align_coord_source"] = "nonpoint_fallback_lonlat"

    road.loc[road["align_longitude"].isna() | road["align_latitude"].isna(), "align_coord_source"] = "missing"
    return road


def process_road_repair(args: argparse.Namespace) -> None:
    log("[road-repair] Loading input")
    borough, area = load_borough_area(args.borough_shp, args.area_shp, WGS84, args.exclude_area_ids)
    road = pd.read_csv(args.road_repair_csv, low_memory=False)
    road = road.drop(
        columns=[
            c
            for c in [
                "borough_id",
                "borough_name",
                "area_id",
                "area_name",
                "align_longitude",
                "align_latitude",
                "align_coord_source",
                "index_right",
            ]
            if c in road.columns
        ],
        errors="ignore",
    )
    road = choose_road_repair_alignment(road)
    out = assign_point_events(road, "align_longitude", "align_latitude", borough, area, True)
    ensure_parent(args.road_repair_out)
    out.to_csv(args.road_repair_out, index=False, encoding="utf-8-sig")
    log(f"  wrote {args.road_repair_out} ({len(out):,} rows)")
    log(f"  matched borough_id: {out['borough_id'].notna().sum():,}")
    log(f"  matched area_id: {out['area_id'].notna().sum():,}")


def spatial_aggregate_lists(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    rename: dict[str, str],
    predicate: str,
    row_col: str = "_row_id",
) -> pd.DataFrame:
    if left.empty:
        return pd.DataFrame(columns=list(rename.values())).rename_axis(row_col)
    left_geom_col = left.geometry.name
    left_for_join = left[[row_col, left_geom_col]].copy()
    if left_geom_col != "geometry":
        left_for_join = left_for_join.set_geometry(left_geom_col)
    joined = gpd.sjoin(
        left_for_join,
        right[list(rename.keys()) + ["geometry"]].rename(columns=rename),
        how="left",
        predicate=predicate,
    ).drop(columns=["index_right"], errors="ignore")
    agg = {}
    for col in rename.values():
        agg[col] = (lambda values, c=col: unique_clean(values, as_string=c.endswith("_name_list")))
    return joined.groupby(row_col, dropna=False).agg(agg)


def single_or_na(values):
    if isinstance(values, list) and len(values) == 1:
        return values[0]
    return pd.NA


def process_road_change(args: argparse.Namespace) -> None:
    log("[road-change] Loading input")
    require_path(args.road_change_gpkg, "road change gpkg")
    layers = None
    try:
        import fiona

        layers = fiona.listlayers(args.road_change_gpkg)
    except Exception:
        layers = None
    layer = args.road_change_layer or (layers[0] if layers else None)
    events = gpd.read_file(args.road_change_gpkg, layer=layer)
    if events.crs is None:
        events = events.set_crs("EPSG:2263")
    borough, area = load_borough_area(args.borough_shp, args.area_shp, events.crs, args.exclude_area_ids)

    events = events.drop(
        columns=[
            c
            for c in [
                "borough_id",
                "borough_name",
                "area_id",
                "area_name",
                "borough_id_list",
                "borough_name_list",
                "area_id_list",
                "area_name_list",
                "borough_count",
                "area_count",
                "geometry_type",
                "index_right",
            ]
            if c in events.columns
        ],
        errors="ignore",
    ).copy()
    events["_row_id"] = np.arange(len(events))
    events["geometry_type"] = events.geometry.geom_type

    is_point = events.geometry.geom_type.eq("Point")
    point = events.loc[is_point].copy()
    nonpoint = events.loc[~is_point].copy()
    parts = []

    if not point.empty:
        lookup = point_borough_area_lookup(point, borough, area)
        point_out = point.set_index("_row_id").join(lookup, how="left").reset_index()
        point_out["borough_id_list"] = point_out["borough_id"].map(lambda value: [] if pd.isna(value) else [clean_id(value)])
        point_out["borough_name_list"] = point_out["borough_name"].map(lambda value: [] if pd.isna(value) else [str(value)])
        point_out["area_id_list"] = point_out["area_id"].map(lambda value: [] if pd.isna(value) else [clean_id(value)])
        point_out["area_name_list"] = point_out["area_name"].map(lambda value: [] if pd.isna(value) else [str(value)])
        parts.append(point_out)

    if not nonpoint.empty:
        b_lists = spatial_aggregate_lists(
            nonpoint,
            borough,
            {"BoroCode": "borough_id_list", "BoroName": "borough_name_list"},
            "intersects",
        )
        a_lists = spatial_aggregate_lists(
            nonpoint,
            area,
            {"OBJECTID": "area_id_list", "zone": "area_name_list"},
            "intersects",
        )
        line_out = nonpoint.set_index("_row_id").join(b_lists, how="left").join(a_lists, how="left").reset_index()
        for col in ["borough_id_list", "borough_name_list", "area_id_list", "area_name_list"]:
            line_out[col] = line_out[col].map(lambda value: value if isinstance(value, list) else [])
        parts.append(line_out)

    result = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=events.crs)
    result = result.sort_values("_row_id").reset_index(drop=True)
    result["borough_id"] = result["borough_id_list"].map(single_or_na)
    result["borough_name"] = result["borough_name_list"].map(single_or_na)
    result["area_id"] = result["area_id_list"].map(single_or_na)
    result["area_name"] = result["area_name_list"].map(single_or_na)
    result["borough_count"] = result["borough_id_list"].map(lambda values: len(values) if isinstance(values, list) else 0)
    result["area_count"] = result["area_id_list"].map(lambda values: len(values) if isinstance(values, list) else 0)
    result["borough_id_list_json"] = result["borough_id_list"].map(lambda values: json.dumps(values, ensure_ascii=False))
    result["borough_name_list_json"] = result["borough_name_list"].map(lambda values: json.dumps(values, ensure_ascii=False))
    result["area_id_list_json"] = result["area_id_list"].map(lambda values: json.dumps(values, ensure_ascii=False))
    result["area_name_list_json"] = result["area_name_list"].map(lambda values: json.dumps(values, ensure_ascii=False))
    result = result.drop(
        columns=["borough_id_list", "borough_name_list", "area_id_list", "area_name_list", "_row_id"],
        errors="ignore",
    )

    ensure_parent(args.road_change_out_gpkg)
    result.to_file(args.road_change_out_gpkg, layer=args.road_change_out_layer, driver="GPKG")
    pd.DataFrame(result.drop(columns="geometry")).to_csv(args.road_change_out_csv, index=False, encoding="utf-8-sig")
    log(f"  wrote {args.road_change_out_gpkg} ({len(result):,} rows)")
    log(f"  wrote {args.road_change_out_csv}")
    log(f"  single borough_id rows: {result['borough_id'].notna().sum():,}")
    log(f"  single area_id rows: {result['area_id'].notna().sum():,}")
    log(f"  multi-borough rows: {(result['borough_count'] > 1).sum():,}")
    log(f"  multi-area rows: {(result['area_count'] > 1).sum():,}")


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args.data_root = args.data_root.resolve()
    meta_nyc = args.data_root / "Meta_data" / "NYC"
    processed_nyc = (args.processed_dir or (args.data_root / "Processed_data" / "NYC")).resolve()
    event_meta = meta_nyc / "Event"

    args.borough_shp = args.borough_shp or (meta_nyc / "Administrative_data" / "Borough" / "Borough.shp")
    args.area_shp = args.area_shp or (meta_nyc / "Administrative_data" / "Area" / "Area.shp")

    args.static_poi_csv = args.static_poi_csv or (meta_nyc / "POI" / "poi_filter.csv")
    args.static_road_csv = args.static_road_csv or (meta_nyc / "RoadNetwork" / "road_filter.csv")
    args.static_node_csv = args.static_node_csv or (meta_nyc / "RoadNetwork" / "node_filter.csv")
    args.static_borough_out = args.static_borough_out or (processed_nyc / "NYC_borough.csv")
    args.static_area_out = args.static_area_out or (processed_nyc / "NYC_area.csv")
    args.static_poi_out = args.static_poi_out or (processed_nyc / "NYC_poi.csv")
    args.static_road_out = args.static_road_out or (processed_nyc / "NYC_road.csv")
    args.static_junction_out = args.static_junction_out or (processed_nyc / "NYC_junction.csv")

    args.poi_events_csv = args.poi_events_csv or choose_existing(
        processed_nyc / "nyc_poi_events.csv",
        event_meta / "nyc_poi_events.csv",
    )
    args.poi_events_out = args.poi_events_out or (processed_nyc / "nyc_poi_events_with_borough_area.csv")

    args.land_dev_csv = args.land_dev_csv or choose_existing(
        processed_nyc / "nyc_land_development.csv",
        event_meta / "nyc_land_development.csv",
    )
    args.land_dev_out = args.land_dev_out or (processed_nyc / "nyc_land_development_with_borough_area.csv")

    args.road_repair_csv = args.road_repair_csv or choose_existing(
        processed_nyc / "nyc_road_repair.csv",
        event_meta / "nyc_road_repair.csv",
    )
    args.road_repair_out = args.road_repair_out or (processed_nyc / "nyc_road_repair_with_borough_area.csv")

    args.road_change_gpkg = args.road_change_gpkg or (processed_nyc / "nyc_road_change.gpkg")
    args.road_change_out_gpkg = args.road_change_out_gpkg or (processed_nyc / "nyc_road_change_with_borough_area.gpkg")
    args.road_change_out_csv = args.road_change_out_csv or (processed_nyc / "nyc_road_change_with_borough_area.csv")
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NYC static preprocessing and event borough/area alignment in one script."
    )
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=["all"], choices=["all", *TASKS])
    parser.add_argument("--exclude-area-ids", type=int, nargs="*", default=[1, 103, 104])

    parser.add_argument("--borough-shp", type=Path, default=None)
    parser.add_argument("--area-shp", type=Path, default=None)

    parser.add_argument("--static-poi-csv", type=Path, default=None)
    parser.add_argument("--static-road-csv", type=Path, default=None)
    parser.add_argument("--static-node-csv", type=Path, default=None)
    parser.add_argument("--static-borough-out", type=Path, default=None)
    parser.add_argument("--static-area-out", type=Path, default=None)
    parser.add_argument("--static-poi-out", type=Path, default=None)
    parser.add_argument("--static-road-out", type=Path, default=None)
    parser.add_argument("--static-junction-out", type=Path, default=None)

    parser.add_argument("--poi-events-csv", type=Path, default=None)
    parser.add_argument("--poi-events-out", type=Path, default=None)

    parser.add_argument("--land-dev-csv", type=Path, default=None)
    parser.add_argument("--land-dev-out", type=Path, default=None)

    parser.add_argument("--road-repair-csv", type=Path, default=None)
    parser.add_argument("--road-repair-out", type=Path, default=None)

    parser.add_argument("--road-change-gpkg", type=Path, default=None)
    parser.add_argument("--road-change-layer", default=None)
    parser.add_argument("--road-change-out-gpkg", type=Path, default=None)
    parser.add_argument("--road-change-out-csv", type=Path, default=None)
    parser.add_argument("--road-change-out-layer", default="lion_change_events_with_borough_area")
    return resolve_defaults(parser.parse_args())


def main() -> None:
    args = parse_args()
    tasks = list(TASKS) if "all" in args.tasks else args.tasks
    log(f"Tasks: {', '.join(tasks)}")
    log(f"Data root: {args.data_root}")
    ensure_geo_deps()

    if "static" in tasks:
        process_static(args)
    if "poi-events" in tasks:
        process_poi_events(args)
    if "land-dev" in tasks:
        process_land_dev(args)
    if "road-repair" in tasks:
        process_road_repair(args)
    if "road-change" in tasks:
        process_road_change(args)

    log("Done.")


if __name__ == "__main__":
    main()
