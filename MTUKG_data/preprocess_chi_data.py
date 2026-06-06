from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


TASKS = ("static", "poi-events", "land-dev", "road-repair", "road-change")
STATIC_SUBTASKS = ("admin", "poi", "road", "junction")
WGS84 = "EPSG:4326"
ROAD_CHANGE_CRS = "EPSG:3435"

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
            "This script needs pandas, geopandas, shapely, pyproj, and a GeoPandas "
            "I/O backend such as fiona or pyogrio. Run it in the GIS Python "
            "environment used by the original CHI preprocessing scripts."
        ) from exc
    gpd = _gpd
    np = _np
    pd = _pd
    wkt = _wkt


def log(message: str) -> None:
    print(message, flush=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def first_available(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def first_existing(df, candidates: Iterable[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing {label}. Tried columns: {', '.join(candidates)}")


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


def clean_string_id(series):
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def infer_point_columns(
    df,
    lon_candidates=("lng", "longitude", "lon", "x", "Longitude", "LONGITUDE"),
    lat_candidates=("lat", "latitude", "y", "Latitude", "LATITUDE"),
    explicit_lon: str | None = None,
    explicit_lat: str | None = None,
):
    if explicit_lon and explicit_lat:
        if explicit_lon not in df.columns or explicit_lat not in df.columns:
            raise ValueError(f"Missing coordinate columns: {explicit_lon}, {explicit_lat}")
        return explicit_lon, explicit_lat, "explicit"

    lon_col = first_existing(df, lon_candidates, "longitude")
    lat_col = first_existing(df, lat_candidates, "latitude")

    lon = pd.to_numeric(df[lon_col], errors="coerce")
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon_med = lon.dropna().median()
    lat_med = lat.dropna().median()

    # Chicago coordinates should be roughly lon=-88, lat=42. Some node files
    # label these two columns in reverse order.
    if pd.notna(lon_med) and pd.notna(lat_med) and lon_med > 0 and lat_med < 0:
        return lat_col, lon_col, "swapped_by_value"
    return lon_col, lat_col, "as_labeled"


def admin_roots(data_root: Path, processed_dir: Path | None):
    meta_chi = data_root / "Meta_data" / "CHI"
    if (meta_chi / "Administrative_data").exists():
        source_root = meta_chi
        default_processed = data_root / "Processed_data" / "CHI"
        layout = "mtukg"
    else:
        source_root = data_root
        default_processed = data_root / "Processed_data"
        layout = "meta_data_collect"
    processed = (processed_dir or default_processed).resolve()
    return source_root, processed, layout


def load_admin(args: argparse.Namespace, target_crs=WGS84):
    require_path(args.borough_shp, "borough shapefile")
    require_path(args.area_shp, "area shapefile")
    borough = gpd.read_file(args.borough_shp)
    area = gpd.read_file(args.area_shp)

    if borough.crs is None:
        borough = borough.set_crs(WGS84, allow_override=True)
    if area.crs is None:
        area = area.set_crs(WGS84, allow_override=True)

    borough = borough.to_crs(target_crs)
    area = area.to_crs(target_crs)

    missing_b = [c for c in [args.borough_id_col, args.borough_name_col] if c not in borough.columns]
    if missing_b:
        raise ValueError(f"Borough shapefile missing columns: {missing_b}")
    missing_a = [c for c in [args.area_id_col, args.area_name_col] if c not in area.columns]
    if missing_a:
        raise ValueError(f"Area shapefile missing columns: {missing_a}")

    borough = borough[[args.borough_id_col, args.borough_name_col, "geometry"]].copy()
    area = area[[args.area_id_col, args.area_name_col, "geometry"]].copy()

    if args.exclude_area_ids:
        area = area[~area[args.area_id_col].isin(args.exclude_area_ids)].copy()
    return borough, area


def spatial_lookup(left, right, rename: dict[str, str], predicate: str, row_col="_row_id"):
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


def join_borough_area(gdf, borough, area, args):
    b_rename = {args.borough_id_col: "borough_id", args.borough_name_col: "borough_name"}
    a_rename = {args.area_id_col: "area_id", args.area_name_col: "area_name"}

    b_primary = spatial_lookup(gdf, borough, b_rename, args.primary_point_predicate)
    b_fallback = spatial_lookup(gdf, borough, b_rename, "intersects")
    a_primary = spatial_lookup(gdf, area, a_rename, args.primary_point_predicate)
    a_fallback = spatial_lookup(gdf, area, a_rename, "intersects")
    return b_primary.combine_first(b_fallback).join(a_primary.combine_first(a_fallback), how="outer")


def process_static_point_csv(
    input_csv: Path,
    output_csv: Path,
    borough,
    area,
    args: argparse.Namespace,
    task_name: str,
) -> None:
    log(f"[{task_name}] Loading {input_csv}")
    require_path(input_csv, f"{task_name} input")
    df = pd.read_csv(input_csv, low_memory=False)
    df = df.drop(
        columns=[c for c in ["borough_id", "borough_name", "area_id", "area_name", "index_right"] if c in df.columns]
    )
    lon_col, lat_col, coord_mode = infer_point_columns(df)
    log(f"  coordinates: lon={lon_col}, lat={lat_col} ({coord_mode})")
    df = df.dropna(subset=[lon_col, lat_col]).copy()
    df["_row_id"] = np.arange(len(df))
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=WGS84)
    lookup = join_borough_area(gdf, borough, area, args)
    out = pd.DataFrame(gdf.drop(columns="geometry")).set_index("_row_id").join(lookup, how="left").reset_index(drop=True)
    out = out.dropna(subset=["borough_id", "area_id"])
    ensure_parent(output_csv)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    log(f"  wrote {output_csv} ({len(out):,} rows)")
    log(f"  matched borough_id: {out['borough_id'].notna().sum():,}")
    log(f"  matched area_id: {out['area_id'].notna().sum():,}")


def process_static_road(args: argparse.Namespace, borough, area) -> None:
    log(f"[road] Loading {args.road_csv}")
    require_path(args.road_csv, "road input")
    road = pd.read_csv(args.road_csv, low_memory=False)
    if "geometry" not in road.columns:
        raise ValueError("Road CSV must contain a WKT geometry column named 'geometry'.")
    road = road.drop(
        columns=[c for c in ["borough_id", "borough_name", "area_id", "area_name", "index_right"] if c in road.columns]
    )
    road["_row_id"] = np.arange(len(road))

    def parse_geom(value):
        if pd.isna(value):
            return None
        try:
            return wkt.loads(str(value))
        except Exception:
            return None

    road_geom = road["geometry"].map(parse_geom)
    road_gdf = gpd.GeoDataFrame(
        road.copy(),
        geometry=gpd.GeoSeries(road_geom, index=road.index, crs=WGS84, name="_shape"),
        crs=WGS84,
    )
    road_gdf = road_gdf.loc[road_gdf.geometry.notna()].copy()
    b_lookup = spatial_lookup(
        road_gdf,
        borough,
        {args.borough_id_col: "borough_id", args.borough_name_col: "borough_name"},
        args.road_predicate,
    )
    a_lookup = spatial_lookup(
        road_gdf,
        area,
        {args.area_id_col: "area_id", args.area_name_col: "area_name"},
        args.road_predicate,
    )
    out = pd.DataFrame(road_gdf.drop(columns="_shape")).set_index("_row_id")
    out = out.join(b_lookup, how="left").join(a_lookup, how="left").reset_index(drop=True)
    out = out.dropna(subset=["borough_id", "area_id"])
    ensure_parent(args.road_out)
    out.to_csv(args.road_out, index=False, encoding="utf-8-sig")
    log(f"  wrote {args.road_out} ({len(out):,} rows)")
    log(f"  matched borough_id: {out['borough_id'].notna().sum():,}")
    log(f"  matched area_id: {out['area_id'].notna().sum():,}")


def process_static(args: argparse.Namespace, subtasks: list[str]) -> None:
    log("[static] Loading CHI borough and area shapefiles")
    borough, area = load_admin(args, WGS84)

    if "admin" in subtasks:
        ensure_parent(args.borough_out)
        borough.to_csv(args.borough_out, index=False, encoding="utf-8-sig")
        area.to_csv(args.area_out, index=False, encoding="utf-8-sig")
        log(f"  wrote {args.borough_out} ({len(borough):,} rows)")
        log(f"  wrote {args.area_out} ({len(area):,} rows)")
    if "poi" in subtasks:
        process_static_point_csv(args.poi_csv, args.poi_out, borough, area, args, "static-poi")
    if "road" in subtasks:
        process_static_road(args, borough, area)
    if "junction" in subtasks:
        process_static_point_csv(args.node_csv, args.junction_out, borough, area, args, "static-junction")


def read_boundary_for_poi(path: Path, kind: str, args: argparse.Namespace):
    require_path(path, f"{kind} boundary")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)
    gdf = gdf.to_crs(WGS84)
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    if kind == "area":
        id_col = args.area_id_col if args.area_id_col in gdf.columns else "area_num_1"
        name_col = args.area_name_col if args.area_name_col in gdf.columns else "community"
    elif kind == "borough":
        id_col = args.borough_id_col
        name_col = args.borough_name_col
    else:
        raise ValueError("kind must be area or borough")

    if id_col not in gdf.columns or name_col not in gdf.columns:
        raise ValueError(f"{kind} boundary missing expected columns. Current columns: {list(gdf.columns)}")

    out = gdf[[id_col, name_col, "geometry"]].copy()
    out = out.rename(columns={id_col: "boundary_id", name_col: "boundary_name"})
    out["boundary_id"] = clean_string_id(out["boundary_id"])
    out["boundary_name"] = out["boundary_name"].astype(str).str.strip()
    return out


def valid_coordinate_mask(df, lat_col: str, lon_col: str):
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Input table is missing coordinate columns: {lat_col}, {lon_col}")
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    mask = lat.notna() & lon.notna() & lat.between(41.0, 43.0) & lon.between(-89.0, -86.0)
    return mask, lat, lon


def first_match_by_point(pair_array):
    if pair_array.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    point_idx = pair_array[0]
    poly_idx = pair_array[1]
    _, first_pos = np.unique(point_idx, return_index=True)
    return point_idx[first_pos], poly_idx[first_pos]


def assign_boundary(df_len, row_ids, lon_values, lat_values, boundary_gdf, prefix, tolerance_m, metric_crs):
    from pyproj import Transformer
    from shapely import points as shapely_points
    from shapely.strtree import STRtree

    result = pd.DataFrame({"__row_id": np.arange(df_len)})
    result[f"{prefix}_id"] = pd.NA
    result[f"{prefix}_name"] = pd.NA
    result[f"{prefix}_match_method"] = pd.NA
    result[f"{prefix}_nearest_distance_m"] = pd.NA

    if len(row_ids) == 0:
        return result

    poly_geoms = boundary_gdf.geometry.to_numpy()
    poly_ids = boundary_gdf["boundary_id"].to_numpy()
    poly_names = boundary_gdf["boundary_name"].to_numpy()

    point_geoms = shapely_points(lon_values.to_numpy(), lat_values.to_numpy())
    tree = STRtree(poly_geoms)
    pairs = tree.query(point_geoms, predicate="intersects")
    direct_pt_idx, direct_poly_idx = first_match_by_point(pairs)

    if len(direct_pt_idx) > 0:
        direct_rows = row_ids[direct_pt_idx]
        result.loc[direct_rows, f"{prefix}_id"] = poly_ids[direct_poly_idx]
        result.loc[direct_rows, f"{prefix}_name"] = poly_names[direct_poly_idx]
        result.loc[direct_rows, f"{prefix}_match_method"] = "intersects"
        result.loc[direct_rows, f"{prefix}_nearest_distance_m"] = 0.0

    matched_valid = np.zeros(len(row_ids), dtype=bool)
    if len(direct_pt_idx) > 0:
        matched_valid[direct_pt_idx] = True
    unmatched_idx = np.where(~matched_valid)[0]

    if len(unmatched_idx) > 0 and tolerance_m > 0:
        boundary_m = boundary_gdf.to_crs(metric_crs)
        poly_m = boundary_m.geometry.to_numpy()
        tree_m = STRtree(poly_m)

        transformer = Transformer.from_crs(WGS84, metric_crs, always_xy=True)
        x_m, y_m = transformer.transform(
            lon_values.iloc[unmatched_idx].to_numpy(),
            lat_values.iloc[unmatched_idx].to_numpy(),
        )
        points_m = shapely_points(x_m, y_m)
        nearest_pairs, distances = tree_m.query_nearest(
            points_m,
            max_distance=tolerance_m,
            return_distance=True,
            all_matches=False,
        )
        if nearest_pairs.size > 0:
            near_pt_rel = nearest_pairs[0]
            near_poly_idx = nearest_pairs[1]
            original_idx = unmatched_idx[near_pt_rel]
            near_rows = row_ids[original_idx]
            result.loc[near_rows, f"{prefix}_id"] = poly_ids[near_poly_idx]
            result.loc[near_rows, f"{prefix}_name"] = poly_names[near_poly_idx]
            result.loc[near_rows, f"{prefix}_match_method"] = f"nearest_within_{int(tolerance_m)}m"
            result.loc[near_rows, f"{prefix}_nearest_distance_m"] = np.round(distances, 3)

    return result


def update_or_create(result, join_df, prefix, overwrite_non_null=True):
    indexed = join_df.set_index("__row_id")
    for field in ["id", "name", "match_method", "nearest_distance_m"]:
        col = f"{prefix}_{field}"
        if col not in result.columns:
            result[col] = pd.NA
        mapped = indexed[col]
        if overwrite_non_null:
            result[col] = mapped.combine_first(result[col])
        else:
            result[col] = result[col].combine_first(mapped)
    return result


def build_area_to_borough_map(area_gdf, borough_gdf):
    from shapely.strtree import STRtree

    area_pts = area_gdf.geometry.representative_point().to_numpy()
    tree = STRtree(borough_gdf.geometry.to_numpy())
    pairs = tree.query(area_pts, predicate="intersects")
    pt_idx, poly_idx = first_match_by_point(pairs)

    mapping = pd.DataFrame(
        {
            "area_id": area_gdf["boundary_id"].astype(str).values,
            "area_name": area_gdf["boundary_name"].astype(str).values,
            "map_borough_id": pd.NA,
            "map_borough_name": pd.NA,
        }
    )
    if len(pt_idx) > 0:
        mapping.loc[pt_idx, "map_borough_id"] = borough_gdf["boundary_id"].to_numpy()[poly_idx]
        mapping.loc[pt_idx, "map_borough_name"] = borough_gdf["boundary_name"].to_numpy()[poly_idx]
    return mapping


def strip_admin_columns(df):
    remove = []
    exact = {
        "index_right",
        "__row_id",
        "area_id",
        "area_name",
        "area_match_method",
        "area_nearest_distance_m",
        "borough_id",
        "borough_name",
        "borough_match_method",
        "borough_nearest_distance_m",
    }
    for col in df.columns:
        if col in exact or col.startswith("area_") or col.startswith("borough_"):
            remove.append(col)
    return df.drop(columns=remove, errors="ignore")


def process_poi_events(args: argparse.Namespace) -> None:
    log(f"[poi-events] Loading {args.poi_events_csv}")
    require_path(args.poi_events_csv, "POI event input")
    df = pd.read_csv(args.poi_events_csv, low_memory=False)
    df = strip_admin_columns(df)

    lon_col, lat_col, coord_mode = infer_point_columns(df, explicit_lon=args.event_lon_col, explicit_lat=args.event_lat_col)
    log(f"  coordinates: lon={lon_col}, lat={lat_col} ({coord_mode})")
    mask, lat, lon = valid_coordinate_mask(df, lat_col, lon_col)
    valid_row_ids = np.where(mask.to_numpy())[0]
    lat_valid = lat.iloc[valid_row_ids].reset_index(drop=True)
    lon_valid = lon.iloc[valid_row_ids].reset_index(drop=True)
    log(f"  valid coordinate rows: {len(valid_row_ids):,} / {len(df):,}")

    area = read_boundary_for_poi(args.area_shp, "area", args)
    borough = read_boundary_for_poi(args.borough_shp, "borough", args)
    log(f"  area polygons: {len(area):,}; borough polygons: {len(borough):,}")

    area_join = assign_boundary(
        len(df),
        valid_row_ids,
        lon_valid,
        lat_valid,
        area,
        "area",
        args.nearest_tolerance_m,
        args.poi_metric_crs,
    )

    result = df.copy()
    result["__row_id"] = np.arange(len(result))
    result = update_or_create(result, area_join, "area", overwrite_non_null=True)
    result["area_id"] = clean_string_id(result["area_id"])

    area_borough_map = build_area_to_borough_map(area, borough)
    map_dict_id = dict(zip(area_borough_map["area_id"], area_borough_map["map_borough_id"]))
    map_dict_name = dict(zip(area_borough_map["area_id"], area_borough_map["map_borough_name"]))

    result["borough_id"] = pd.NA
    result["borough_name"] = pd.NA
    result["borough_match_method"] = pd.NA
    result["borough_nearest_distance_m"] = pd.NA
    mapped_boro_id = result["area_id"].map(map_dict_id)
    mapped_boro_name = result["area_id"].map(map_dict_name)
    can_map = mapped_boro_id.notna()
    result.loc[can_map, "borough_id"] = mapped_boro_id[can_map]
    result.loc[can_map, "borough_name"] = mapped_boro_name[can_map]
    result.loc[can_map, "borough_match_method"] = "area_to_borough"
    result.loc[can_map, "borough_nearest_distance_m"] = 0.0

    result["borough_id"] = clean_string_id(result["borough_id"])
    result = result.drop(columns=["__row_id"])

    ensure_parent(args.poi_events_out)
    result.to_csv(args.poi_events_out, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "input_path": str(args.poi_events_csv),
                "output_path": str(args.poi_events_out),
                "area_path": str(args.area_shp),
                "borough_path": str(args.borough_shp),
                "rows": len(result),
                "valid_coordinate_rows": len(valid_row_ids),
                "area_matched_rows": int(result["area_match_method"].notna().sum()),
                "borough_matched_rows": int(result["borough_match_method"].notna().sum()),
                "nearest_tolerance_m": args.nearest_tolerance_m,
                "poi_metric_crs": args.poi_metric_crs,
            }
        ]
    )
    ensure_parent(args.poi_events_summary)
    summary.to_csv(args.poi_events_summary, index=False, encoding="utf-8-sig")

    log(f"  wrote {args.poi_events_out} ({len(result):,} rows)")
    log(f"  area matched rows: {summary.loc[0, 'area_matched_rows']:,}")
    log(f"  borough matched rows: {summary.loc[0, 'borough_matched_rows']:,}")
    log(f"  summary: {args.poi_events_summary}")


def read_polygon_layer(path: Path, prefix: str, target_crs=WGS84):
    require_path(path, f"{prefix}polygon layer")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        log(f"[warn] {path.name} has no CRS; assuming {WGS84}")
        gdf = gdf.set_crs(WGS84, allow_override=True)
    gdf = gdf.to_crs(target_crs)

    rename_map = {col: f"{prefix}{col}" for col in gdf.columns if col != "geometry"}
    gdf = gdf.rename(columns=rename_map)
    return gdf[list(rename_map.values()) + ["geometry"]].copy()


def spatial_join_one_match(points, polygons, predicate: str):
    joined = gpd.sjoin(points, polygons, how="left", predicate=predicate)
    if joined["__row_id"].duplicated().any():
        joined = (
            joined.sort_values(["__row_id", "index_right"], na_position="last", kind="stable")
            .drop_duplicates(subset=["__row_id"], keep="first")
        )
    return joined.drop(columns=["index_right"], errors="ignore").sort_values("__row_id", kind="stable")


def count_prefixed_matches(df, prefix: str) -> int:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return 0
    return int(df[cols].notna().any(axis=1).sum())


def process_point_event_admin_join(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    args: argparse.Namespace,
    task_name: str,
) -> None:
    log(f"[{task_name}] Loading {input_path}")
    require_path(input_path, f"{task_name} input")
    area = read_polygon_layer(args.area_shp, "area_", WGS84)
    borough = read_polygon_layer(args.borough_shp, "borough_", WGS84)
    log(f"  area polygons: {len(area):,}; borough polygons: {len(borough):,}")

    same_input_output = input_path.resolve() == output_path.resolve()
    write_path = output_path
    if same_input_output:
        write_path = output_path.with_name(f"{output_path.stem}.tmp_admin_joined{output_path.suffix}")
        log(f"  input and output are the same path; writing temporary file first: {write_path}")

    ensure_parent(write_path)
    if write_path.exists():
        write_path.unlink()

    total_rows = 0
    valid_rows = 0
    written_rows = 0
    area_matched = 0
    borough_matched = 0
    first_write = True
    coord_info = None

    reader = pd.read_csv(input_path, chunksize=args.chunksize, dtype=str, encoding="utf-8-sig", low_memory=False)
    for chunk_idx, chunk in enumerate(reader, start=1):
        original_len = len(chunk)
        chunk = strip_admin_columns(chunk)
        lon_col, lat_col, coord_mode = infer_point_columns(
            chunk,
            explicit_lon=args.event_lon_col,
            explicit_lat=args.event_lat_col,
        )
        if coord_info is None:
            coord_info = (lon_col, lat_col, coord_mode)
            log(f"  coordinates: lon={lon_col}, lat={lat_col} ({coord_mode})")

        chunk[lat_col] = pd.to_numeric(chunk[lat_col], errors="coerce")
        chunk[lon_col] = pd.to_numeric(chunk[lon_col], errors="coerce")
        chunk = chunk.loc[chunk[lat_col].notna() & chunk[lon_col].notna()].copy()
        if chunk.empty:
            total_rows += original_len
            continue

        chunk["__row_id"] = range(total_rows, total_rows + len(chunk))
        points = gpd.GeoDataFrame(
            chunk,
            geometry=gpd.points_from_xy(chunk[lon_col], chunk[lat_col]),
            crs=WGS84,
        )

        joined = spatial_join_one_match(points, area, predicate=args.point_event_predicate)
        joined = spatial_join_one_match(joined, borough, predicate=args.point_event_predicate)
        area_matched += count_prefixed_matches(joined, "area_")
        borough_matched += count_prefixed_matches(joined, "borough_")

        out = pd.DataFrame(joined.drop(columns=["geometry", "__row_id"], errors="ignore"))
        out.to_csv(
            write_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig" if first_write else "utf-8",
        )
        first_write = False
        total_rows += original_len
        valid_rows += len(chunk)
        written_rows += len(out)

        log(
            f"  chunk={chunk_idx:,}, processed={total_rows:,}, written={written_rows:,}, "
            f"area_matched={area_matched:,}, borough_matched={borough_matched:,}"
        )

    if same_input_output and write_path.exists():
        write_path.replace(output_path)

    summary = pd.DataFrame(
        [
            {
                "task": task_name,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "area_path": str(args.area_shp),
                "borough_path": str(args.borough_shp),
                "predicate": args.point_event_predicate,
                "processed_rows": total_rows,
                "valid_coordinate_rows": valid_rows,
                "written_rows": written_rows,
                "area_matched_rows": area_matched,
                "area_unmatched_rows": written_rows - area_matched,
                "borough_matched_rows": borough_matched,
                "borough_unmatched_rows": written_rows - borough_matched,
            }
        ]
    )
    ensure_parent(summary_path)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    log(f"  wrote {output_path} ({written_rows:,} rows)")
    log(f"  summary: {summary_path}")


def choose_best_polygon_by_intersection_length(events, polygons, prefix: str):
    attr_cols = [col for col in polygons.columns if col != "geometry"]
    length_col = f"{prefix}intersection_length_ft"
    if events.empty:
        return pd.DataFrame(columns=["__row_id"] + attr_cols + [length_col])

    candidates = gpd.sjoin(events[["__row_id", "geometry"]], polygons, how="left", predicate="intersects")
    matched = candidates.loc[candidates["index_right"].notna()].copy()
    if matched.empty:
        return pd.DataFrame(columns=["__row_id"] + attr_cols + [length_col])

    right_index = matched["index_right"].astype(int)
    right_geoms = polygons.geometry.loc[right_index].reset_index(drop=True)
    left_geoms = matched.geometry.reset_index(drop=True)
    matched[length_col] = left_geoms.intersection(right_geoms, align=False).length

    sort_cols = ["__row_id", length_col, "index_right"]
    matched = matched.sort_values(sort_cols, ascending=[True, False, True], kind="stable")
    best = matched.drop_duplicates(subset=["__row_id"], keep="first")
    return pd.DataFrame(best[["__row_id"] + attr_cols + [length_col]])


def validate_csv_gpkg_alignment(csv_path: Path, events, event_id_col="event_id") -> None:
    if not csv_path.exists():
        log(f"  [warn] CSV not found, using GPKG attributes only: {csv_path}")
        return
    try:
        csv_header = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig")
        if event_id_col not in csv_header.columns:
            log(f"  [warn] CSV has no {event_id_col}; skipped CSV/GPKG alignment check.")
            return
        csv_ids = pd.read_csv(csv_path, usecols=[event_id_col], dtype=str, encoding="utf-8-sig")
        csv_count = len(csv_ids)
        gpkg_count = len(events)
        missing_in_gpkg = 0
        if event_id_col in events.columns:
            missing_in_gpkg = int((~csv_ids[event_id_col].isin(events[event_id_col].astype(str))).sum())
        log(f"  CSV rows={csv_count:,}, GPKG rows={gpkg_count:,}, CSV ids missing in GPKG={missing_in_gpkg:,}")
    except Exception as exc:
        log(f"  [warn] CSV/GPKG alignment check failed: {exc}")


def process_road_change(args: argparse.Namespace) -> None:
    log(f"[road-change] Loading {args.road_change_gpkg}")
    require_path(args.road_change_gpkg, "road-change GPKG input")
    events = gpd.read_file(args.road_change_gpkg, layer=args.road_change_layer or None)
    if events.crs is None:
        raise RuntimeError(f"Road-change GPKG has no CRS: {args.road_change_gpkg}")
    events = events.to_crs(args.road_change_crs)
    events = events.loc[events.geometry.notna() & ~events.geometry.is_empty].copy()
    events["__row_id"] = range(len(events))
    validate_csv_gpkg_alignment(args.road_change_csv, events)

    area = read_polygon_layer(args.area_shp, "area_", events.crs)
    borough = read_polygon_layer(args.borough_shp, "borough_", events.crs)
    log(f"  event rows with geometry: {len(events):,}")
    log(f"  area polygons: {len(area):,}; borough polygons: {len(borough):,}")

    area_match = choose_best_polygon_by_intersection_length(events, area, "area_")
    borough_match = choose_best_polygon_by_intersection_length(events, borough, "borough_")
    enriched = events.merge(area_match, on="__row_id", how="left")
    enriched = enriched.merge(borough_match, on="__row_id", how="left")
    enriched = enriched.drop(columns=["__row_id"])

    area_matched = count_prefixed_matches(enriched, "area_")
    borough_matched = count_prefixed_matches(enriched, "borough_")

    ensure_parent(args.road_change_out_csv)
    ensure_parent(args.road_change_out_gpkg)
    enriched.drop(columns="geometry").to_csv(args.road_change_out_csv, index=False, encoding="utf-8-sig")
    if args.road_change_out_gpkg.exists():
        args.road_change_out_gpkg.unlink()
    enriched.to_file(
        args.road_change_out_gpkg,
        layer=args.road_change_out_layer,
        driver="GPKG",
    )

    summary = pd.DataFrame(
        [
            {
                "input_csv": str(args.road_change_csv),
                "input_gpkg": str(args.road_change_gpkg),
                "output_csv": str(args.road_change_out_csv),
                "output_gpkg": str(args.road_change_out_gpkg),
                "area_path": str(args.area_shp),
                "borough_path": str(args.borough_shp),
                "event_rows_with_geometry": len(enriched),
                "area_matched_rows": area_matched,
                "area_unmatched_rows": len(enriched) - area_matched,
                "borough_matched_rows": borough_matched,
                "borough_unmatched_rows": len(enriched) - borough_matched,
                "area_match_rule": "polygon with maximum line intersection length",
                "borough_match_rule": "polygon with maximum line intersection length",
                "road_change_crs": str(args.road_change_crs),
            }
        ]
    )
    ensure_parent(args.road_change_summary)
    summary.to_csv(args.road_change_summary, index=False, encoding="utf-8-sig")

    log(f"  wrote {args.road_change_out_csv} ({len(enriched):,} rows)")
    log(f"  wrote {args.road_change_out_gpkg}")
    log(f"  area matched rows: {area_matched:,}")
    log(f"  borough matched rows: {borough_matched:,}")
    log(f"  summary: {args.road_change_summary}")


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args.data_root = args.data_root.resolve()
    source_root, processed, layout = admin_roots(args.data_root, args.processed_dir)
    args.source_root = source_root
    args.processed_dir = processed
    args.layout = layout

    admin_root = source_root / "Administrative_data"
    road_event_dir = source_root / "chicago_road_event_outputs"
    road_change_dir = source_root / "chicago_road_network_change_outputs_osm"

    args.borough_shp = args.borough_shp or (admin_root / "Borough" / "Borough.shp")
    args.area_shp = args.area_shp or (admin_root / "Area" / "Area.shp")
    args.poi_csv = args.poi_csv or (source_root / "POI" / "poi_filter.csv")
    args.road_csv = args.road_csv or (source_root / "RoadNetwork" / "road_filter.csv")
    args.node_csv = args.node_csv or (source_root / "RoadNetwork" / "node_filter.csv")

    args.borough_out = args.borough_out or (processed / "CHI_borough.csv")
    args.area_out = args.area_out or (processed / "CHI_area.csv")
    args.poi_out = args.poi_out or (processed / "CHI_poi.csv")
    args.road_out = args.road_out or (processed / "CHI_road.csv")
    args.junction_out = args.junction_out or (processed / "CHI_junction.csv")

    args.poi_events_csv = args.poi_events_csv or first_available(
        [
            source_root / "chicago_poi_event.csv",
            source_root / "Event" / "chicago_poi_event.csv",
            source_root / "Event" / "chicago_poi_events.csv",
            processed / "chicago_poi_event.csv",
            processed / "chicago_poi_events.csv",
            processed / "chicago_poi_event_with_area_borough.csv",
        ]
    )
    args.poi_events_out = args.poi_events_out or (processed / "chicago_poi_event_with_area_borough.csv")
    args.poi_events_summary = args.poi_events_summary or (processed / "chicago_poi_event_admin_join_summary.csv")

    args.land_dev_csv = args.land_dev_csv or first_available(
        [
            processed / "chi_land_event_time_table_with_end_time.csv",
            source_root / "chi_land_event_time_table_with_end_time.csv",
            source_root / "Event" / "chi_land_event.csv",
            processed / "chi_land_event.csv",
            processed / "chi_land_event_time_table_with_end_time_strong_admin_joined.csv",
        ]
    )
    args.land_dev_out = args.land_dev_out or (processed / "chi_land_event_time_table_with_end_time_strong_admin_joined.csv")
    args.land_dev_summary = args.land_dev_summary or (processed / "chi_land_event_admin_join_summary.csv")

    args.road_repair_csv = args.road_repair_csv or first_available(
        [
            road_event_dir / "chicago_road_event_locator_complete_from_2014.csv",
            source_root / "Event" / "chicago_road_repair.csv",
            processed / "chicago_road_repair.csv",
            processed / "chicago_road_event_locator_complete_from_2014_admin_joined.csv",
        ]
    )
    args.road_repair_out = args.road_repair_out or (processed / "chicago_road_event_locator_complete_from_2014_admin_joined.csv")
    args.road_repair_summary = args.road_repair_summary or (processed / "chicago_road_event_admin_join_summary.csv")

    args.road_change_csv = args.road_change_csv or first_available(
        [
            road_change_dir / "chicago_road_network_change_events_osm_quarterly.csv",
            source_root / "Event" / "chicago_road_change.csv",
            processed / "chicago_road_change.csv",
            processed / "chicago_road_network_change_events_osm_quarterly_admin_joined.csv",
        ]
    )
    args.road_change_gpkg = args.road_change_gpkg or first_available(
        [
            road_change_dir / "chicago_road_network_change_events_osm_quarterly.gpkg",
            processed / "chicago_road_network_change_events_osm_quarterly_admin_joined.gpkg",
        ]
    )
    args.road_change_out_csv = args.road_change_out_csv or (
        processed / "chicago_road_network_change_events_osm_quarterly_admin_joined.csv"
    )
    args.road_change_out_gpkg = args.road_change_out_gpkg or (
        processed / "chicago_road_network_change_events_osm_quarterly_admin_joined.gpkg"
    )
    args.road_change_summary = args.road_change_summary or (
        processed / "chicago_road_network_change_events_osm_quarterly_admin_join_summary.csv"
    )
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Chicago static preprocessing and event-to-area/borough matching in one script. "
            "Use --data-root C:\\Users\\HJH\\Desktop\\CHI\\meta_data_collect for the collected CHI source folder."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=["all"], choices=["all", *TASKS, *STATIC_SUBTASKS])

    parser.add_argument("--borough-shp", type=Path, default=None)
    parser.add_argument("--area-shp", type=Path, default=None)
    parser.add_argument("--poi-csv", type=Path, default=None, help="Static POI CSV.")
    parser.add_argument("--road-csv", type=Path, default=None, help="Static road CSV with WKT geometry.")
    parser.add_argument("--node-csv", type=Path, default=None, help="Static junction/node CSV.")

    parser.add_argument("--borough-out", type=Path, default=None)
    parser.add_argument("--area-out", type=Path, default=None)
    parser.add_argument("--poi-out", type=Path, default=None)
    parser.add_argument("--road-out", type=Path, default=None)
    parser.add_argument("--junction-out", type=Path, default=None)

    parser.add_argument("--poi-events-csv", type=Path, default=None)
    parser.add_argument("--poi-events-out", type=Path, default=None)
    parser.add_argument("--poi-events-summary", type=Path, default=None)
    parser.add_argument("--land-dev-csv", type=Path, default=None)
    parser.add_argument("--land-dev-out", type=Path, default=None)
    parser.add_argument("--land-dev-summary", type=Path, default=None)
    parser.add_argument("--road-repair-csv", type=Path, default=None)
    parser.add_argument("--road-repair-out", type=Path, default=None)
    parser.add_argument("--road-repair-summary", type=Path, default=None)
    parser.add_argument("--road-change-csv", type=Path, default=None)
    parser.add_argument("--road-change-gpkg", type=Path, default=None)
    parser.add_argument("--road-change-out-csv", type=Path, default=None)
    parser.add_argument("--road-change-out-gpkg", type=Path, default=None)
    parser.add_argument("--road-change-summary", type=Path, default=None)

    parser.add_argument("--borough-id-col", default="BoroCode")
    parser.add_argument("--borough-name-col", default="BoroName")
    parser.add_argument("--area-id-col", default="area_numbe")
    parser.add_argument("--area-name-col", default="community")
    parser.add_argument("--exclude-area-ids", type=int, nargs="*", default=[])
    parser.add_argument("--primary-point-predicate", choices=["within", "intersects"], default="within")
    parser.add_argument("--road-predicate", choices=["intersects", "within", "touches"], default="intersects")
    parser.add_argument("--point-event-predicate", choices=["within", "intersects"], default="intersects")
    parser.add_argument("--event-lat-col", default=None)
    parser.add_argument("--event-lon-col", default=None)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--nearest-tolerance-m", type=float, default=30.0)
    parser.add_argument("--poi-metric-crs", default="EPSG:26916")
    parser.add_argument("--road-change-crs", default=ROAD_CHANGE_CRS)
    parser.add_argument("--road-change-layer", default="")
    parser.add_argument("--road-change-out-layer", default="road_network_changes_osm_quarterly_admin_joined")
    return resolve_defaults(parser.parse_args())


def expand_tasks(raw_tasks: list[str]) -> tuple[list[str], list[str]]:
    if "all" in raw_tasks:
        return list(TASKS), list(STATIC_SUBTASKS)
    tasks = []
    static_subtasks = []
    for task in raw_tasks:
        if task in STATIC_SUBTASKS:
            if "static" not in tasks:
                tasks.append("static")
            static_subtasks.append(task)
        else:
            tasks.append(task)
    if "static" in tasks and not static_subtasks:
        static_subtasks = list(STATIC_SUBTASKS)
    return tasks, static_subtasks


def main() -> None:
    args = parse_args()
    tasks, static_subtasks = expand_tasks(args.tasks)
    log(f"Tasks: {', '.join(tasks)}")
    log(f"Data root: {args.data_root}")
    log(f"Source layout: {args.layout}")
    log(f"Source root: {args.source_root}")
    log(f"Processed dir: {args.processed_dir}")
    ensure_geo_deps()

    if "static" in tasks:
        process_static(args, static_subtasks)
    if "poi-events" in tasks:
        process_poi_events(args)
    if "land-dev" in tasks:
        process_point_event_admin_join(
            args.land_dev_csv,
            args.land_dev_out,
            args.land_dev_summary,
            args,
            "land-dev",
        )
    if "road-repair" in tasks:
        process_point_event_admin_join(
            args.road_repair_csv,
            args.road_repair_out,
            args.road_repair_summary,
            args,
            "road-repair",
        )
    if "road-change" in tasks:
        process_road_change(args)

    log("Done.")


if __name__ == "__main__":
    main()
