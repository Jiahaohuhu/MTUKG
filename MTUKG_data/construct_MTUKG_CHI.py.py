from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from build_chi_urban_kg import (
    BASE_DIR,
    CHICAGO_CRS,
    DEFAULT_PATHS,
    KGWriter,
    ZoneCache,
    date_only,
    entity,
    limited_reader,
    load_roads,
    log,
    norm_number,
    read_wkt_gdf,
    time_pair,
)
from build_chi_urban_kg_nyc_style_temporal import (
    add_points_to_functional_zones_by_overlap,
    available_zone_quarters,
    build_temporal_poi_nyc_style,
    build_temporal_road_repair_nyc_style,
    finalize_dataset,
    prepare_borough_projected,
    quarter_labels_between,
)
from build_chi_urban_kg_nyc_style_temporal_with_fz import (
    add_functional_zone_background_relations,
    build_area_name_to_id,
)
from build_chi_urban_kg_static_temporal_2019_2022 import (
    DEFAULT_POI_EVENT_START_GE_2014,
    DEFAULT_ROAD_REPAIR_2019_2022,
    build_static_kg,
)


DEFAULT_OUT_DIR = BASE_DIR / "CHIUrbanKG_nyc_style_all_events_with_fz"
DEFAULT_LAND_DEV = DEFAULT_PATHS["land_dev"]
DEFAULT_ROAD_CHANGE_GPKG = (
    BASE_DIR
    / "chicago_road_network_change_outputs_osm_2014_2025"
    / "chicago_road_network_change_events_osm_quarterly.gpkg"
)


def build_osm_to_road_map(road_gdf: gpd.GeoDataFrame) -> dict[str, list[str]]:
    osm_to_roads: dict[str, list[str]] = {}
    for row in road_gdf.itertuples(index=False):
        osm_id = norm_number(getattr(row, "osm_way_id", ""))
        road = getattr(row, "road_entity", "")
        if osm_id and road:
            osm_to_roads.setdefault(osm_id, [])
            if road not in osm_to_roads[osm_id]:
                osm_to_roads[osm_id].append(road)
    return osm_to_roads


def add_land_dev_to_borough(kg: KGWriter, chunk: pd.DataFrame) -> None:
    for row in chunk.to_dict("records"):
        event_id = row.get("event_entity", "")
        start, end = time_pair(row, "start_time", "end_time")
        kg.add_entity(event_id, "LandDevelopmentEvent", row.get("event_type", ""))
        kg.add_quad(
            event_id,
            "LDIB",
            entity("BOROUGH", row.get("borough_BoroCode", "")),
            start,
            end,
            source="chicago_land_development",
            head_type="LandDevelopmentEvent",
            tail_type="Borough",
            head_name=row.get("event_type", ""),
        )


def build_temporal_land_dev_relations(
    kg: KGWriter,
    land_dev_path: Path,
    zone_cache: ZoneCache,
    quarters: list[str],
    chunk_size: int,
    max_rows: int,
) -> None:
    log("Building land development event relations: LDIB/LDIF")
    usecols = [
        "event_id",
        "event_type",
        "start_time",
        "end_time",
        "latitude",
        "longitude",
        "borough_BoroCode",
    ]
    for chunk_idx, chunk in enumerate(limited_reader(land_dev_path, chunk_size, max_rows, usecols=usecols), start=1):
        if chunk.empty:
            continue
        log(f"  land development chunk {chunk_idx}: {len(chunk):,} rows")
        chunk["event_entity"] = chunk["event_id"].map(lambda value: entity("LAND_DEV_EVENT", value))
        add_land_dev_to_borough(kg, chunk)
        add_points_to_functional_zones_by_overlap(
            kg,
            chunk,
            "event_entity",
            "LDIF",
            "latitude",
            "longitude",
            "start_time",
            "end_time",
            "LandDevelopmentEvent",
            "chicago_land_development+functional_zones",
            zone_cache,
            quarters,
        )


def load_road_change_gpkg(path: Path, max_rows: int = 0) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if max_rows:
        gdf = gdf.head(max_rows)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs).upper() != CHICAGO_CRS:
        gdf = gdf.to_crs(CHICAGO_CRS)
    gdf = gdf.loc[gdf.geometry.notna()].copy()
    gdf["event_entity"] = gdf["event_id"].map(lambda value: entity("ROAD_CHANGE_EVENT", value))
    gdf["start_iso"] = gdf["change_window_start"].map(date_only)
    gdf["end_iso"] = gdf["change_window_end"].map(date_only)
    gdf.loc[gdf["end_iso"].eq("") & gdf["start_iso"].ne(""), "end_iso"] = gdf.loc[
        gdf["end_iso"].eq("") & gdf["start_iso"].ne(""), "start_iso"
    ]
    gdf["quarter"] = pd.to_datetime(gdf["change_window_start"], errors="coerce").dt.to_period("Q").astype(str)
    return gdf


def add_road_change_to_roads(
    kg: KGWriter,
    road_change: gpd.GeoDataFrame,
    osm_to_roads: dict[str, list[str]],
) -> None:
    log("  road change -> road relations")
    for row in road_change.to_dict("records"):
        event_id = row.get("event_entity", "")
        kg.add_entity(event_id, "RoadChangeEvent", row.get("change_type", ""))
        osm_after = norm_number(row.get("osm_way_id_after", ""))
        osm_before = norm_number(row.get("osm_way_id_before", ""))
        roads = []
        for osm_id in [osm_after, osm_before]:
            for road in osm_to_roads.get(osm_id, []):
                if road not in roads:
                    roads.append(road)
        for road in roads:
            kg.add_quad(
                event_id,
                "RCIR",
                road,
                row.get("start_iso", ""),
                row.get("end_iso", ""),
                source="chicago_road_network_change+CHI_road_osm_way",
                head_type="RoadChangeEvent",
                tail_type="Road",
                head_name=row.get("change_type", ""),
            )


def add_road_change_to_borough(
    kg: KGWriter,
    road_change: gpd.GeoDataFrame,
    borough_projected: gpd.GeoDataFrame,
) -> None:
    log("  road change -> borough relations")
    joined = gpd.sjoin(
        road_change[["event_entity", "start_iso", "end_iso", "change_type", "geometry"]],
        borough_projected[["borough_entity", "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined = joined.drop_duplicates(["event_entity", "borough_entity", "start_iso", "end_iso"])
    for row in joined.itertuples(index=False):
        kg.add_quad(
            row.event_entity,
            "RCIB",
            row.borough_entity,
            row.start_iso,
            row.end_iso,
            source="chicago_road_network_change+CHI_borough",
            head_type="RoadChangeEvent",
            tail_type="Borough",
            head_name=getattr(row, "change_type", ""),
        )


def add_road_change_to_functional_zones(
    kg: KGWriter,
    road_change: gpd.GeoDataFrame,
    zone_cache: ZoneCache,
    quarters: list[str],
) -> None:
    log("  road change -> functional zone relations")
    for quarter in quarters:
        part = road_change.loc[road_change["quarter"].eq(quarter)].copy()
        if part.empty:
            continue
        zones = zone_cache.get(quarter)
        if zones is None or zones.empty:
            continue
        log(f"    RCIF {quarter}: {len(part):,} candidate rows")
        joined = gpd.sjoin(
            part[["event_entity", "start_iso", "end_iso", "change_type", "geometry"]],
            zones[["fz_entity", "geometry"]],
            how="left",
            predicate="intersects",
        )
        joined = joined.drop_duplicates(["event_entity", "fz_entity", "start_iso", "end_iso"])
        for row in joined.itertuples(index=False):
            kg.add_quad(
                row.event_entity,
                "RCIF",
                row.fz_entity,
                row.start_iso,
                row.end_iso,
                source="chicago_road_network_change+functional_zones",
                head_type="RoadChangeEvent",
                tail_type="FunctionalZone",
                head_name=getattr(row, "change_type", ""),
            )


def build_temporal_road_change_relations(
    kg: KGWriter,
    road_change_gpkg: Path,
    road_gdf: gpd.GeoDataFrame,
    borough_projected: gpd.GeoDataFrame,
    zone_cache: ZoneCache,
    quarters: list[str],
    max_rows: int,
) -> None:
    log("Building road network change event relations: RCIR/RCIB/RCIF")
    road_change = load_road_change_gpkg(road_change_gpkg, max_rows=max_rows)
    osm_to_roads = build_osm_to_road_map(road_gdf)
    log(f"  road change rows loaded: {len(road_change):,}")
    add_road_change_to_roads(kg, road_change, osm_to_roads)
    add_road_change_to_borough(kg, road_change, borough_projected)
    add_road_change_to_functional_zones(kg, road_change, zone_cache, quarters)


def build_temporal_kg_all_events(
    out_dir: Path,
    args: argparse.Namespace,
    road_gdf: gpd.GeoDataFrame,
    road_projected: gpd.GeoDataFrame,
    borough_projected: gpd.GeoDataFrame,
    area_name_to_id: dict[str, str],
) -> KGWriter:
    temporal_dir = out_dir / "temporal_kg"
    temporal_dir.mkdir(parents=True, exist_ok=True)
    zone_cache = ZoneCache(args.fz_dir)
    quarters = available_zone_quarters(args.fz_dir)
    repair_quarters = quarter_labels_between(args.start_date, args.end_date, quarters)
    raw_path = temporal_dir / "quadruples_raw.csv"
    kg = KGWriter(raw_path)
    kg.__enter__()
    try:
        add_functional_zone_background_relations(kg, args.fz_dir, area_name_to_id, args.chunksize)
        build_temporal_poi_nyc_style(
            kg,
            args.poi_event,
            road_projected,
            zone_cache,
            quarters,
            args.chunksize,
            args.max_dynamic_rows,
            args.poi_road_max_ft,
        )
        build_temporal_road_repair_nyc_style(
            kg,
            args.road_repair,
            road_projected,
            borough_projected,
            zone_cache,
            repair_quarters,
            args.chunksize,
            args.start_date,
            args.end_date,
            args.max_dynamic_rows,
            args.repair_road_buffer_ft,
            args.event_road_max_ft,
        )
        build_temporal_land_dev_relations(
            kg,
            args.land_dev,
            zone_cache,
            quarters,
            args.chunksize,
            args.max_dynamic_rows,
        )
        build_temporal_road_change_relations(
            kg,
            args.road_change_gpkg,
            road_gdf,
            borough_projected,
            zone_cache,
            quarters,
            args.max_dynamic_rows,
        )
        log(f"Temporal raw quadruples written: {kg.n_quads:,}")
    finally:
        kg.__exit__(None, None, None)
    return kg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build separated CHIUrbanKG with static KG and all-event temporal KG.")
    parser.add_argument("--poi", type=Path, default=DEFAULT_PATHS["poi"])
    parser.add_argument("--road", type=Path, default=DEFAULT_PATHS["road"])
    parser.add_argument("--borough", type=Path, default=DEFAULT_PATHS["borough"])
    parser.add_argument("--area", type=Path, default=DEFAULT_PATHS["area"])
    parser.add_argument("--poi-event", type=Path, default=DEFAULT_PATHS["poi_event"].with_name("chicago_commercial_poi_events_start_ge_20140101_with_area_borough.csv"))
    parser.add_argument("--road-repair", type=Path, default=BASE_DIR / "chicago_road_event_outputs" / "chicago_road_event_locator_20190101_20221231.csv")
    parser.add_argument("--land-dev", type=Path, default=DEFAULT_LAND_DEV)
    parser.add_argument("--road-change-gpkg", type=Path, default=DEFAULT_ROAD_CHANGE_GPKG)
    parser.add_argument("--fz-dir", type=Path, default=BASE_DIR / "chi_fz_output")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2022-12-31")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-dynamic-rows", type=int, default=0)
    parser.add_argument("--poi-road-max-ft", type=float, default=300.0)
    parser.add_argument("--event-road-max-ft", type=float, default=500.0)
    parser.add_argument("--repair-road-buffer-ft", type=float, default=75.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    from build_chi_urban_kg_nyc_style_temporal import finalize_dataset, prepare_borough_projected
    from build_chi_urban_kg_nyc_style_temporal_with_fz import build_area_name_to_id
    from build_chi_urban_kg_static_temporal_2019_2022 import build_static_kg

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in [
        args.poi,
        args.road,
        args.borough,
        args.area,
        args.poi_event,
        args.road_repair,
        args.land_dev,
        args.road_change_gpkg,
        args.fz_dir,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    log("Reading static inputs")
    area_gdf = read_wkt_gdf(args.area)
    borough_gdf = read_wkt_gdf(args.borough)
    road_gdf = load_roads(args.road)
    borough_projected = prepare_borough_projected(borough_gdf)
    area_name_to_id = build_area_name_to_id(area_gdf)

    static_kg, road_projected = build_static_kg(args.out_dir, args, area_gdf, borough_gdf, road_gdf)
    temporal_kg = build_temporal_kg_all_events(
        args.out_dir,
        args,
        road_gdf,
        road_projected,
        borough_projected,
        area_name_to_id,
    )
    finalize_dataset(args.out_dir, static_kg, temporal_kg, args)
    (args.out_dir / "README_all_events.txt").write_text(
        "CHIUrbanKG all-event temporal dataset.\n"
        "static_kg uses CHI_poi, CHI_road, CHI_area, CHI_borough.\n"
        "temporal_kg includes FunctionalZone background FHPC/FLA/BBF/FNF, POI events, road repair events, land development events, and road network change events.\n"
        "POI and road repair follow the NYC-style processing used in the recent CHI build.\n"
        "static_kg and temporal_kg share identical entity2id.csv and relation2id.csv.\n",
        encoding="utf-8",
    )
    log(f"Finished all-event CHIUrbanKG at {args.out_dir}")
    log(f"Static quadruples: {static_kg.n_quads:,}; Temporal quadruples: {temporal_kg.n_quads:,}")


if __name__ == "__main__":
    main()
