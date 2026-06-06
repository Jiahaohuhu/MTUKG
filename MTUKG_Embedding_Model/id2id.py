from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


STATIC_SPLITS = ("static_train.csv", "static_valid.csv", "static_test.csv")


def parse_area_id(value) -> Optional[str]:
    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if "::" in text:
        text = text.rsplit("::", 1)[-1]
    try:
        return str(int(text))
    except ValueError:
        return None


def entity_prefix(entity: str) -> str:
    return entity.split("::", 1)[0] if "::" in entity else entity.split("/", 1)[0].lower()


def entity_suffix(entity: str) -> str:
    if "::" in entity:
        return entity.split("::", 1)[1]
    if "/" in entity:
        return entity.split("/", 1)[1]
    return entity


def load_entity2id(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    entity_to_id: Dict[str, int] = {}
    id_to_entity: Dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("empty entity2id file: {}".format(path))
        entity_col = "entity" if "entity" in reader.fieldnames else reader.fieldnames[0]
        id_col = "entity_id" if "entity_id" in reader.fieldnames else reader.fieldnames[-1]
        for row in reader:
            entity = str(row[entity_col]).strip()
            if not entity:
                continue
            entity_id = int(str(row[id_col]).strip())
            entity_to_id[entity] = entity_id
            id_to_entity[entity_id] = entity
    return entity_to_id, id_to_entity


def load_relation2id(path: Path) -> Dict[str, int]:
    result: Dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("empty relation2id file: {}".format(path))
        relation_col = "relation" if "relation" in reader.fieldnames else reader.fieldnames[0]
        id_col = "relation_id" if "relation_id" in reader.fieldnames else reader.fieldnames[-1]
        for row in reader:
            result[str(row[relation_col]).strip()] = int(str(row[id_col]).strip())
    return result


def load_geo_area_order(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("empty geo file: {}".format(path))
        geo_col = "geo_id" if "geo_id" in reader.fieldnames else reader.fieldnames[0]
        area_ids = []
        for row in reader:
            area_id = parse_area_id(row[geo_col])
            if area_id is None:
                raise ValueError("cannot parse geo_id value {!r} in {}".format(row[geo_col], path))
            area_ids.append(area_id)
    return area_ids


def iter_static_triples(data_dir: Path) -> Iterable[Tuple[int, int, int]]:
    for split in STATIC_SPLITS:
        path = data_dir / split
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield int(row["head_id"]), int(row["relation_id"]), int(row["tail_id"])


def area_id_from_entity(entity: str) -> Optional[str]:
    if entity_prefix(entity) != "area":
        return None
    return parse_area_id(entity)


def build_area_rows(
    entity_to_id: Dict[str, int],
    geo_area_order: Optional[List[str]],
) -> Tuple[List[List[str]], int]:
    rows: List[List[str]] = []
    missing = 0

    if geo_area_order:
        for area_id in geo_area_order:
            entity = "area::{}".format(area_id)
            kg_id = entity_to_id.get(entity)
            if kg_id is None:
                missing += 1
                rows.append(["Area/{}".format(area_id), ""])
            else:
                rows.append(["Area/{}".format(area_id), str(kg_id)])
        return rows, missing

    area_entities = [
        (entity, entity_id)
        for entity, entity_id in entity_to_id.items()
        if entity_prefix(entity) == "area"
    ]
    for entity, entity_id in sorted(area_entities, key=lambda item: item[1]):
        area_id = area_id_from_entity(entity)
        if area_id is not None:
            rows.append(["Area/{}".format(area_id), str(entity_id)])
    return rows, missing


def build_entity_region_rows(
    entity_to_id: Dict[str, int],
    id_to_entity: Dict[int, str],
    relation2id: Dict[str, int],
    data_dir: Path,
    relation_name: str,
    target_prefix: str,
) -> Tuple[List[List[str]], int]:
    relation_id = relation2id[relation_name]
    entity_to_region: Dict[str, str] = {}
    duplicate_conflicts = 0

    for head_id, rel_id, tail_id in iter_static_triples(data_dir):
        if rel_id != relation_id:
            continue
        head = id_to_entity.get(head_id)
        tail = id_to_entity.get(tail_id)
        if head is None or tail is None:
            continue

        head_prefix = entity_prefix(head)
        tail_prefix = entity_prefix(tail)
        entity_name = None
        area_name = None
        if head_prefix == target_prefix and tail_prefix == "area":
            entity_name = head
            area_name = tail
        elif tail_prefix == target_prefix and head_prefix == "area":
            entity_name = tail
            area_name = head
        if entity_name is None or area_name is None:
            continue

        region_id = area_id_from_entity(area_name)
        if region_id is None:
            continue
        previous = entity_to_region.get(entity_name)
        if previous is not None and previous != region_id:
            duplicate_conflicts += 1
            continue
        entity_to_region[entity_name] = region_id

    rows = []
    for entity_name, region_id in sorted(
        entity_to_region.items(),
        key=lambda item: entity_to_id[item[0]],
    ):
        kg_id = entity_to_id[entity_name]
        rows.append([entity_name, str(kg_id), region_id])
    return rows, duplicate_conflicts


def write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Regenerate area/road/POI to current KG entity-id mapping files."
    )
    parser.add_argument("--dataset", default="NYC")
    parser.add_argument("--data_dir", default=None, type=Path)
    parser.add_argument("--entity2id", default=None, type=Path)
    parser.add_argument("--relation2id", default=None, type=Path)
    parser.add_argument(
        "--geo_file",
        default=None,
        type=Path,
        help="Optional LibCity .geo file. If set explicitly, areaid2KGid.csv is written in this geo_id order.",
    )
    parser.add_argument("--output_dir", default=script_dir, type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir or (script_dir / "data" / args.dataset)
    entity2id_path = args.entity2id or (data_dir / "entity2id.csv")
    relation2id_path = args.relation2id or (data_dir / "relation2id.csv")
    geo_file = args.geo_file

    entity_to_id, id_to_entity = load_entity2id(entity2id_path)
    relation2id = load_relation2id(relation2id_path)
    geo_area_order = load_geo_area_order(geo_file) if geo_file is not None and geo_file.exists() else None

    area_rows, missing_areas = build_area_rows(entity_to_id, geo_area_order)
    poi_rows, poi_conflicts = build_entity_region_rows(
        entity_to_id=entity_to_id,
        id_to_entity=id_to_entity,
        relation2id=relation2id,
        data_dir=data_dir,
        relation_name="PLA",
        target_prefix="point",
    )
    road_rows, road_conflicts = build_entity_region_rows(
        entity_to_id=entity_to_id,
        id_to_entity=id_to_entity,
        relation2id=relation2id,
        data_dir=data_dir,
        relation_name="RLA",
        target_prefix="road",
    )

    write_csv(args.output_dir / "areaid2KGid.csv", ["region_id", "KG_id"], area_rows)
    write_csv(args.output_dir / "POIid2KGid.csv", ["poi_id", "KG_id", "Region_id"], poi_rows)
    write_csv(args.output_dir / "roadid2KGid.csv", ["road_id", "KG_id", "Region_id"], road_rows)

    area_order_source = str(geo_file) if geo_area_order else str(entity2id_path)
    print("Wrote areaid2KGid.csv rows={}, missing_kg={}, order_source={}".format(
        len(area_rows), missing_areas, area_order_source
    ))
    print("Wrote POIid2KGid.csv rows={}, duplicate_conflicts={}".format(len(poi_rows), poi_conflicts))
    print("Wrote roadid2KGid.csv rows={}, duplicate_conflicts={}".format(len(road_rows), road_conflicts))


if __name__ == "__main__":
    main()
