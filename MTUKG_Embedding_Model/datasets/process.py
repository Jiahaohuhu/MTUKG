"""Pre-process static KG files into pickles consumed by GIE training."""

from __future__ import annotations

import argparse
import collections
import csv
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


SPLITS = ("train", "valid", "test")


def _is_int_token(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


def _resolve_split_file(dataset_path: Path, split: str) -> Path:
    candidates = [
        dataset_path / split,
        dataset_path / f"{split}.txt",
        dataset_path / f"{split}.csv",
        dataset_path / f"static_{split}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Unable to locate split file for '{split}' under {dataset_path}. "
        f"Tried: {[str(p.name) for p in candidates]}"
    )


def _read_raw_triples(split_file: Path) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    if split_file.suffix.lower() == ".csv":
        with split_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                if {"head_id", "relation_id", "tail_id"}.issubset(row):
                    triples.append(
                        (
                            row["head_id"].strip(),
                            row["relation_id"].strip(),
                            row["tail_id"].strip(),
                        )
                    )
                    continue
                if {"head", "relation", "tail"}.issubset(row):
                    triples.append(
                        (
                            row["head"].strip(),
                            row["relation"].strip(),
                            row["tail"].strip(),
                        )
                    )
                    continue
                keys = list(row.keys())
                if len(keys) < 3:
                    continue
                triples.append(
                    (
                        str(row[keys[0]]).strip(),
                        str(row[keys[1]]).strip(),
                        str(row[keys[2]]).strip(),
                    )
                )
        return triples

    with split_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def _find_mapping_file(path: Path, stem: str) -> Path | None:
    candidates = [
        path / f"{stem}.csv",
        path / f"{stem}.txt",
        *sorted(path.glob(f"{stem}_*.csv")),
        *sorted(path.glob(f"{stem}_*.txt")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_mapping(mapping_path: Path, name_column: str, id_column: str) -> Dict[str, int]:
    if mapping_path.suffix.lower() == ".csv":
        mapping: Dict[str, int] = {}
        with mapping_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return mapping
            name_key = name_column if name_column in reader.fieldnames else reader.fieldnames[0]
            id_key = id_column if id_column in reader.fieldnames else reader.fieldnames[-1]
            for row in reader:
                if row[name_key] is None or row[id_key] is None:
                    continue
                mapping[str(row[name_key]).strip()] = int(str(row[id_key]).strip())
        return mapping

    mapping = {}
    with mapping_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            mapping[parts[0]] = int(parts[1])
    return mapping


def _save_embedding_indices(path: Path, filename: str, max_index: int) -> None:
    indices = np.arange(max_index + 1, dtype=np.int64)
    np.savetxt(path / filename, indices, encoding="utf-8", delimiter=",", fmt="%d")


def load_explicit_mappings(path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    entity_mapping_path = _find_mapping_file(path, "entity2id")
    relation_mapping_path = _find_mapping_file(path, "relation2id")
    if entity_mapping_path is None or relation_mapping_path is None:
        raise FileNotFoundError("Missing entity2id / relation2id mapping files.")

    ent2idx = _load_mapping(entity_mapping_path, "entity", "entity_id")
    rel2idx = _load_mapping(relation_mapping_path, "relation", "relation_id")
    return ent2idx, rel2idx


def _build_fallback_mapping(raw_examples: Iterable[Tuple[str, str, str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    entities, relations = set(), set()
    for lhs, rel, rhs in raw_examples:
        entities.add(lhs)
        entities.add(rhs)
        relations.add(rel)
    ent2idx = {entity: idx for idx, entity in enumerate(sorted(entities))}
    rel2idx = {relation: idx for idx, relation in enumerate(sorted(relations))}
    return ent2idx, rel2idx


def _to_idx(token: str, mapping: Optional[Dict[str, int]]) -> int:
    if mapping is not None and token in mapping:
        return mapping[token]
    if _is_int_token(token):
        return int(token)
    if mapping is not None:
        raise KeyError(f"Token '{token}' is missing from mapping.")
    raise ValueError(f"Token '{token}' is not numeric and no mapping is available.")


def to_np_array(
    raw_examples: Iterable[Tuple[str, str, str]],
    ent2idx: Optional[Dict[str, int]],
    rel2idx: Optional[Dict[str, int]],
) -> np.ndarray:
    """Map raw triples to a numpy array with integer ids."""
    examples = []
    for lhs, rel, rhs in raw_examples:
        examples.append([_to_idx(lhs, ent2idx), _to_idx(rel, rel2idx), _to_idx(rhs, ent2idx)])
    return np.array(examples, dtype="int64")


def get_filters(examples: Iterable[Iterable[int]], n_relations: int):
    """Create filtering lists for evaluation."""
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for lhs, rel, rhs in examples:
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)
    lhs_final = {key: sorted(values) for key, values in lhs_filters.items()}
    rhs_final = {key: sorted(values) for key, values in rhs_filters.items()}
    return lhs_final, rhs_final


def process_dataset(path: str | Path, dataset_name: str | None = None):
    """Map entities/relations to ids and save corresponding pickle arrays."""
    del dataset_name  # kept for backward compatibility with older callers

    dataset_path = Path(path)
    split_files = {split: _resolve_split_file(dataset_path, split) for split in SPLITS}
    raw_examples = {split: _read_raw_triples(split_files[split]) for split in SPLITS}

    ent2idx: Optional[Dict[str, int]]
    rel2idx: Optional[Dict[str, int]]
    try:
        ent2idx, rel2idx = load_explicit_mappings(dataset_path)
    except FileNotFoundError:
        ent2idx, rel2idx = None, None

    if ent2idx is None or rel2idx is None:
        all_raw = [triple for split in SPLITS for triple in raw_examples[split]]
        all_numeric = all(
            _is_int_token(lhs) and _is_int_token(rel) and _is_int_token(rhs)
            for lhs, rel, rhs in all_raw
        )
        if not all_numeric:
            ent2idx, rel2idx = _build_fallback_mapping(all_raw)

    examples = {}
    for split in SPLITS:
        examples[split] = to_np_array(raw_examples[split], ent2idx, rel2idx)

    all_examples = np.concatenate([examples["train"], examples["valid"], examples["test"]], axis=0)
    max_entity_id = int(max(np.max(all_examples[:, 0]), np.max(all_examples[:, 2])))
    max_relation_id = int(np.max(all_examples[:, 1]))
    _save_embedding_indices(dataset_path, "entity_idx_embedding.csv", max_entity_id)
    _save_embedding_indices(dataset_path, "relations_idx_embeddings.csv", max_relation_id)

    relation_count = len(rel2idx) if rel2idx is not None else (max_relation_id + 1)
    lhs_skip, rhs_skip = get_filters(all_examples, relation_count)
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}
    return examples, filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Process static KG text files for GIE training.")
    parser.add_argument("--data_path", default="data", help="Root folder containing dataset directories.")
    parser.add_argument("--dataset", required=True, help="Dataset name under data_path.")
    args = parser.parse_args()

    dataset_path = Path(args.data_path) / args.dataset
    dataset_examples, dataset_filters = process_dataset(dataset_path, args.dataset)

    for dataset_split in SPLITS:
        save_path = dataset_path / f"{dataset_split}.pickle"
        with save_path.open("wb") as save_file:
            pickle.dump(dataset_examples[dataset_split], save_file)

    with dataset_path.joinpath("to_skip.pickle").open("wb") as save_file:
        pickle.dump(dataset_filters, save_file)


if __name__ == "__main__":
    main()
