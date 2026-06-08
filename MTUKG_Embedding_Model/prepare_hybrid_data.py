"""Prepare GIE static KG and TimePlex temporal KG files from CSV splits."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import shutil
from pathlib import Path
from typing import Iterable, List, Mapping

from datasets.process import process_dataset


SPLITS = ("train", "valid", "test")


def _read_csv_rows(csv_path: Path) -> List[Mapping[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            normalized = {}
            for key, value in row.items():
                if key is None:
                    continue
                normalized_key = str(key).strip().lstrip("\ufeff")
                normalized[normalized_key] = value
            rows.append(normalized)
        return rows


def _pick_column(row: Mapping[str, str], candidates: List[str]) -> str:
    for name in candidates:
        if name in row:
            return str(row[name]).strip()
    raise KeyError(f"Missing required columns. Expected one of {candidates}, got {list(row.keys())}")


def _clean_time_value(value: str) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "####-##-##"
    return text


def _write_static_split(target_path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    with target_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            head = _pick_column(row, ["head_id", "head", "lhs"])
            rel = _pick_column(row, ["relation_id", "relation", "rel"])
            tail = _pick_column(row, ["tail_id", "tail", "rhs"])
            handle.write(f"{head}\t{rel}\t{tail}\n")


def _write_temporal_split(target_path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    with target_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            head = _pick_column(row, ["head_id", "head", "lhs"])
            rel = _pick_column(row, ["relation_id", "relation", "rel"])
            tail = _pick_column(row, ["tail_id", "tail", "rhs"])
            start_time = _clean_time_value(_pick_column(row, ["start_time", "start", "start_date"]))
            end_time = _clean_time_value(_pick_column(row, ["end_time", "end", "end_date"]))
            handle.write(f"{head}\t{rel}\t{tail}\t{start_time}\t{end_time}\n")


def _materialize_static_pickles(dataset_dir: Path) -> None:
    examples, filters = process_dataset(dataset_dir)
    for split in SPLITS:
        with dataset_dir.joinpath(f"{split}.pickle").open("wb") as save_file:
            pickle.dump(examples[split], save_file)
    with dataset_dir.joinpath("to_skip.pickle").open("wb") as save_file:
        pickle.dump(filters, save_file)


def prepare_hybrid_data(dataset: str, data_path: str, timeplex_root: str, timeplex_dataset: str) -> None:
    dataset_dir = Path(data_path) / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_dir}")

    temporal_dir = Path(timeplex_root) / timeplex_dataset
    temporal_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        static_csv = dataset_dir / f"static_{split}.csv"
        temporal_csv = dataset_dir / f"temporal_{split}.csv"
        if not static_csv.exists():
            raise FileNotFoundError(f"Missing static split: {static_csv}")
        if not temporal_csv.exists():
            raise FileNotFoundError(f"Missing temporal split: {temporal_csv}")

        static_rows = _read_csv_rows(static_csv)
        temporal_rows = _read_csv_rows(temporal_csv)

        _write_static_split(dataset_dir / split, static_rows)
        _write_temporal_split(temporal_dir / f"{split}.txt", temporal_rows)

    intervals_dir = temporal_dir / "intervals"
    intervals_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(temporal_dir / "valid.txt", intervals_dir / "valid.txt")
    shutil.copyfile(temporal_dir / "test.txt", intervals_dir / "test.txt")

    _materialize_static_pickles(dataset_dir)

    print(f"Static GIE data ready under: {dataset_dir}")
    print(f"TimePlex temporal data ready under: {temporal_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare static/temporal files for GIE + TimePlex.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., NYC.")
    parser.add_argument("--data_path", default="data", help="Root path containing <dataset> folder.")
    parser.add_argument(
        "--timeplex_root",
        default=os.path.join("tkbi-master", "data"),
        help="Root path where TimePlex datasets are stored.",
    )
    parser.add_argument(
        "--timeplex_dataset",
        default=None,
        help="Output TimePlex dataset folder name. Defaults to <dataset>_temporal.",
    )
    args = parser.parse_args()

    timeplex_dataset = args.timeplex_dataset or f"{args.dataset}_temporal"
    prepare_hybrid_data(args.dataset, args.data_path, args.timeplex_root, timeplex_dataset)


if __name__ == "__main__":
    main()
