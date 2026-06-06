"""Export fused embeddings from static GIE and temporal TimePlex checkpoints."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import pickle
import re
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import models
from datasets.kg_dataset import KGDataset
from datasets.process import process_dataset

TIMEPLEX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tkbi-master")
if os.path.isdir(TIMEPLEX_ROOT) and TIMEPLEX_ROOT not in sys.path:
    sys.path.append(TIMEPLEX_ROOT)


def torch_load(path: str, *, allow_pickle: bool = False) -> Any:
    if allow_pickle:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    return torch.load(path, map_location="cpu")


def ensure_static_dataset_ready(dataset_path: str) -> None:
    required = ["train.pickle", "valid.pickle", "test.pickle", "to_skip.pickle"]
    if all(os.path.exists(os.path.join(dataset_path, name)) for name in required):
        return

    examples, filters = process_dataset(dataset_path)
    for split in ["train", "valid", "test"]:
        with open(os.path.join(dataset_path, f"{split}.pickle"), "wb") as save_file:
            pickle.dump(examples[split], save_file)
    with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
        pickle.dump(filters, save_file)


def load_entity_name_to_id(dataset_path: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    csv_path = os.path.join(dataset_path, "entity2id.csv")
    txt_path = os.path.join(dataset_path, "entity2id.txt")

    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return mapping
            name_col = "entity" if "entity" in reader.fieldnames else reader.fieldnames[0]
            id_col = "entity_id" if "entity_id" in reader.fieldnames else reader.fieldnames[-1]
            for row in reader:
                try:
                    mapping[str(row[name_col]).strip()] = int(str(row[id_col]).strip())
                except (TypeError, ValueError):
                    continue
        return mapping

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                parts = raw_line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    mapping[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    return mapping


def load_gie_embeddings(
    dataset_path: str,
    gie_checkpoint: str,
    gie_config: Optional[str],
    model_name: str,
) -> np.ndarray:
    ensure_static_dataset_ready(dataset_path)
    dataset = KGDataset(dataset_path, debug=False)
    sizes = dataset.get_shape()

    state = torch_load(gie_checkpoint)
    if "entity.weight" not in state:
        raise ValueError(f"Checkpoint does not look like a GIE/AttH state_dict: {gie_checkpoint}")

    config = {}
    if gie_config is not None:
        with open(gie_config, "r", encoding="utf-8") as handle:
            config = json.load(handle)

    checkpoint_multi_c = any(
        key in state and getattr(state[key], "shape", None) is not None and len(state[key].shape) > 0 and state[key].shape[0] > 1
        for key in ("c", "c1", "c2")
    )
    multi_c = bool(config.get("multi_c", checkpoint_multi_c))
    if checkpoint_multi_c and not multi_c:
        print("Warning: overriding multi_c=True because checkpoint curvature tensors are relation-specific.")
        multi_c = True

    rank = int(state["entity.weight"].shape[1])
    gie_args = SimpleNamespace(
        sizes=sizes,
        rank=rank,
        dropout=float(config.get("dropout", 0.0)),
        gamma=float(config.get("gamma", 0.0)),
        dtype=str(config.get("dtype", "double")),
        bias=str(config.get("bias", "constant")),
        init_size=float(config.get("init_size", 1e-3)),
        multi_c=multi_c,
    )
    model_cls = getattr(models, model_name)
    model = model_cls(gie_args)
    model.load_state_dict(state, strict=True)
    model.eval()

    entity_embeddings = model.entity.weight.detach().cpu().numpy()
    idx_path = os.path.join(dataset_path, "entity_idx_embedding.csv")
    if not os.path.exists(idx_path):
        return entity_embeddings

    idx = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    idx = np.atleast_1d(idx).astype(np.int64)
    if len(idx) != entity_embeddings.shape[0]:
        return entity_embeddings

    max_idx = int(np.max(idx))
    aligned = np.zeros((max_idx + 1, entity_embeddings.shape[1]), dtype=entity_embeddings.dtype)
    for local_id, global_id in enumerate(idx):
        if global_id >= 0:
            aligned[int(global_id)] = entity_embeddings[local_id]
    return aligned


def _parse_global_entity_id(token: str, entity_name_to_id: Dict[str, int]) -> Optional[int]:
    text = str(token).strip()
    if text in entity_name_to_id:
        return entity_name_to_id[text]
    try:
        return int(text)
    except ValueError:
        return None


def load_timeplex_entity_embeddings(
    checkpoint_path: str,
    entity_name_to_id: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    state = torch_load(checkpoint_path, allow_pickle=True)
    model_weights = state.get("model_weights", state)
    e_re, e_im = _load_timeplex_complex_entity_weights(model_weights)

    temporal_raw = np.concatenate([e_re, e_im], axis=1)
    aligned_pairs: List[Tuple[int, int]] = []

    datamap = state.get("datamap")
    entity_map = getattr(datamap, "entity_map", None) if datamap is not None else None
    if not entity_map:
        entity_map = state.get("entity_map")
    if entity_map:
        for token, local_id in entity_map.items():
            if local_id < 0 or local_id >= temporal_raw.shape[0]:
                continue
            global_id = _parse_global_entity_id(token, entity_name_to_id)
            if global_id is None or global_id < 0:
                continue
            aligned_pairs.append((int(global_id), int(local_id)))
    else:
        # Fallback: assume local ids are globally valid ids.
        for local_id in range(temporal_raw.shape[0]):
            aligned_pairs.append((int(local_id), int(local_id)))

    if not aligned_pairs:
        return np.zeros((0, temporal_raw.shape[1]), dtype=temporal_raw.dtype), np.zeros((0,), dtype=np.bool_)

    max_global_id = max(global_id for global_id, _ in aligned_pairs)
    aligned = np.zeros((max_global_id + 1, temporal_raw.shape[1]), dtype=temporal_raw.dtype)
    mask = np.zeros(max_global_id + 1, dtype=np.bool_)
    for global_id, local_id in aligned_pairs:
        aligned[global_id] = temporal_raw[local_id]
        mask[global_id] = True
    return aligned, mask


def _weight_to_numpy(model_weights: Dict[str, torch.Tensor], *names: str) -> Optional[np.ndarray]:
    for name in names:
        value = model_weights.get(name)
        if value is not None:
            return value.detach().cpu().numpy()
    return None


def _load_timeplex_complex_entity_weights(model_weights: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    e_re = _weight_to_numpy(model_weights, "base_model.E_re.weight", "E_re.weight")
    e_im = _weight_to_numpy(model_weights, "base_model.E_im.weight", "E_im.weight")
    if e_re is None or e_im is None:
        raise ValueError(
            "TimePlex checkpoint is missing entity complex embeddings "
            "(`base_model.E_re/E_im.weight` or `E_re/E_im.weight`)."
        )
    return e_re, e_im


def _load_timeplex_complex_time_weights(
    model_weights: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts_re = _weight_to_numpy(model_weights, "base_model.Ts_re.weight", "Ts_re.weight")
    ts_im = _weight_to_numpy(model_weights, "base_model.Ts_im.weight", "Ts_im.weight")
    to_re = _weight_to_numpy(model_weights, "base_model.To_re.weight", "To_re.weight")
    to_im = _weight_to_numpy(model_weights, "base_model.To_im.weight", "To_im.weight")

    if ts_re is not None and ts_im is not None:
        if to_re is not None and to_im is not None:
            time_re = 0.5 * (ts_re + to_re)
            time_im = 0.5 * (ts_im + to_im)
            raw_time = np.concatenate([ts_re, ts_im, to_re, to_im], axis=1)
        else:
            time_re = ts_re
            time_im = ts_im
            raw_time = np.concatenate([ts_re, ts_im], axis=1)
        return time_re, time_im, raw_time

    t_re = _weight_to_numpy(model_weights, "base_model.T_re.weight", "T_re.weight")
    t_im = _weight_to_numpy(model_weights, "base_model.T_im.weight", "T_im.weight")
    if t_re is None or t_im is None:
        raise ValueError(
            "TimePlex checkpoint is missing time complex embeddings "
            "(`Ts_re/Ts_im`, `To_re/To_im`, or `T_re/T_im`)."
        )
    return t_re, t_im, np.concatenate([t_re, t_im], axis=1)


def _time_count_from_datamap(datamap: Any) -> Optional[int]:
    if datamap is None:
        return None
    if getattr(datamap, "use_time_interval", False):
        year2id = getattr(datamap, "year2id", None)
        if year2id:
            return len(year2id)
    date_year2id = getattr(datamap, "dateYear2id", None)
    if date_year2id:
        return len(date_year2id)
    return None


def _is_sentinel_time(value: Any, datamap: Any = None) -> bool:
    text = str(value)
    if text in {"UNK-TIME", "0", "3000", "####-##-##", "####-##-##\t####-##-##"}:
        return True
    if datamap is not None:
        try:
            numeric = int(value)
            min_value = datamap.min_time_value() if hasattr(datamap, "min_time_value") else None
            max_value = datamap.max_time_value() if hasattr(datamap, "max_time_value") else None
            return numeric == min_value or numeric == max_value
        except (TypeError, ValueError):
            return False
    return False


def _parse_date_parts(value: Any) -> Optional[Tuple[int, Optional[int], Optional[int]]]:
    match = re.search(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", str(value))
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) is not None else None
    day = int(match.group(3)) if match.group(3) is not None else None
    return year, month, day


def _quarter_from_month(month: Optional[int]) -> Optional[int]:
    if month is None or month < 1 or month > 12:
        return None
    return (month - 1) // 3 + 1


def _quarter_key(year: int, quarter: int) -> str:
    return f"{year}-Q{quarter}"


def _month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _decode_compact_month(value: Any) -> Optional[Tuple[int, int]]:
    try:
        month_index = int(value)
    except (TypeError, ValueError):
        return None
    year = month_index // 12
    month = month_index % 12 + 1
    if month < 1 or month > 12:
        return None
    return int(year), int(month)


def _decode_month_like_value(value: Any, source_time_granularity: str = "year") -> Optional[Tuple[int, int]]:
    if source_time_granularity == "month":
        decoded = _decode_compact_month(value)
        if decoded is not None:
            return decoded
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None and numeric > 10000:
        decoded = _decode_compact_month(numeric)
        if decoded is not None and 1800 <= decoded[0] <= 2200:
            return decoded
    parsed = _parse_date_parts(value)
    if parsed is not None and parsed[1] is not None:
        return int(parsed[0]), int(parsed[1])
    return None


def _is_valid_calendar_year(year: Optional[int]) -> bool:
    if year is None:
        return False
    try:
        year_int = int(year)
    except (TypeError, ValueError):
        return False
    return 1 <= year_int <= 9999


def _date_from_parts(parts: Optional[Tuple[int, Optional[int], Optional[int]]]) -> Optional[datetime.date]:
    if parts is None:
        return None
    year, month, day = parts
    if month is None or day is None:
        return None
    try:
        return datetime.date(int(year), int(month), int(day))
    except ValueError:
        return None


def _date_key(value: datetime.date) -> str:
    return value.isoformat()


def _iter_dates(start_date: datetime.date, end_date: datetime.date):
    if end_date < start_date:
        end_date = start_date
    current = start_date
    one_day = datetime.timedelta(days=1)
    while current <= end_date:
        yield current
        current += one_day


def _iter_year_dates(year: int):
    start_date = datetime.date(int(year), 1, 1)
    end_date = datetime.date(int(year), 12, 31)
    yield from _iter_dates(start_date, end_date)


def _iter_months(start_year: int, start_month: int, end_year: int, end_month: int):
    start = start_year * 12 + (start_month - 1)
    end = end_year * 12 + (end_month - 1)
    if end < start:
        end = start
    for value in range(start, end + 1):
        year = value // 12
        month = value % 12 + 1
        yield year, month


def _iter_quarters(start_year: int, start_quarter: int, end_year: int, end_quarter: int):
    start = start_year * 4 + (start_quarter - 1)
    end = end_year * 4 + (end_quarter - 1)
    if end < start:
        end = start
    for value in range(start, end + 1):
        year = value // 4
        quarter = value % 4 + 1
        yield year, quarter


def _build_month_time_records(datamap: Any, raw_time_count: int, time_filter: str) -> List[Dict[str, Any]]:
    id2date_year = getattr(datamap, "id2dateYear", None) if datamap is not None else None
    datamap_count = _time_count_from_datamap(datamap)
    usable_count = min(raw_time_count, datamap_count) if datamap_count is not None else max(raw_time_count - 2, 0)
    source_time_granularity = getattr(datamap, "time_granularity", "year") if datamap is not None else "year"

    records: List[Dict[str, Any]] = []
    for original_id in range(usable_count):
        value = id2date_year.get(original_id, original_id) if id2date_year else original_id
        if time_filter == "observed" and _is_sentinel_time(value, datamap):
            continue

        decoded = None if _is_sentinel_time(value, datamap) else _decode_month_like_value(value, source_time_granularity)
        if decoded is not None:
            year, month = decoded
            records.append(
                {
                    "index": len(records),
                    "timeplex_time_id": int(original_id),
                    "time": _month_key(year, month),
                    "kind": "date_month",
                    "year": int(year),
                    "month": int(month),
                    "source_granularity": source_time_granularity,
                }
            )
            continue

        year = None
        if not _is_sentinel_time(value, datamap):
            try:
                year = int(value)
            except (TypeError, ValueError):
                parsed = _parse_date_parts(value)
                year = parsed[0] if parsed is not None else None
        if not _is_valid_calendar_year(year):
            if time_filter == "all":
                records.append(
                    {
                        "index": len(records),
                        "timeplex_time_id": int(original_id),
                        "time": str(value),
                        "kind": "sentinel_time",
                        "source_granularity": source_time_granularity,
                    }
                )
            continue
        for month in range(1, 13):
            records.append(
                {
                    "index": len(records),
                    "timeplex_time_id": int(original_id),
                    "time": _month_key(year, month),
                    "kind": "date_month",
                    "year": int(year),
                    "month": int(month),
                    "source_granularity": source_time_granularity,
                }
            )
    return records


def _build_quarter_time_records(datamap: Any, raw_time_count: int, time_filter: str) -> List[Dict[str, Any]]:
    id2_time_str = getattr(datamap, "id2TimeStr", None) if datamap is not None else None
    quarter_to_time_ids: Dict[Tuple[int, int], List[int]] = {}

    use_time_strings = False
    if id2_time_str:
        try:
            use_time_strings = max(int(item) for item in id2_time_str.keys()) < raw_time_count
        except (TypeError, ValueError):
            use_time_strings = False

    if use_time_strings:
        for original_id, time_text in id2_time_str.items():
            original_id = int(original_id)
            if original_id < 0 or original_id >= raw_time_count:
                continue
            if time_filter == "observed" and _is_sentinel_time(time_text):
                continue
            parts = str(time_text).split("\t")
            start_parts = _parse_date_parts(parts[0]) if parts else None
            end_parts = _parse_date_parts(parts[-1]) if len(parts) > 1 else start_parts
            if start_parts is None:
                continue
            if end_parts is None:
                end_parts = start_parts
            start_year, start_month, _ = start_parts
            end_year, end_month, _ = end_parts
            start_quarter = _quarter_from_month(start_month) or 1
            end_quarter = _quarter_from_month(end_month) or 4
            for year, quarter in _iter_quarters(start_year, start_quarter, end_year, end_quarter):
                quarter_to_time_ids.setdefault((year, quarter), []).append(original_id)

    if quarter_to_time_ids:
        records = []
        for year, quarter in sorted(quarter_to_time_ids):
            time_ids = sorted(set(quarter_to_time_ids[(year, quarter)]))
            records.append(
                {
                    "index": len(records),
                    "timeplex_time_id": int(time_ids[0]),
                    "timeplex_time_ids": [int(item) for item in time_ids],
                    "time": _quarter_key(year, quarter),
                    "kind": "date_quarter",
                    "year": int(year),
                    "quarter": int(quarter),
                    "source_granularity": "time_string",
                }
            )
        return records

    id2date_year = getattr(datamap, "id2dateYear", None) if datamap is not None else None
    datamap_count = _time_count_from_datamap(datamap)
    usable_count = min(raw_time_count, datamap_count) if datamap_count is not None else max(raw_time_count - 2, 0)
    source_time_granularity = getattr(datamap, "time_granularity", "year") if datamap is not None else "year"
    records = []
    for original_id in range(usable_count):
        value = id2date_year.get(original_id, original_id) if id2date_year else original_id
        if time_filter == "observed" and _is_sentinel_time(value, datamap):
            continue

        decoded = None if _is_sentinel_time(value, datamap) else _decode_month_like_value(value, source_time_granularity)
        if decoded is not None:
            year, month = decoded
            quarter = _quarter_from_month(month)
            if quarter is not None:
                records.append(
                    {
                        "index": len(records),
                        "timeplex_time_id": int(original_id),
                        "time": _quarter_key(year, quarter),
                        "kind": "date_quarter",
                        "year": int(year),
                        "quarter": int(quarter),
                        "source_granularity": source_time_granularity,
                    }
                )
                continue

        year = None
        if not _is_sentinel_time(value, datamap):
            try:
                year = int(value)
            except (TypeError, ValueError):
                parsed = _parse_date_parts(value)
                year = parsed[0] if parsed is not None else None
        if not _is_valid_calendar_year(year):
            continue
        for quarter in range(1, 5):
            records.append(
                {
                    "index": len(records),
                    "timeplex_time_id": int(original_id),
                    "time": _quarter_key(year, quarter),
                    "kind": "date_quarter",
                    "year": int(year),
                    "quarter": int(quarter),
                    "source_granularity": source_time_granularity,
                }
            )
    return records


def _build_day_time_records(datamap: Any, raw_time_count: int, time_filter: str) -> List[Dict[str, Any]]:
    id2_time_str = getattr(datamap, "id2TimeStr", None) if datamap is not None else None
    day_to_time_ids: Dict[datetime.date, List[int]] = {}

    use_time_strings = False
    if id2_time_str:
        try:
            use_time_strings = max(int(item) for item in id2_time_str.keys()) < raw_time_count
        except (TypeError, ValueError):
            use_time_strings = False

    if use_time_strings:
        for original_id, time_text in id2_time_str.items():
            original_id = int(original_id)
            if original_id < 0 or original_id >= raw_time_count:
                continue
            if time_filter == "observed" and _is_sentinel_time(time_text):
                continue
            parts = str(time_text).split("\t")
            start_date = _date_from_parts(_parse_date_parts(parts[0])) if parts else None
            end_date = _date_from_parts(_parse_date_parts(parts[-1])) if len(parts) > 1 else start_date
            if start_date is None:
                continue
            if end_date is None:
                end_date = start_date
            for current in _iter_dates(start_date, end_date):
                day_to_time_ids.setdefault(current, []).append(original_id)

    if day_to_time_ids:
        records = []
        for current in sorted(day_to_time_ids):
            time_ids = sorted(set(day_to_time_ids[current]))
            records.append(
                {
                    "index": len(records),
                    "timeplex_time_id": int(time_ids[0]),
                    "timeplex_time_ids": [int(item) for item in time_ids],
                    "time": _date_key(current),
                    "kind": "date_day",
                    "year": int(current.year),
                    "month": int(current.month),
                    "day": int(current.day),
                    "source_granularity": "time_string",
                }
            )
        return records

    id2date_year = getattr(datamap, "id2dateYear", None) if datamap is not None else None
    datamap_count = _time_count_from_datamap(datamap)
    usable_count = min(raw_time_count, datamap_count) if datamap_count is not None else max(raw_time_count - 2, 0)
    source_time_granularity = getattr(datamap, "time_granularity", "year") if datamap is not None else "year"
    records = []
    for original_id in range(usable_count):
        value = id2date_year.get(original_id, original_id) if id2date_year else original_id
        if time_filter == "observed" and _is_sentinel_time(value, datamap):
            continue

        decoded = None if _is_sentinel_time(value, datamap) else _decode_month_like_value(value, source_time_granularity)
        if decoded is not None:
            year, month = decoded
            try:
                month_start = datetime.date(int(year), int(month), 1)
                month_end = _month_end(int(year), int(month))
            except ValueError:
                continue
            for current in _iter_dates(month_start, month_end):
                records.append(
                    {
                        "index": len(records),
                        "timeplex_time_id": int(original_id),
                        "time": _date_key(current),
                        "kind": "date_day",
                        "year": int(current.year),
                        "month": int(current.month),
                        "day": int(current.day),
                        "source_granularity": source_time_granularity,
                    }
                )
            continue

        year = None
        if not _is_sentinel_time(value, datamap):
            try:
                year = int(value)
            except (TypeError, ValueError):
                parsed = _parse_date_parts(value)
                year = parsed[0] if parsed is not None else None
        if not _is_valid_calendar_year(year):
            continue
        for current in _iter_year_dates(year):
            records.append(
                {
                    "index": len(records),
                    "timeplex_time_id": int(original_id),
                    "time": _date_key(current),
                    "kind": "date_day",
                    "year": int(current.year),
                    "month": int(current.month),
                    "day": int(current.day),
                    "source_granularity": source_time_granularity,
                }
            )
    return records


def _build_time_records(
    datamap: Any,
    raw_time_count: int,
    time_filter: str,
    time_granularity: str = "year",
) -> List[Dict[str, Any]]:
    if time_granularity == "day":
        return _build_day_time_records(datamap, raw_time_count, time_filter)
    if time_granularity == "month":
        return _build_month_time_records(datamap, raw_time_count, time_filter)
    if time_granularity == "quarter":
        return _build_quarter_time_records(datamap, raw_time_count, time_filter)

    datamap_count = _time_count_from_datamap(datamap)
    usable_count = min(raw_time_count, datamap_count) if datamap_count is not None else max(raw_time_count - 2, 0)

    records: List[Dict[str, Any]] = []
    if datamap is not None and getattr(datamap, "use_time_interval", False) and getattr(datamap, "year2id", None):
        for interval, original_id in sorted(datamap.year2id.items(), key=lambda item: item[1]):
            if int(original_id) >= usable_count:
                continue
            start, end = interval
            if time_filter == "observed" and (_is_sentinel_time(start, datamap) or _is_sentinel_time(end, datamap)):
                continue
            records.append(
                {
                    "index": len(records),
                    "timeplex_time_id": int(original_id),
                    "time": f"{start}-{end}",
                    "start": int(start),
                    "end": int(end),
                    "kind": "interval_bin",
                }
            )
        return records

    id2date_year = getattr(datamap, "id2dateYear", None) if datamap is not None else None
    source_time_granularity = getattr(datamap, "time_granularity", "year") if datamap is not None else "year"
    for original_id in range(usable_count):
        value = id2date_year.get(original_id, original_id) if id2date_year else original_id
        if time_filter == "observed" and _is_sentinel_time(value, datamap):
            continue

        decoded = None if _is_sentinel_time(value, datamap) else _decode_month_like_value(value, source_time_granularity)
        if decoded is not None:
            year, month = decoded
            records.append(
                {
                    "index": len(records),
                    "timeplex_time_id": int(original_id),
                    "time": _month_key(year, month),
                    "kind": "date_month",
                    "year": int(year),
                    "month": int(month),
                    "source_granularity": source_time_granularity,
                }
            )
            continue

        record: Dict[str, Any] = {
            "index": len(records),
            "timeplex_time_id": int(original_id),
            "time": str(value),
            "kind": "date_year",
        }
        if not _is_sentinel_time(value, datamap):
            try:
                year = int(value)
            except (TypeError, ValueError):
                parsed = _parse_date_parts(value)
                year = parsed[0] if parsed is not None else None
            if _is_valid_calendar_year(year):
                record["year"] = int(year)
        record["source_granularity"] = source_time_granularity
        records.append(record)
    return records


def _build_global_to_local_entity_map(
    state: Dict[str, Any],
    entity_name_to_id: Dict[str, int],
    entity_count: int,
) -> Dict[int, int]:
    datamap = state.get("datamap")
    entity_map = getattr(datamap, "entity_map", None) if datamap is not None else None
    if not entity_map:
        entity_map = state.get("entity_map")

    global_to_local: Dict[int, int] = {}
    if entity_map:
        for token, local_id in entity_map.items():
            if local_id < 0 or local_id >= entity_count:
                continue
            global_id = _parse_global_entity_id(token, entity_name_to_id)
            if global_id is not None and global_id >= 0:
                global_to_local[int(global_id)] = int(local_id)
        return global_to_local

    for local_id in range(entity_count):
        global_to_local[int(local_id)] = int(local_id)
    return global_to_local


def _read_mapping_rows(mapping_csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(mapping_csv_path):
        return []
    with open(mapping_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _parse_area_id(value: Any) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if "::" in text:
        text = text.rsplit("::", 1)[-1]
    try:
        return int(text)
    except ValueError:
        return None


def _read_region_rows_from_geo(region_geo_file: str) -> List[Dict[str, str]]:
    if not region_geo_file:
        return []
    if not os.path.exists(region_geo_file):
        raise FileNotFoundError(f"Region geo file not found: {region_geo_file}")

    rows: List[Dict[str, str]] = []
    with open(region_geo_file, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return rows
        geo_col = "geo_id" if "geo_id" in reader.fieldnames else reader.fieldnames[0]
        for index, row in enumerate(reader):
            area_id = _parse_area_id(row.get(geo_col, ""))
            if area_id is None:
                raise ValueError(
                    f"Cannot parse geo_id at row {index} in {region_geo_file}: {row.get(geo_col, '')}"
                )
            rows.append(
                {
                    "region_id": f"Area/{area_id}",
                    "geo_id": str(area_id),
                    "geo_index": str(index),
                    "KG_id": "",
                }
            )
    return rows


def load_region_rows_for_export(mapping_csv_path: str, region_geo_file: str = "") -> Tuple[List[Dict[str, str]], str]:
    if region_geo_file:
        rows = _read_region_rows_from_geo(region_geo_file)
        if not rows:
            raise ValueError(f"No regions found in region geo file: {region_geo_file}")
        return rows, region_geo_file
    return _read_mapping_rows(mapping_csv_path), mapping_csv_path


def load_relation_name_to_id(dataset_path: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    path = os.path.join(dataset_path, "relation2id.csv")
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return mapping
        name_col = "relation" if "relation" in reader.fieldnames else reader.fieldnames[0]
        id_col = "relation_id" if "relation_id" in reader.fieldnames else reader.fieldnames[-1]
        for row in reader:
            try:
                mapping[str(row[name_col]).strip()] = int(str(row[id_col]).strip())
            except (TypeError, ValueError):
                continue
    return mapping


def _mapping_entity_candidates(value: Any) -> List[str]:
    text = str(value).strip()
    if not text:
        return []
    candidates = [text]
    lowered = text.lower()
    candidates.append(lowered)
    if "/" in text:
        prefix, suffix = text.split("/", 1)
        prefix_lower = prefix.lower()
        candidates.append(f"{prefix_lower}::{suffix}")
        if prefix_lower == "poi":
            candidates.append(f"point::static_{suffix}")
        if prefix_lower == "area":
            candidates.append(f"area::{suffix}")
        if prefix_lower == "road":
            candidates.append(f"road::{suffix}")
    return list(dict.fromkeys(candidates))


def _resolve_mapping_kg_id(
    row: Dict[str, str],
    entity_name_to_id: Optional[Dict[str, int]] = None,
    entity_fields: Optional[List[str]] = None,
) -> Optional[int]:
    if entity_name_to_id is not None and entity_fields is not None:
        for field in entity_fields:
            if field not in row:
                continue
            for candidate in _mapping_entity_candidates(row[field]):
                if candidate in entity_name_to_id:
                    return int(entity_name_to_id[candidate])

    try:
        return int(str(row["KG_id"]).strip())
    except (KeyError, TypeError, ValueError):
        return None


def _region_keys(value: Any) -> List[str]:
    text = str(value).strip()
    if not text:
        return []
    keys = [text, text.lower()]
    if "/" in text:
        prefix, suffix = text.split("/", 1)
        if prefix.lower() == "area":
            keys.extend([suffix, f"area::{suffix}", f"Area/{suffix}"])
    elif text.isdigit():
        keys.extend([f"area::{text}", f"Area/{text}"])
    return list(dict.fromkeys(keys))


def _build_region_lookup(
    region_rows: List[Dict[str, str]],
    entity_name_to_id: Dict[str, int],
) -> Tuple[Dict[int, int], Dict[str, int], List[Dict[str, Any]]]:
    kg_id_to_index: Dict[int, int] = {}
    key_to_index: Dict[str, int] = {}
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(region_rows):
        region_id = str(row.get("region_id", row.get("Region_id", index))).strip()
        kg_id = _resolve_mapping_kg_id(row, entity_name_to_id, ["region_id", "Region_id"])
        if kg_id is not None:
            kg_id_to_index[int(kg_id)] = index
        for key in _region_keys(region_id):
            key_to_index[key] = index
        records.append({"index": index, "region_id": region_id, "kg_id": kg_id})
    return kg_id_to_index, key_to_index, records


def _resolve_region_index(region_value: Any, region_key_to_index: Dict[str, int]) -> Optional[int]:
    for key in _region_keys(region_value):
        if key in region_key_to_index:
            return region_key_to_index[key]
    return None


def _load_entity_to_region_index(
    mapping_csv_path: str,
    entity_name_to_id: Dict[str, int],
    region_key_to_index: Dict[str, int],
    entity_fields: List[str],
) -> Dict[int, int]:
    result: Dict[int, int] = {}
    rows = _read_mapping_rows(mapping_csv_path)
    for row in rows:
        entity_id = _resolve_mapping_kg_id(row, entity_name_to_id, entity_fields)
        region_index = _resolve_region_index(row.get("Region_id", row.get("region_id", "")), region_key_to_index)
        if entity_id is not None and region_index is not None:
            result[int(entity_id)] = int(region_index)
    return result


def _parse_relation_filter(text: str, relation_name_to_id: Dict[str, int]) -> Optional[set]:
    if not text:
        return None
    result = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item in relation_name_to_id:
            result.add(int(relation_name_to_id[item]))
            continue
        try:
            result.add(int(item))
        except ValueError:
            raise ValueError(f"Unknown relation in --event_relation_filter: {item}")
    return result


def _parse_date(value: Any) -> Optional[datetime.date]:
    parts = _parse_date_parts(value)
    return _date_from_parts(parts)


def _month_end(year: int, month: int) -> datetime.date:
    if month == 12:
        return datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
    return datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)


def _time_record_date_range(record: Dict[str, Any]) -> Optional[Tuple[datetime.date, datetime.date]]:
    year = record.get("year")
    month = record.get("month")
    day = record.get("day")
    quarter = record.get("quarter")
    if year is None:
        year = _parse_date_parts(record.get("time", ""))[0] if _parse_date_parts(record.get("time", "")) else None
    if year is None:
        return None
    year = int(year)
    if day is not None and month is not None:
        current = datetime.date(year, int(month), int(day))
        return current, current
    if month is not None:
        month = int(month)
        return datetime.date(year, month, 1), _month_end(year, month)
    if quarter is not None:
        start_month = (int(quarter) - 1) * 3 + 1
        end_month = start_month + 2
        return datetime.date(year, start_month, 1), _month_end(year, end_month)
    if "start" in record and "end" in record:
        return datetime.date(int(record["start"]), 1, 1), datetime.date(int(record["end"]), 12, 31)
    return datetime.date(year, 1, 1), datetime.date(year, 12, 31)


def _filter_time_records_by_range(
    time_records: List[Dict[str, Any]],
    start_text: str,
    end_text: str,
) -> List[Dict[str, Any]]:
    start_date = _parse_date(start_text) if start_text else None
    end_date = _parse_date(end_text) if end_text else None
    if start_date is None and end_date is None:
        return time_records
    if start_date is None:
        start_date = end_date
    if end_date is None:
        end_date = start_date
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    filtered: List[Dict[str, Any]] = []
    for record in time_records:
        record_range = _time_record_date_range(record)
        if record_range is None:
            continue
        record_start, record_end = record_range
        if start_date <= record_end and end_date >= record_start:
            item = dict(record)
            item["index"] = len(filtered)
            filtered.append(item)
    return filtered


def _overlapping_record_indices(
    start_date: datetime.date,
    end_date: datetime.date,
    record_ranges: List[Optional[Tuple[datetime.date, datetime.date]]],
) -> List[int]:
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    indices: List[int] = []
    for index, item in enumerate(record_ranges):
        if item is None:
            continue
        record_start, record_end = item
        if start_date <= record_end and end_date >= record_start:
            indices.append(index)
    return indices


def _complex_single_entity_time_feature(
    entity_re: np.ndarray,
    entity_im: np.ndarray,
    time_re: np.ndarray,
    time_im: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "hadamard":
        real = time_re * entity_re - time_im * entity_im
        imag = time_re * entity_im + time_im * entity_re
        return np.concatenate([real, imag], axis=0)
    if mode == "concat":
        return np.concatenate([entity_re, entity_im, time_re, time_im], axis=0)
    if mode == "entity_plus_time":
        real = entity_re + time_re
        imag = entity_im + time_im
        return np.concatenate([real, imag], axis=0)
    raise ValueError(f"Unknown temporal export mode: {mode}")


def _complex_entity_time_features(
    entity_re: np.ndarray,
    entity_im: np.ndarray,
    time_re: np.ndarray,
    time_im: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "hadamard":
        real = time_re[:, None, :] * entity_re[None, :, :] - time_im[:, None, :] * entity_im[None, :, :]
        imag = time_re[:, None, :] * entity_im[None, :, :] + time_im[:, None, :] * entity_re[None, :, :]
        return np.concatenate([real, imag], axis=2)
    if mode == "concat":
        entity = np.concatenate([entity_re, entity_im], axis=1)
        time = np.concatenate([time_re, time_im], axis=1)
        entity_part = np.broadcast_to(entity[None, :, :], (time.shape[0], entity.shape[0], entity.shape[1]))
        time_part = np.broadcast_to(time[:, None, :], (time.shape[0], entity.shape[0], time.shape[1]))
        return np.concatenate([entity_part, time_part], axis=2).copy()
    if mode == "entity_plus_time":
        real = entity_re[None, :, :] + time_re[:, None, :]
        imag = entity_im[None, :, :] + time_im[:, None, :]
        return np.concatenate([real, imag], axis=2)
    raise ValueError(f"Unknown temporal export mode: {mode}")


def _select_time_weights_by_records(weights: np.ndarray, time_records: List[Dict[str, Any]]) -> np.ndarray:
    selected = []
    for record in time_records:
        ids = record.get("timeplex_time_ids")
        if ids:
            valid_ids = [int(item) for item in ids if 0 <= int(item) < weights.shape[0]]
            if not valid_ids:
                valid_ids = [int(record["timeplex_time_id"])]
            selected.append(weights[valid_ids].mean(axis=0))
        else:
            selected.append(weights[int(record["timeplex_time_id"])])
    return np.stack(selected, axis=0)


def export_region_temporal_embeddings(
    checkpoint_path: str,
    entity_name_to_id: Dict[str, int],
    mapping_csv_path: str,
    output_dir: str,
    dataset: str,
    mode: str,
    dtype: str,
    time_filter: str,
    time_granularity: str,
    temporal_start: str,
    temporal_end: str,
    region_rows: Optional[List[Dict[str, str]]] = None,
    node_order_source: str = "",
) -> None:
    rows = region_rows if region_rows is not None else _read_mapping_rows(mapping_csv_path)
    if not rows:
        print(f"Skipped temporal region export because mapping file is missing or empty: {mapping_csv_path}")
        return
    _, _, region_records = _build_region_lookup(rows, entity_name_to_id)

    state = torch_load(checkpoint_path, allow_pickle=True)
    model_weights = state.get("model_weights", state)
    entity_re, entity_im = _load_timeplex_complex_entity_weights(model_weights)
    time_re, time_im, raw_time_embeddings = _load_timeplex_complex_time_weights(model_weights)
    datamap = state.get("datamap")
    time_records = _build_time_records(datamap, time_re.shape[0], time_filter, time_granularity)
    time_records = _filter_time_records_by_range(time_records, temporal_start, temporal_end)
    time_ids = np.asarray([record["timeplex_time_id"] for record in time_records], dtype=np.int64)
    if time_ids.size == 0:
        raise ValueError("No time ids are available for temporal export. Try `--temporal_time_filter all`.")

    global_to_local = _build_global_to_local_entity_map(state, entity_name_to_id, entity_re.shape[0])
    region_entity_re = np.zeros((len(rows), entity_re.shape[1]), dtype=entity_re.dtype)
    region_entity_im = np.zeros((len(rows), entity_im.shape[1]), dtype=entity_im.dtype)
    region_mask_1d = np.zeros(len(rows), dtype=np.bool_)

    for index, row in enumerate(rows):
        kg_id = _resolve_mapping_kg_id(row, entity_name_to_id, ["region_id", "Region_id"])
        local_id = global_to_local.get(kg_id) if kg_id is not None else None
        if local_id is not None and 0 <= local_id < entity_re.shape[0]:
            region_entity_re[index] = entity_re[local_id]
            region_entity_im[index] = entity_im[local_id]
            region_mask_1d[index] = True
        region_records[index].update(
            {
                "kg_id": kg_id,
                "timeplex_local_id": local_id,
            }
        )

    selected_time_re = _select_time_weights_by_records(time_re, time_records)
    selected_time_im = _select_time_weights_by_records(time_im, time_records)
    temporal = _complex_entity_time_features(
        region_entity_re,
        region_entity_im,
        selected_time_re,
        selected_time_im,
        mode,
    ).astype(np.dtype(dtype), copy=False)
    time_embeddings = _select_time_weights_by_records(raw_time_embeddings, time_records).astype(np.dtype(dtype), copy=False)
    mask = np.broadcast_to(region_mask_1d[None, :], temporal.shape[:2]).copy()

    os.makedirs(output_dir, exist_ok=True)
    temporal_path = os.path.join(output_dir, f"{dataset}_region_temporal_embedding.npy")
    mask_path = os.path.join(output_dir, f"{dataset}_region_temporal_mask.npy")
    time_path = os.path.join(output_dir, f"{dataset}_time_embedding.npy")
    index_path = os.path.join(output_dir, f"{dataset}_region_temporal_index.json")

    np.save(temporal_path, temporal)
    np.save(mask_path, mask)
    np.save(time_path, time_embeddings)
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset,
                "embedding_file": os.path.basename(temporal_path),
                "mask_file": os.path.basename(mask_path),
                "time_embedding_file": os.path.basename(time_path),
                "shape": list(temporal.shape),
                "temporal_mode": mode,
                "time_filter": time_filter,
                "time_granularity": time_granularity,
                "node_order_source": node_order_source or mapping_csv_path,
                "geo_ids": [_parse_area_id(record.get("geo_id") or record.get("region_id")) for record in region_records],
                "time_axis": time_records,
                "region_axis": region_records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved region temporal embeddings: {temporal_path}")
    print(f"Saved region temporal mask: {mask_path}")
    print(f"Saved temporal index metadata: {index_path}")


def export_region_event_temporal_embeddings(
    checkpoint_path: str,
    dataset_path: str,
    entity_name_to_id: Dict[str, int],
    relation_name_to_id: Dict[str, int],
    region_map_csv: str,
    road_map_csv: str,
    poi_map_csv: str,
    output_dir: str,
    dataset: str,
    mode: str,
    dtype: str,
    time_filter: str,
    time_granularity: str,
    temporal_start: str,
    temporal_end: str,
    event_relation_filter: str,
    event_entity_pattern: str,
    event_region_sources: str,
    append_count: bool,
    region_rows: Optional[List[Dict[str, str]]] = None,
    node_order_source: str = "",
) -> None:
    region_rows = region_rows if region_rows is not None else _read_mapping_rows(region_map_csv)
    if not region_rows:
        print(f"Skipped event temporal export because region mapping file is missing or empty: {region_map_csv}")
        return

    state = torch_load(checkpoint_path, allow_pickle=True)
    model_weights = state.get("model_weights", state)
    entity_re, entity_im = _load_timeplex_complex_entity_weights(model_weights)
    time_re, time_im, _ = _load_timeplex_complex_time_weights(model_weights)
    datamap = state.get("datamap")
    time_records = _build_time_records(datamap, time_re.shape[0], time_filter, time_granularity)
    time_records = _filter_time_records_by_range(time_records, temporal_start, temporal_end)
    if not time_records:
        raise ValueError("No time ids are available for event temporal export.")

    region_kg_to_index, region_key_to_index, region_records = _build_region_lookup(region_rows, entity_name_to_id)
    sources = {item.strip().lower() for item in event_region_sources.split(",") if item.strip()}
    entity_to_region: Dict[int, int] = {}
    if "road" in sources:
        entity_to_region.update(
            _load_entity_to_region_index(road_map_csv, entity_name_to_id, region_key_to_index, ["road_id", "Road_id"])
        )
    if "poi" in sources:
        entity_to_region.update(
            _load_entity_to_region_index(poi_map_csv, entity_name_to_id, region_key_to_index, ["poi_id", "POI_id"])
        )

    id_to_entity_name = {entity_id: name for name, entity_id in entity_name_to_id.items()}
    relation_id_to_name = {relation_id: name for name, relation_id in relation_name_to_id.items()}
    relation_filter = _parse_relation_filter(event_relation_filter, relation_name_to_id)
    event_regex = re.compile(event_entity_pattern, re.IGNORECASE)
    global_to_local = _build_global_to_local_entity_map(state, entity_name_to_id, entity_re.shape[0])
    selected_time_re = _select_time_weights_by_records(time_re, time_records)
    selected_time_im = _select_time_weights_by_records(time_im, time_records)
    record_ranges = [_time_record_date_range(record) for record in time_records]

    entity_dim = entity_re.shape[1]
    if mode == "concat":
        event_dim = entity_dim * 4
    else:
        event_dim = entity_dim * 2

    sums = np.zeros((len(time_records), len(region_rows), event_dim), dtype=np.float64)
    counts = np.zeros((len(time_records), len(region_rows)), dtype=np.float32)
    relation_counts: Dict[int, int] = {}
    total_rows = 0
    matched_events = 0
    mapped_events = 0
    used_assignments = 0
    skipped_no_region = 0
    skipped_no_timeplex_entity = 0

    temporal_files = [
        os.path.join(dataset_path, f"temporal_{split}.csv")
        for split in ("train", "valid", "test")
    ]
    for temporal_file in temporal_files:
        if not os.path.exists(temporal_file):
            continue
        with open(temporal_file, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                total_rows += 1
                try:
                    head_id = int(str(row["head_id"]).strip())
                    relation_id = int(str(row["relation_id"]).strip())
                    tail_id = int(str(row["tail_id"]).strip())
                except (KeyError, TypeError, ValueError):
                    continue
                if relation_filter is not None and relation_id not in relation_filter:
                    continue

                head_name = id_to_entity_name.get(head_id, "")
                tail_name = id_to_entity_name.get(tail_id, "")
                head_is_event = bool(event_regex.search(head_name))
                tail_is_event = bool(event_regex.search(tail_name))
                if not head_is_event and not tail_is_event:
                    continue

                matched_events += 1
                event_id = head_id if head_is_event else tail_id
                other_id = tail_id if head_is_event else head_id
                region_index = None
                if "region" in sources and other_id in region_kg_to_index:
                    region_index = region_kg_to_index[other_id]
                if region_index is None:
                    region_index = entity_to_region.get(other_id)
                if region_index is None:
                    skipped_no_region += 1
                    continue

                local_event_id = global_to_local.get(event_id)
                if local_event_id is None or local_event_id < 0 or local_event_id >= entity_re.shape[0]:
                    skipped_no_timeplex_entity += 1
                    continue

                start_date = _parse_date(row.get("start_time"))
                end_date = _parse_date(row.get("end_time"))
                if start_date is None and end_date is None:
                    continue
                if start_date is None:
                    start_date = end_date
                if end_date is None:
                    end_date = start_date
                time_indices = _overlapping_record_indices(start_date, end_date, record_ranges)
                if not time_indices:
                    continue

                mapped_events += 1
                relation_counts[relation_id] = relation_counts.get(relation_id, 0) + 1
                for time_index in time_indices:
                    event_vec = _complex_single_entity_time_feature(
                        entity_re[local_event_id],
                        entity_im[local_event_id],
                        selected_time_re[time_index],
                        selected_time_im[time_index],
                        mode,
                    )
                    sums[time_index, region_index] += event_vec
                    counts[time_index, region_index] += 1.0
                    used_assignments += 1

    mask = counts > 0
    event_temporal = np.zeros_like(sums, dtype=np.float32)
    event_temporal[mask] = (sums[mask] / counts[mask, None]).astype(np.float32)
    if append_count:
        count_feature = np.log1p(counts).astype(np.float32)[..., None]
        event_temporal = np.concatenate([event_temporal, count_feature], axis=-1)
    event_temporal = event_temporal.astype(np.dtype(dtype), copy=False)

    os.makedirs(output_dir, exist_ok=True)
    temporal_path = os.path.join(output_dir, f"{dataset}_region_event_temporal_embedding.npy")
    mask_path = os.path.join(output_dir, f"{dataset}_region_event_temporal_mask.npy")
    count_path = os.path.join(output_dir, f"{dataset}_region_event_temporal_count.npy")
    index_path = os.path.join(output_dir, f"{dataset}_region_event_temporal_index.json")
    np.save(temporal_path, event_temporal)
    np.save(mask_path, mask)
    np.save(count_path, counts)

    relation_summary = [
        {
            "relation_id": int(relation_id),
            "relation": relation_id_to_name.get(relation_id, str(relation_id)),
            "count": int(count),
        }
        for relation_id, count in sorted(relation_counts.items(), key=lambda item: item[0])
    ]
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset,
                "embedding_file": os.path.basename(temporal_path),
                "mask_file": os.path.basename(mask_path),
                "count_file": os.path.basename(count_path),
                "shape": list(event_temporal.shape),
                "temporal_mode": mode,
                "time_filter": time_filter,
                "time_granularity": time_granularity,
                "event_entity_pattern": event_entity_pattern,
                "event_relation_filter": event_relation_filter,
                "event_region_sources": sorted(sources),
                "append_count": bool(append_count),
                "node_order_source": node_order_source or region_map_csv,
                "geo_ids": [_parse_area_id(record.get("geo_id") or record.get("region_id")) for record in region_records],
                "stats": {
                    "total_temporal_rows": int(total_rows),
                    "matched_event_rows": int(matched_events),
                    "mapped_event_rows": int(mapped_events),
                    "used_region_time_assignments": int(used_assignments),
                    "skipped_no_region": int(skipped_no_region),
                    "skipped_no_timeplex_entity": int(skipped_no_timeplex_entity),
                },
                "event_relations": relation_summary,
                "time_axis": time_records,
                "region_axis": region_records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved region event temporal embeddings: {temporal_path}")
    print(f"Saved region event temporal mask: {mask_path}")
    print(f"Saved region event temporal counts: {count_path}")
    print(f"Saved event temporal index metadata: {index_path}")
    print(
        "Event export stats: matched={}, mapped={}, assignments={}, skipped_no_region={}".format(
            matched_events, mapped_events, used_assignments, skipped_no_region
        )
    )


def fuse_embeddings(
    static_embeddings: np.ndarray,
    static_mask: Optional[np.ndarray],
    temporal_embeddings: Optional[np.ndarray],
    temporal_mask: Optional[np.ndarray],
    fuse_mode: str,
    alpha: float,
) -> np.ndarray:
    if temporal_embeddings is None or temporal_mask is None:
        return static_embeddings

    if fuse_mode == "static_only":
        return static_embeddings
    if fuse_mode == "time_only":
        fused = static_embeddings.copy()
        fused[temporal_mask] = temporal_embeddings[temporal_mask]
        return fused
    if fuse_mode == "concat":
        return np.concatenate([static_embeddings, temporal_embeddings], axis=1)
    if fuse_mode == "weighted_sum":
        if static_embeddings.shape[1] != temporal_embeddings.shape[1]:
            raise ValueError(
                "weighted_sum requires equal embedding dimensions. "
                f"Got static={static_embeddings.shape[1]}, temporal={temporal_embeddings.shape[1]}."
            )
        fused = static_embeddings.copy()
        has_static = static_mask if static_mask is not None else np.ones(static_embeddings.shape[0], dtype=np.bool_)
        overlap = has_static & temporal_mask
        temporal_only = (~has_static) & temporal_mask
        fused[overlap] = (1.0 - alpha) * static_embeddings[overlap] + alpha * temporal_embeddings[overlap]
        fused[temporal_only] = temporal_embeddings[temporal_only]
        return fused
    raise ValueError(f"Unknown fuse mode: {fuse_mode}")


def export_embeddings_by_mapping(
    mapping_csv_path: str,
    entity_embeddings: np.ndarray,
    output_path: str,
    include_region: bool = False,
    entity_name_to_id: Optional[Dict[str, int]] = None,
    entity_fields: Optional[List[str]] = None,
    mapping_rows: Optional[List[Dict[str, str]]] = None,
    metadata_path: Optional[str] = None,
    node_order_source: str = "",
) -> None:
    if mapping_rows is None and not os.path.exists(mapping_csv_path):
        return

    rows = mapping_rows if mapping_rows is not None else _read_mapping_rows(mapping_csv_path)

    if not rows:
        return

    dim = entity_embeddings.shape[1]
    output_dim = dim + 1 if include_region else dim
    result = np.zeros((len(rows), output_dim), dtype=entity_embeddings.dtype)
    resolved_by_name = 0
    region_records: List[Dict[str, Any]] = []
    missing_regions: List[str] = []
    for i, row in enumerate(rows):
        kg_id = _resolve_mapping_kg_id(row, entity_name_to_id, entity_fields)
        if kg_id is None:
            if entity_fields and any(field in row for field in entity_fields):
                missing_regions.append(str(row.get(entity_fields[0], row.get("region_id", i))).strip())
            region_records.append(
                {
                    "index": i,
                    "region_id": str(row.get("region_id", row.get("Region_id", i))).strip(),
                    "geo_id": row.get("geo_id", ""),
                    "kg_id": None,
                    "resolved": False,
                }
            )
            continue
        if entity_name_to_id is not None and entity_fields is not None:
            for field in entity_fields:
                if field in row and any(candidate in entity_name_to_id for candidate in _mapping_entity_candidates(row[field])):
                    resolved_by_name += 1
                    break
        if 0 <= kg_id < entity_embeddings.shape[0]:
            result[i, :dim] = entity_embeddings[kg_id]
            resolved = True
        else:
            resolved = False
            missing_regions.append(str(row.get("region_id", row.get("Region_id", i))).strip())
        if include_region:
            try:
                result[i, dim] = float(str(row["Region_id"]).strip())
            except (KeyError, ValueError):
                result[i, dim] = 0.0
        region_records.append(
            {
                "index": i,
                "region_id": str(row.get("region_id", row.get("Region_id", i))).strip(),
                "geo_id": row.get("geo_id", ""),
                "kg_id": int(kg_id),
                "resolved": bool(resolved),
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, result)
    if metadata_path:
        geo_ids = []
        for record in region_records:
            geo_id = _parse_area_id(record.get("geo_id") or record.get("region_id"))
            geo_ids.append(geo_id)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "embedding_file": os.path.basename(output_path),
                    "shape": list(result.shape),
                    "node_order_source": node_order_source or mapping_csv_path,
                    "geo_ids": geo_ids,
                    "missing_regions": missing_regions,
                    "region_axis": region_records,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
    if entity_name_to_id is not None and entity_fields is not None:
        print(
            "Saved mapped embeddings: {} rows={}, name_resolved={}".format(
                output_path, len(rows), resolved_by_name
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fused GIE + TimePlex embeddings.")
    parser.add_argument("--dataset", default="NYC", choices=["NYC", "CHI"])
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--gie_model", default="GIE", help="Static model class in models/__init__.py.")
    parser.add_argument("--gie_checkpoint", required=True, help="Path to GIE model checkpoint (model.pt).")
    parser.add_argument("--gie_config", default=None, help="Path to GIE config.json (optional).")
    parser.add_argument(
        "--timeplex_checkpoint",
        default=None,
        help="Path to TimePlex checkpoint (best_valid_model.pt). If omitted, exports static embeddings only.",
    )
    parser.add_argument(
        "--fuse_mode",
        default="concat",
        choices=["concat", "weighted_sum", "static_only", "time_only"],
        help="How to combine static and temporal embeddings.",
    )
    parser.add_argument(
        "--alpha",
        default=0.7,
        type=float,
        help="Temporal weight for weighted_sum fusion.",
    )
    parser.add_argument("--output_dir", default="embedding")
    parser.add_argument("--entity_output", default=None, help="Output .npy filename for entity embeddings.")
    parser.add_argument("--region_map_csv", default="areaid2KGid.csv")
    parser.add_argument(
        "--region_geo_file",
        default="",
        help="Optional LibCity .geo file. If set, region embeddings are exported in this geo_id order.",
    )
    parser.add_argument("--poi_map_csv", default="POIid2KGid.csv")
    parser.add_argument("--road_map_csv", default="roadid2KGid.csv")
    parser.add_argument(
        "--temporal_export_mode",
        default="hadamard",
        choices=["hadamard", "concat", "entity_plus_time"],
        help="How to build region-time embeddings from TimePlex entity/time factors.",
    )
    parser.add_argument(
        "--temporal_time_filter",
        default="observed",
        choices=["observed", "all"],
        help="Use observed calendar times only, or include TimePlex sentinel/padding-like times.",
    )
    parser.add_argument(
        "--temporal_time_granularity",
        default="year",
        choices=["year", "month", "quarter", "day"],
        help="Temporal axis granularity for exported region-time embeddings.",
    )
    parser.add_argument(
        "--temporal_dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Numpy dtype for exported temporal tensors.",
    )
    parser.add_argument(
        "--temporal_start",
        default="",
        help="Optional inclusive start date for exported temporal axes, e.g. 2020-04-01.",
    )
    parser.add_argument(
        "--temporal_end",
        default="",
        help="Optional inclusive end date for exported temporal axes, e.g. 2020-06-30.",
    )
    parser.add_argument(
        "--skip_temporal_region_export",
        action="store_true",
        help="Do not export <DATASET>_region_temporal_embedding.npy even when a TimePlex checkpoint is provided.",
    )
    parser.add_argument(
        "--export_event_temporal_region",
        action="store_true",
        help="Export <DATASET>_region_event_temporal_embedding.npy by aggregating active event entities per region-time.",
    )
    parser.add_argument(
        "--event_relation_filter",
        default="",
        help="Comma-separated relation names or ids to keep for event export. Empty keeps all event-like triples.",
    )
    parser.add_argument(
        "--event_entity_pattern",
        default="event",
        help="Regex used to identify event entities from entity names.",
    )
    parser.add_argument(
        "--event_region_sources",
        default="region,road,poi",
        help="Comma-separated mapping sources for event export: region, road, poi.",
    )
    parser.add_argument(
        "--skip_event_count_feature",
        action="store_true",
        help="Do not append log1p(event_count) as the last channel of event temporal embeddings.",
    )
    args = parser.parse_args()

    dataset_path = os.path.join(args.data_path, args.dataset)
    static_embeddings = load_gie_embeddings(
        dataset_path=dataset_path,
        gie_checkpoint=args.gie_checkpoint,
        gie_config=args.gie_config,
        model_name=args.gie_model,
    )

    temporal_embeddings = None
    temporal_mask = None
    static_mask = np.ones(static_embeddings.shape[0], dtype=np.bool_)
    entity_name_to_id = load_entity_name_to_id(dataset_path)
    relation_name_to_id = load_relation_name_to_id(dataset_path)
    region_rows, region_order_source = load_region_rows_for_export(args.region_map_csv, args.region_geo_file)
    if args.timeplex_checkpoint:
        temporal_embeddings, temporal_mask = load_timeplex_entity_embeddings(
            checkpoint_path=args.timeplex_checkpoint,
            entity_name_to_id=entity_name_to_id,
        )
        if temporal_embeddings.shape[0] > static_embeddings.shape[0]:
            pad_rows = temporal_embeddings.shape[0] - static_embeddings.shape[0]
            static_embeddings = np.concatenate(
                [static_embeddings, np.zeros((pad_rows, static_embeddings.shape[1]), dtype=static_embeddings.dtype)],
                axis=0,
            )
            static_mask = np.concatenate([static_mask, np.zeros(pad_rows, dtype=np.bool_)], axis=0)
        elif temporal_embeddings.shape[0] < static_embeddings.shape[0]:
            pad_rows = static_embeddings.shape[0] - temporal_embeddings.shape[0]
            temporal_embeddings = np.concatenate(
                [temporal_embeddings, np.zeros((pad_rows, temporal_embeddings.shape[1]), dtype=temporal_embeddings.dtype)],
                axis=0,
            )
            temporal_mask = np.concatenate([temporal_mask, np.zeros(pad_rows, dtype=np.bool_)], axis=0)

    fused_embeddings = fuse_embeddings(
        static_embeddings=static_embeddings,
        static_mask=static_mask,
        temporal_embeddings=temporal_embeddings,
        temporal_mask=temporal_mask,
        fuse_mode=args.fuse_mode,
        alpha=args.alpha,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    entity_output = args.entity_output or f"{args.dataset}_entity_hybrid_embedding.npy"
    entity_output_path = os.path.join(args.output_dir, entity_output)
    np.save(entity_output_path, fused_embeddings)

    export_embeddings_by_mapping(
        mapping_csv_path=args.region_map_csv,
        entity_embeddings=fused_embeddings,
        output_path=os.path.join(args.output_dir, f"{args.dataset}_region_hybrid_embedding.npy"),
        include_region=False,
        entity_name_to_id=entity_name_to_id,
        entity_fields=["region_id", "Region_id"],
        mapping_rows=region_rows,
        metadata_path=os.path.join(args.output_dir, f"{args.dataset}_region_hybrid_index.json"),
        node_order_source=region_order_source,
    )
    export_embeddings_by_mapping(
        mapping_csv_path=args.poi_map_csv,
        entity_embeddings=fused_embeddings,
        output_path=os.path.join(args.output_dir, f"{args.dataset}_POI_hybrid_embedding.npy"),
        include_region=True,
        entity_name_to_id=entity_name_to_id,
        entity_fields=["poi_id", "POI_id"],
    )
    export_embeddings_by_mapping(
        mapping_csv_path=args.road_map_csv,
        entity_embeddings=fused_embeddings,
        output_path=os.path.join(args.output_dir, f"{args.dataset}_Road_hybrid_embedding.npy"),
        include_region=True,
        entity_name_to_id=entity_name_to_id,
        entity_fields=["road_id", "Road_id"],
    )

    if args.timeplex_checkpoint and not args.skip_temporal_region_export:
        export_region_temporal_embeddings(
            checkpoint_path=args.timeplex_checkpoint,
            entity_name_to_id=entity_name_to_id,
            mapping_csv_path=args.region_map_csv,
            output_dir=args.output_dir,
            dataset=args.dataset,
            mode=args.temporal_export_mode,
            dtype=args.temporal_dtype,
            time_filter=args.temporal_time_filter,
            time_granularity=args.temporal_time_granularity,
            temporal_start=args.temporal_start,
            temporal_end=args.temporal_end,
            region_rows=region_rows,
            node_order_source=region_order_source,
        )

    if args.timeplex_checkpoint and args.export_event_temporal_region:
        export_region_event_temporal_embeddings(
            checkpoint_path=args.timeplex_checkpoint,
            dataset_path=dataset_path,
            entity_name_to_id=entity_name_to_id,
            relation_name_to_id=relation_name_to_id,
            region_map_csv=args.region_map_csv,
            road_map_csv=args.road_map_csv,
            poi_map_csv=args.poi_map_csv,
            output_dir=args.output_dir,
            dataset=args.dataset,
            mode=args.temporal_export_mode,
            dtype=args.temporal_dtype,
            time_filter=args.temporal_time_filter,
            time_granularity=args.temporal_time_granularity,
            temporal_start=args.temporal_start,
            temporal_end=args.temporal_end,
            event_relation_filter=args.event_relation_filter,
            event_entity_pattern=args.event_entity_pattern,
            event_region_sources=args.event_region_sources,
            append_count=not args.skip_event_count_feature,
            region_rows=region_rows,
            node_order_source=region_order_source,
        )

    print(f"Saved fused entity embeddings: {entity_output_path}")


if __name__ == "__main__":
    main()
