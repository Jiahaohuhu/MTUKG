"""Export static entity/relation embeddings from a trained GIE checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

import models
from datasets.kg_dataset import KGDataset
from datasets.process import process_dataset


def ensure_static_dataset_ready(dataset_path: str) -> None:
    required = ["train.pickle", "valid.pickle", "test.pickle", "to_skip.pickle"]
    if all(os.path.exists(os.path.join(dataset_path, name)) for name in required):
        return
    examples, filters = process_dataset(dataset_path)
    import pickle
    for split in ["train", "valid", "test"]:
        with open(os.path.join(dataset_path, f"{split}.pickle"), "wb") as handle:
            pickle.dump(examples[split], handle)
    with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as handle:
        pickle.dump(filters, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static GIE embeddings.")
    parser.add_argument("--dataset", required=True, help="Dataset name under data_path.")
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--model", default="GIE")
    parser.add_argument("--checkpoint", required=True, help="Path to GIE model.pt")
    parser.add_argument("--config", default=None, help="Optional config.json from the same run.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--entity_file", default="static_entity_embedding.npy")
    parser.add_argument("--relation_file", default="static_relation_embedding.npy")
    args = parser.parse_args()

    dataset_path = os.path.join(args.data_path, args.dataset)
    ensure_static_dataset_ready(dataset_path)
    dataset = KGDataset(dataset_path, debug=False)
    sizes = dataset.get_shape()

    state = torch.load(args.checkpoint, map_location="cpu")
    if "entity.weight" not in state:
        raise ValueError("Checkpoint does not look like a static GIE checkpoint.")

    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as handle:
            config = json.load(handle)

    rank = int(state["entity.weight"].shape[1])
    model_args = SimpleNamespace(
        sizes=sizes,
        rank=rank,
        dropout=float(config.get("dropout", 0.0)),
        gamma=float(config.get("gamma", 0.0)),
        dtype=str(config.get("dtype", "double")),
        bias=str(config.get("bias", "constant")),
        init_size=float(config.get("init_size", 1e-3)),
        multi_c=bool(config.get("multi_c", False)),
    )
    model_cls = getattr(models, args.model)
    model = model_cls(model_args)
    model.load_state_dict(state, strict=True)
    model.eval()

    entity_embeddings = model.entity.weight.detach().cpu().numpy()
    relation_embeddings = model.rel.weight.detach().cpu().numpy()

    idx_path = os.path.join(dataset_path, "entity_idx_embedding.csv")
    if os.path.exists(idx_path):
        idx = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
        idx = np.atleast_1d(idx)
        if len(idx) == entity_embeddings.shape[0]:
            max_idx = int(np.max(idx))
            aligned = np.zeros((max_idx + 1, entity_embeddings.shape[1]), dtype=entity_embeddings.dtype)
            for local_id, global_id in enumerate(idx):
                if global_id >= 0:
                    aligned[int(global_id)] = entity_embeddings[local_id]
            entity_embeddings = aligned

    os.makedirs(args.output_dir, exist_ok=True)
    entity_path = os.path.join(args.output_dir, args.entity_file)
    relation_path = os.path.join(args.output_dir, args.relation_file)
    np.save(entity_path, entity_embeddings)
    np.save(relation_path, relation_embeddings)

    print("Saved static entity embeddings:", entity_path)
    print("Saved static relation embeddings:", relation_path)
    print("Checkpoint (for reproducibility):", args.checkpoint)


if __name__ == "__main__":
    main()

