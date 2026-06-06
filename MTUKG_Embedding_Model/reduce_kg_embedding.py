import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def pca_reduce_array(x, dim):
    original_shape = x.shape
    if x.ndim == 2:
        flat = x
        reshape_prefix = None
    elif x.ndim == 3:
        reshape_prefix = x.shape[:2]
        flat = x.reshape(-1, x.shape[-1])
    else:
        raise ValueError("Expected a 2D or 3D embedding array, got shape {}".format(x.shape))

    flat = flat.astype(np.float32, copy=False)
    dim = min(int(dim), flat.shape[0], flat.shape[1])
    mean = flat.mean(axis=0, keepdims=True)
    flat0 = flat - mean

    cov = (flat0.T @ flat0) / max(flat0.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:dim]
    components = eigvecs[:, order].astype(np.float32, copy=False)
    reduced = (flat0 @ components).astype(np.float32, copy=False)

    if reshape_prefix is not None:
        reduced = reduced.reshape(reshape_prefix[0], reshape_prefix[1], dim)
    return reduced, original_shape


def pca_reduce_sparse_event_array(x, dim, mask):
    original_shape = x.shape
    if x.ndim != 3:
        raise ValueError("Sparse event PCA expects a 3D embedding array, got shape {}".format(x.shape))
    if mask.shape != x.shape[:2]:
        raise ValueError("Event mask shape {} does not match embedding shape {}".format(mask.shape, x.shape[:2]))

    flat = x.reshape(-1, x.shape[-1]).astype(np.float32, copy=False)
    flat_mask = mask.reshape(-1).astype(bool, copy=False)
    dim = min(int(dim), flat.shape[1])
    reduced_flat = np.zeros((flat.shape[0], dim), dtype=np.float32)

    if not np.any(flat_mask):
        return reduced_flat.reshape(x.shape[0], x.shape[1], dim), original_shape, 0

    event_flat = flat[flat_mask]
    dim = min(dim, event_flat.shape[0], event_flat.shape[1])
    if reduced_flat.shape[1] != dim:
        reduced_flat = np.zeros((flat.shape[0], dim), dtype=np.float32)
    mean = event_flat.mean(axis=0, keepdims=True)
    event0 = event_flat - mean
    cov = (event0.T @ event0) / max(event0.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:dim]
    components = eigvecs[:, order].astype(np.float32, copy=False)
    reduced_flat[flat_mask] = (event0 @ components).astype(np.float32, copy=False)
    return reduced_flat.reshape(x.shape[0], x.shape[1], dim), original_shape, int(flat_mask.sum())


def reduce_file(input_path, output_path, dim, mask_path=None):
    x = np.load(input_path)
    if mask_path is not None and mask_path.exists():
        mask = np.load(mask_path)
        reduced, original_shape, event_rows = pca_reduce_sparse_event_array(x, dim, mask)
        mode_text = "masked_event_pca rows={}".format(event_rows)
    else:
        reduced, original_shape = pca_reduce_array(x, dim)
        mode_text = "pca"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, reduced)
    print("{} {} -> {} {} ({})".format(input_path, original_shape, output_path, reduced.shape, mode_text))


def event_mask_path_for(input_dir, dataset, input_name):
    if input_name != "{}_region_event_temporal_embedding.npy".format(dataset):
        return None
    return input_dir / "{}_region_event_temporal_mask.npy".format(dataset)


def copy_sidecar_files(input_dir, output_dir, dataset):
    sidecars = [
        "{}_region_hybrid_index.json".format(dataset),
        "{}_region_temporal_mask.npy".format(dataset),
        "{}_region_temporal_index.json".format(dataset),
        "{}_time_embedding.npy".format(dataset),
        "{}_region_event_temporal_mask.npy".format(dataset),
        "{}_region_event_temporal_count.npy".format(dataset),
        "{}_region_event_temporal_index.json".format(dataset),
    ]
    for name in sidecars:
        src = input_dir / name
        if src.exists():
            dst = output_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_reduction_config(output_dir, args):
    config_path = output_dir / "reduction_config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "dim": int(args.dim),
                "files": args.files,
                "event_masked_pca": not args.disable_event_masked_pca,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description="Reduce exported KG embedding .npy files with PCA.")
    parser.add_argument("--dataset", default="NYC")
    parser.add_argument("--input_dir", default="embedding_month", type=Path)
    parser.add_argument("--output_dir", default="embedding_month_reduced", type=Path)
    parser.add_argument("--dim", default=32, type=int)
    parser.add_argument(
        "--files",
        nargs="*",
        default=[
            "{dataset}_region_hybrid_embedding.npy",
            "{dataset}_region_temporal_embedding.npy",
            "{dataset}_region_event_temporal_embedding.npy",
        ],
        help="Embedding files under input_dir. Use {dataset} as a placeholder.",
    )
    parser.add_argument("--copy_sidecars", action="store_true", help="Copy temporal index/mask/count files unchanged.")
    parser.add_argument(
        "--disable_event_masked_pca",
        action="store_true",
        help="Use ordinary PCA for event temporal embeddings. By default, event PCA preserves no-event rows as zero.",
    )
    args = parser.parse_args()

    for pattern in args.files:
        name = pattern.format(dataset=args.dataset)
        input_path = args.input_dir / name
        if not input_path.exists():
            print("Skipped missing file: {}".format(input_path))
            continue
        output_name = "{}_{}.npy".format(input_path.stem, args.dim)
        mask_path = None if args.disable_event_masked_pca else event_mask_path_for(args.input_dir, args.dataset, name)
        reduce_file(input_path, args.output_dir / output_name, args.dim, mask_path=mask_path)

    if args.copy_sidecars:
        copy_sidecar_files(args.input_dir, args.output_dir, args.dataset)
    write_reduction_config(args.output_dir, args)


if __name__ == "__main__":
    main()
