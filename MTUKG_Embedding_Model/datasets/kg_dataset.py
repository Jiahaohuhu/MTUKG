"""Dataset class for loading and processing UrbanKG datasets."""

from __future__ import annotations

import os
import pickle as pkl

import numpy as np
import torch


class _NumpyCompatUnpickler(pkl.Unpickler):
    """Load NumPy 2.x pickles in NumPy 1.x environments."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pickle(path):
    with open(path, "rb") as in_file:
        return _NumpyCompatUnpickler(in_file).load()


class KGDataset(object):
    """Knowledge Graph dataset class."""

    def __init__(self, data_path, debug):
        """Creates KG dataset object for data loading."""
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        for split in ["train", "test", "valid"]:
            file_path = os.path.join(self.data_path, split + ".pickle")
            self.data[split] = _load_pickle(file_path)

        self.to_skip = _load_pickle(os.path.join(self.data_path, "to_skip.pickle"))

        entity_idx_path = os.path.join(self.data_path, "entity_idx_embedding.csv")
        relation_idx_path = os.path.join(self.data_path, "relations_idx_embeddings.csv")
        if os.path.exists(entity_idx_path) and os.path.exists(relation_idx_path):
            entity_idx = np.loadtxt(entity_idx_path, delimiter=",", dtype=np.int64)
            relation_idx = np.loadtxt(relation_idx_path, delimiter=",", dtype=np.int64)
            self.n_entities = int(np.max(np.atleast_1d(entity_idx)) + 1)
            self.n_predicates = int(np.max(np.atleast_1d(relation_idx)) + 1) * 2
        else:
            all_examples = np.concatenate([self.data["train"], self.data["valid"], self.data["test"]], axis=0)
            self.n_entities = int(max(np.max(all_examples[:, 0]), np.max(all_examples[:, 2])) + 1)
            self.n_predicates = int(np.max(all_examples[:, 1]) + 1) * 2

    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split."""
        examples = self.data[split]
        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))

    def get_filters(self):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities
