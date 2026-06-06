"""Base Knowledge Graph embedding model."""
from abc import ABC, abstractmethod
import csv
import os

import torch
from torch import nn

import pdb


ENTITY_TYPE_ALIASES = {
    "poi": "POI",
    "pc": "PC",
    "area": "Area",
    "road": "Road",
    "junction": "Junction",
    "fz": "FZ",
    "rc": "RC",
    "jc": "JC",
    "borough": "Borough",
}


def _canonical_entity_type(entity_name):
    token = str(entity_name).strip()
    if "/" in token:
        token = token.split("/", 1)[0]
    elif "::" in token:
        token = token.split("::", 1)[0]
    else:
        token = token.split(":", 1)[0]
    return ENTITY_TYPE_ALIASES.get(token.lower())


def _load_typed_entities(dataset_path):
    typed_entities = {v: [] for v in ENTITY_TYPE_ALIASES.values()}

    csv_path = os.path.join(dataset_path, "entity2id.csv")
    txt_path = os.path.join(dataset_path, "entity2id_NYC.txt")

    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return None
            name_key = "entity" if "entity" in reader.fieldnames else reader.fieldnames[0]
            id_key = "entity_id" if "entity_id" in reader.fieldnames else reader.fieldnames[-1]
            for row in reader:
                if row.get(name_key) is None or row.get(id_key) is None:
                    continue
                entity_type = _canonical_entity_type(row[name_key])
                if entity_type is None:
                    continue
                try:
                    entity_id = int(str(row[id_key]).strip())
                except ValueError:
                    continue
                typed_entities[entity_type].append(entity_id)
        return typed_entities

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                entity_type = _canonical_entity_type(parts[0])
                if entity_type is None:
                    continue
                try:
                    entity_id = int(parts[1])
                except ValueError:
                    continue
                typed_entities[entity_type].append(entity_id)
        return typed_entities

    return None


def _build_entity_type_index(entitydict, entity_count):
    if entitydict is None:
        return None
    entity_type_index = torch.full((entity_count,), -1, dtype=torch.int16)
    for type_idx, (_, entity_ids) in enumerate(entitydict.items()):
        if not entity_ids:
            continue
        valid_ids = [eid for eid in entity_ids if 0 <= int(eid) < entity_count]
        if not valid_ids:
            continue
        entity_type_index[torch.tensor(valid_ids, dtype=torch.long)] = type_idx
    return entity_type_index


class KGModel(nn.Module, ABC):
    """Base Knowledge Graph Embedding model class.

    Attributes:
        sizes: Tuple[int, int, int] with (n_entities, n_relations, n_entities)
        rank: integer for embedding dimension
        dropout: float for dropout rate
        gamma: torch.nn.Parameter for margin in ranking-based loss
        data_type: torch.dtype for machine precision (single or double)
        bias: string for whether to learn or fix bias (none for no bias)
        init_size: float for embeddings' initialization scale
        entity: torch.nn.Embedding with entity embeddings
        rel: torch.nn.Embedding with relation embeddings
        bh: torch.nn.Embedding with head entity bias embeddings
        bt: torch.nn.Embedding with tail entity bias embeddings
    """

    def __init__(self, sizes, rank, dropout, gamma, data_type, bias, init_size):
        """Initialize KGModel."""
        super(KGModel, self).__init__()
        if data_type == 'double':
            self.data_type = torch.double
        else:
            self.data_type = torch.float
        self.sizes = sizes
        self.rank = rank
        self.dropout = dropout
        self.bias = bias
        self.init_size = init_size
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.entity = nn.Embedding(sizes[0], rank)

        # pdb.set_trace()

        self.rel = nn.Embedding(sizes[1], rank)
        self.bh = nn.Embedding(sizes[0], 1)
        self.bh.weight.data = torch.zeros((sizes[0], 1), dtype=self.data_type)
        self.bt = nn.Embedding(sizes[0], 1)
        self.bt.weight.data = torch.zeros((sizes[0], 1), dtype=self.data_type)

    @abstractmethod
    def get_queries(self, queries):
        """Compute embedding and biases of queries.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
             lhs_e: torch.Tensor with queries' embeddings (embedding of head entities and relations)
             lhs_biases: torch.Tensor with head entities' biases
        """
        pass

    @abstractmethod
    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
             rhs_e: torch.Tensor with targets' embeddings
                    if eval_mode=False returns embedding of tail entities (n_queries x rank)
                    else returns embedding of all possible entities in the KG dataset (n_entities x rank)
             rhs_biases: torch.Tensor with targets' biases
                         if eval_mode=False returns biases of tail entities (n_queries x 1)
                         else returns biases of all possible entities in the KG dataset (n_entities x 1)
        """
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space.

        Args:
            lhs_e: torch.Tensor with queries' embeddings
            rhs_e: torch.Tensor with targets' embeddings
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            scores: torch.Tensor with similarity scores of queries against targets
        """
        pass

    def score(self, lhs, rhs, eval_mode):
        """Scores queries against targets

        Args:
            lhs: Tuple[torch.Tensor, torch.Tensor] with queries' embeddings and head biases
                 returned by get_queries(queries)
            rhs: Tuple[torch.Tensor, torch.Tensor] with targets' embeddings and tail biases
                 returned by get_rhs(queries, eval_mode)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            score: torch.Tensor with scores of queries against targets
                   if eval_mode=True, returns scores against all possible tail entities, shape (n_queries x n_entities)
                   else returns scores for triples in batch (shape n_queries x 1)
        """
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                return lhs_biases + rhs_biases.t() + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score

    def mutiview_score(self):
        pass

    def get_factors(self, queries):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
        return head_e, rel_e, rhs_e

    def forward(self, queries, eval_mode=False):
        """KGModel forward pass.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            predictions: torch.Tensor with triples' scores
                         shape is (n_queries x 1) if eval_mode is false
                         else (n_queries x n_entities)
            factors: embeddings to regularize
        """
        # get embeddings and similarity scores
        lhs_e, lhs_biases = self.get_queries(queries)
        # queries = F.dropout(queries, self.dropout, training=self.training)
        rhs_e, rhs_biases = self.get_rhs(queries, eval_mode)
        # candidates = F.dropout(candidates, self.dropout, training=self.training)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)

        # get factors for regularization
        factors = self.get_factors(queries)
        return predictions, factors

    def get_ranking(self, queries, filters, batch_size=1000, candidate_batch_size=50000, DAPTA_PATH=None):
        """Compute filtered ranking of correct entity for evaluation.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            filters: filters[(head, relation)] gives entities to ignore (filtered setting)
            batch_size: int for evaluation batch size

        Returns:
            ranks: torch.Tensor with ranks or correct entities
        """
        ranks = torch.ones(len(queries))
        candidate_batch_size = max(1, int(candidate_batch_size))
        entitydict = _load_typed_entities(DAPTA_PATH) if DAPTA_PATH else None
        entity_type_index = _build_entity_type_index(entitydict, self.sizes[0])

        with torch.no_grad():
            b_begin = 0
            candidates = self.get_rhs(queries, eval_mode=True)
            all_candidate_embeddings, all_candidate_biases = candidates
            total_candidates = all_candidate_embeddings.shape[0]
            if entity_type_index is not None:
                entity_type_index = entity_type_index.to(all_candidate_embeddings.device)
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()
                query_count = these_queries.shape[0]
                true_entities = queries[b_begin:b_begin + query_count, 2].tolist()

                q = self.get_queries(these_queries)
                rhs = self.get_rhs(these_queries, eval_mode=False)
                targets = self.score(q, rhs, eval_mode=False)

                query_filters = []
                query_target_types = []
                for i, query in enumerate(these_queries):
                    filter_out = set(filters[(query[0].item(), query[1].item())])
                    filter_out.add(true_entities[i])
                    query_filters.append(filter_out)

                    if entity_type_index is None:
                        query_target_types.append(-1)
                    else:
                        tail_id = true_entities[i]
                        if 0 <= tail_id < entity_type_index.shape[0]:
                            query_target_types.append(int(entity_type_index[tail_id].item()))
                        else:
                            query_target_types.append(-1)

                rank_increase = torch.zeros(query_count, device=targets.device)

                c_begin = 0
                while c_begin < total_candidates:
                    c_end = min(total_candidates, c_begin + candidate_batch_size)
                    rhs_chunk = (
                        all_candidate_embeddings[c_begin:c_end],
                        all_candidate_biases[c_begin:c_end],
                    )
                    chunk_scores = self.score(q, rhs_chunk, eval_mode=True)

                    chunk_types = None
                    if entity_type_index is not None:
                        chunk_types = entity_type_index[c_begin:c_end]

                    for i in range(query_count):
                        target_type = query_target_types[i]
                        if chunk_types is not None and target_type >= 0:
                            wrong_type_mask = (chunk_types >= 0) & (chunk_types != target_type)
                            if torch.any(wrong_type_mask):
                                chunk_scores[i, wrong_type_mask] = -1e6

                        local_filtered = [idx - c_begin for idx in query_filters[i] if c_begin <= idx < c_end]
                        if local_filtered:
                            filter_tensor = torch.tensor(
                                local_filtered,
                                dtype=torch.long,
                                device=chunk_scores.device,
                            )
                            chunk_scores[i, filter_tensor] = -1e6

                    rank_increase += torch.sum((chunk_scores >= targets).float(), dim=1)
                    c_begin = c_end
                    del chunk_scores

                ranks[b_begin:b_begin + query_count] += rank_increase.cpu()
                b_begin += batch_size
                del targets
            del entitydict
        return ranks

    def compute_metrics(self, examples, filters, batch_size=500, candidate_batch_size=50000, DAPTA_PATH=None):
        """Compute ranking-based evaluation metrics.
    
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        for m in ["rhs"]: #,"lhs"
            # pdb.set_trace()
            q = examples.clone()
            if m == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.sizes[1] // 2
            ranks = self.get_ranking(
                q,
                filters[m],
                batch_size=batch_size,
                candidate_batch_size=candidate_batch_size,
                DAPTA_PATH=DAPTA_PATH,
            )
            # print("Got Rankings & Getting Scores")
            # pdb.set_trace()
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 3, 10)
            ))))
            # print("Got Scores")
        return mean_rank, mean_reciprocal_rank, hits_at
