#  UrbanKG Embedding 


## How to get the embedding

We establish an index mapping between entities and their learned embeddings, which is stored in **`./data/entity_idx_embedding.csv`**. To obtain the learned UrbanKG embeddings, run **`id2id.py`** followed by **`get_embedding.py`**. The resulting embeddings will be saved in the **`./embedding`** folder in `.npy` format.

## References

Some of the code was forked from the original AttH implementation which can be found at: [https://github.com/HazyResearch/KGEmb](https://github.com/HazyResearch/KGEmb)

## Two-Stage GIE -> TimePlex Workflow

This repository now supports a two-stage KG embedding pipeline:

1. Train `GIE` on static triples.
2. Train `TimePlex` on temporal quadruples.
3. Fuse static + temporal embeddings for downstream tasks.

### Step 1: Prepare static and temporal data

The script below consumes CSV files:

- `data/<DATASET>/static_train.csv`
- `data/<DATASET>/static_valid.csv`
- `data/<DATASET>/static_test.csv`
- `data/<DATASET>/temporal_train.csv`
- `data/<DATASET>/temporal_valid.csv`
- `data/<DATASET>/temporal_test.csv`

and produces:

- GIE files: `train`, `valid`, `test`, `.pickle`, `to_skip.pickle` in `data/<DATASET>`
- TimePlex files: `train.txt`, `valid.txt`, `test.txt`, `intervals/*` in `tkbi-master/data/<DATASET>_temporal`

```bash
python prepare_hybrid_data.py --dataset NYC
```

### Step 2: Train GIE on the static KG

```bash
python run.py --dataset NYC --model GIE --data_path data --rank 200
```

### Step 3: Train TimePlex on the temporal KG

```bash
cd tkbi-master
python main.py ^
  -m TimePlex_base ^
  -d NYC_temporal ^
  --mode train ^
  -a "{\"embedding_dim\":200, \"srt_wt\":5.0, \"ort_wt\":5.0, \"sot_wt\":5.0, \"emb_reg_wt\":0.005}" ^
  -l crossentropy_loss_AllNeg ^
  -r 0.1 -b 1000 -x 500 -n 0 -y 100 ^
  -g_reg 2 -g 1.0 --flag_add_reverse 1 ^
  -e 100 --save_dir nyc_timeplex_base
```

### Optional: Hybrid GIE -> TimePlex initialization + anchor/context losses

1. Export static embeddings from trained GIE:

```bash
python export_static_gie_embeddings.py ^
  --dataset NYC ^
  --data_path data ^
  --checkpoint logs/<date>/NYC/GIE_<time>/model.pt ^
  --config logs/<date>/NYC/GIE_<time>/config.json ^
  --output_dir pretrained/NYC_GIE
```

2. Train `TimePlex_base` with hybrid constraints:

```bash
cd tkbi-master
python main.py ^
  -m TimePlex_base ^
  -d NYC_temporal ^
  --mode train ^
  -a "{\"embedding_dim\":200, \"srt_wt\":5.0, \"ort_wt\":5.0, \"sot_wt\":5.0, \"emb_reg_wt\":0.005, \"gie_entity_path\":\"../pretrained/NYC_GIE/static_entity_embedding.npy\", \"anchor_lambda\":0.1, \"context_lambda\":0.05, \"entity_align_path\":\"../data/NYC/entity2id.csv\", \"context_neighbors_path\":\"../data/NYC/context_neighbors.json\", \"entity_type_path\":\"../data/NYC/entity_types.csv\", \"hybrid_loss_scope\":\"batch\"}" ^
  -l crossentropy_loss_AllNeg ^
  -r 0.1 -b 1000 -x 500 -n 0 -y 100 ^
  -g_reg 2 -g 1.0 --flag_add_reverse 1 ^
  -e 100 --save_dir nyc_timeplex_hybrid
```

Notes:

- Shared entities: initialized from projected GIE embeddings and supervised by `L_anchor`.
- Temporal-only entities: initialized by context aggregation first, then type prototype, otherwise random.
- TimePlex relations remain independent from GIE relations (no relation parameter sharing).

### Step 4: Export fused GIE + TimePlex embeddings

```bash
python get_embedding.py ^
  --dataset NYC ^
  --data_path data ^
  --gie_checkpoint logs/<date>/NYC/GIE_<time>/model.pt ^
  --gie_config logs/<date>/NYC/GIE_<time>/config.json ^
  --timeplex_checkpoint tkbi-master/models/nyc_timeplex_base/best_valid_model.pt ^
  --fuse_mode concat ^
  --output_dir embedding
```

This exports:

- `embedding/<DATASET>_entity_hybrid_embedding.npy`
- `embedding/<DATASET>_region_hybrid_embedding.npy`
- `embedding/<DATASET>_POI_hybrid_embedding.npy`
- `embedding/<DATASET>_Road_hybrid_embedding.npy`
- `embedding/<DATASET>_region_temporal_embedding.npy` with shape `[T, N_region, D]`
- `embedding/<DATASET>_region_temporal_mask.npy` with shape `[T, N_region]`
- `embedding/<DATASET>_time_embedding.npy`
- `embedding/<DATASET>_region_temporal_index.json`

`<DATASET>_region_temporal_embedding.npy` is the region-time KG tensor for downstream
spatiotemporal prediction models. The JSON file records the time axis and region axis,
including the original TimePlex time ids used to build each temporal slice.
