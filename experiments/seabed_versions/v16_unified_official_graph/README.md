# V16: Unified Official Graph

## Non-negotiable objective

V16 does not redefine GED. It keeps `use_raw_features=1`, `ged_column=3`,
`cost_mode=unit`, and the original strict unit-GED preference ordering. V11
MAE/ACC remain the hard baseline.

## Controlled change from V15

V15 projects the GINE input to the undirected `nx.Graph`-compatible last-write
view but leaves candidate costs on the legacy raw edge view. V16 uses one
projected graph for every consumer:

```text
model representation
training rollout GED
BPR preference targets
best/last mapping updates
best-of-k inference selection
final MAE/ACC evaluation
```

Relation equality and every insertion/deletion/substitution cost remain unit
cost. Raw parallel relations are retained only in source data and projection
diagnostics.

## Acceptance invariants

V16 rejects execution if model and cost edge tensors differ, if projected
directed edges are duplicated/asymmetric, if any executable candidate cost is
below column-3 ground truth, or if final FEA is not 1.0.

## Commands

```bash
cd /data/projects/GEDRanker-main

SMOKE=1 DATASETS="LUBM SWDF" \
PYTHON_BIN=/home/vermouth/miniconda3/envs/gedranker/bin/python \
bash experiments/seabed_versions/v16_unified_official_graph/train.sh
```

Formal SWDF:

```bash
cd /data/projects/GEDRanker-main

CUDA_VISIBLE_DEVICES=0 DATASETS=SWDF SEED=0 \
PYTHON_BIN=/home/vermouth/miniconda3/envs/gedranker/bin/python \
bash experiments/seabed_versions/v16_unified_official_graph/train.sh
```
