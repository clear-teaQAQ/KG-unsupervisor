# V11: Relation-Aware GED Training

## Research question

Can GEDRanker improve raw column-3 GED by encoding the relation labels that its
GED evaluator already uses, without adding graph-size features or changing the
edit cost?

V11 changes one model factor relative to the corrected GED-only baseline:

```text
GINConv(node features, topology)
    ->
GINEConv(node features, topology, relation embedding)
```

Both `DiffMatch` and the discriminator receive relation attributes. The
matching objective, BPR strategy, strict-GED pseudo-label update, optimizer,
200-epoch schedule, and raw best-of-100 evaluator remain unchanged.

V11 intentionally excludes V10's equal-GED exact-anchor update. Exact anchors
are measured after inference but never enter V11 training.

## Motivation

SEABED column 3 compares edge labels, but the baseline generator and
discriminator ignore `edge_features` and `edge_labels`. The label contribution
is present in nearly every nontrivial test pair:

| Dataset | Column 3 differs from no-edge-label GED | Mean difference |
| --- | ---: | ---: |
| LUBM | 9,756 / 10,000 (97.56%) | 3.5831 |
| SWDF | 9,949 / 10,000 (99.49%) | 4.4849 |

Relation-aware encoding is therefore aligned with the benchmark's local edit
definition. It is not a graph-count shortcut.

## Relation data policy

```text
embedding dimension: 100 after structured flatten
forward/reverse edge: same relation vector
self-loop: zero relation vector
direction: ignored, matching the undirected benchmark
```

SWDF serializes relation vectors as `[1, 100]`; V11 normalizes every embedding
with a shape-preserving array parser to a flat 100-vector. This is a data-format
normalization, not a dataset-specific algorithm branch.

Every load verifies edge/relation counts, dimensions, and global relation-ID
embedding consistency before training.

## Controls

Set `RELATION_MODE` to:

```text
raw       actual stable relation embeddings; formal V11 method
constant  remove relation identity while retaining topology
shuffled  deterministically break relation/topology assignment per graph
```

Controls use the same architecture. A globally permuted relation vocabulary
would preserve relation identity and is therefore not a valid no-relation
control.

## Tests

```bash
cd /data/projects/GEDRanker-main
/home/vermouth/miniconda3/envs/gedranker/bin/python \
  experiments/seabed_versions/v11_relation_aware_ged_training/test_relation_aware.py
```

## Smoke

```bash
cd /data/projects/GEDRanker-main
SMOKE=1 DATASETS="LUBM SWDF" \
  bash experiments/seabed_versions/v11_relation_aware_ged_training/train.sh
```

The one-epoch smoke is plumbing evidence only. Its GED values are not model
results.

## Formal training

Train the two nontrivial GED datasets from scratch for 200 epochs:

```bash
cd /data/projects/GEDRanker-main
DATASETS="LUBM SWDF" RELATION_MODE=raw \
  bash experiments/seabed_versions/v11_relation_aware_ged_training/train.sh
```

Then independently reload both checkpoints under seed 0:

```bash
DATASETS="LUBM SWDF" RELATION_MODE=raw \
  bash experiments/seabed_versions/v11_relation_aware_ged_training/evaluate.sh
```

The authoritative baseline rows are the independent V10-registry raw baseline
results, which load the original LUBM/SWDF 200-epoch checkpoints with the same
seed, `test_k=100`, unit cost, and no postprocessing.

## Acceptance

Primary criteria:

```text
LUBM: MAE <= 0.110 and ACC >= 0.901
SWDF: MAE < 0.341 and ACC > 0.701
```

Raw exact-anchor recall remains a secondary diagnostic. YAGO/WIKIDATA are not
run until relation-aware training improves at least one nontrivial GED dataset
without materially degrading the other.

After raw V11 acceptance, train the `constant` control first on the affected
dataset. The relation claim is supported only if removing relation identity
removes the GED gain.
