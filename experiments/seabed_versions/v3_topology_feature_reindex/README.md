# V3: Topology-Feature Reindex

## Single change from V2

V3 reconstructs the entity assigned to each topology node from the invariant:

```text
edge_indices[i] = [u, v]
KG[i]           = [source_entity, predicate, target_entity]
```

It reorders the existing node embedding records so that topology index `u`
receives `source_entity`'s embedding and index `v` receives `target_entity`'s
embedding. Isolated entities retain their original relative order. The rule is
dataset-independent and is a no-op for already consistent files.

V3 retains all other V2 behavior:

```text
training                              unchanged (none)
checkpoint                            V0 200-epoch checkpoint
candidate generation and two-swap     unchanged V1
simple and multirelation edit paths   unchanged V2
GED label and operation costs         unchanged
```

No entity embedding distance is added to GED. The corrected entity IDs are
used only for feature placement, path annotations, and the shared-entity
alignment diagnostic.

## Purpose and limitation

This version isolates data integrity. YAGO and WIKIDATA checkpoints were trained
with the old misindexed derived-graph features, so frozen-checkpoint performance
after correction is a compatibility diagnostic, not the final corrected model.
Do not interpret a regression as evidence that correct indexing is harmful.

## Run

Fast unit tests:

```bash
cd /data/projects/GEDRanker-main/experiments/seabed_versions/v3_topology_feature_reindex
/home/vermouth/miniconda3/envs/gedranker/bin/python test_topology_reindex.py
```

Four-dataset smoke run, 100 pairs per dataset and `k=5`:

```bash
cd /data/projects/GEDRanker-main
SMOKE=1 bash experiments/seabed_versions/v3_topology_feature_reindex/run.sh
```

The first smoke can focus on one unaffected and one affected dataset:

```bash
SMOKE=1 DATASETS="LUBM YAGO" \
  bash experiments/seabed_versions/v3_topology_feature_reindex/run.sh
```

Full frozen-checkpoint audit after smoke acceptance:

```bash
DATASETS="YAGO WIKIDATA" \
  bash experiments/seabed_versions/v3_topology_feature_reindex/run.sh
```

The full V3 run is limited to YAGO and WIKIDATA because those are the only
datasets changed by the reindex. LUBM and SWDF are exact no-op controls and
their four-dataset smoke results reproduce V2; their completed V2 full metrics
remain the corresponding V3 reference values.

## Acceptance rule

The loader must report:

```text
edge_consistency_after           = 1.0
fully_consistent_graphs_after    = graphs
```

Both path views must retain mapping validity, cost consistency, and replay
success of `1.0`. LUBM and SWDF must report zero changed graphs. Only after
these checks pass should the frozen V3 metrics be compared with V2.
