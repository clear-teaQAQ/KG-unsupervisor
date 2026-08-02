# V2: Executable Edit-Path Audit

## Single change from V1

V2 loads the same frozen V0 checkpoints and runs the unchanged V1 candidate
generation and `two_swap` repair. It does not train a model and does not use a
new cost to select or repair the mapping.

The only experimental change is that V2 interprets the final V1 correspondence
as two explicit, executable edit paths:

1. `simple_graph_path`: undirected endpoint, one predicate per endpoint, raw
   edge-list `last-write-wins`. This reproduces `networkx.Graph.add_edge` graph
   construction used by SEABED's `cal_GED.py`.
2. `multirelation_path`: undirected endpoint with the complete predicate
   multiset retained. This is the unified KG-aware view; it does not silently
   collapse parallel predicates.

Both views use the benchmark-supported costs only:

```text
node substitution/correspondence = 0
node insertion/deletion           = 1
edge insertion/deletion           = 1
relation substitution             = 1
direction                         = ignored
```

V2 does not add entity embedding distance, direction cost, or type cost.

## Executable path

Each saved path contains:

```text
node_correspondences
node_insertions
node_deletions
edge_insertions
edge_deletions
relation_substitutions
matched_edge_count
cost_breakdown
total_cost
```

For every evaluated pair, including pairs whose complete path is not saved, V2
checks three hard invariants:

```text
mapping is injective and in range
sum(operation costs) == path total
replay(mapped G1, operations) == target graph view
```

An invariant failure stops the run and records no accepted aggregate result.

## Metrics

Each path view reports:

- mapping validity, cost consistency, and replay success rates;
- MAE, ACC/optimal path rate, and FEA against SEABED column 3;
- signed excess cost, normalized gap, and operation-type averages;
- ranking metrics and relation-preservation rate.

V2 also reports disagreement rates among the frozen V1 dense cost, simple-graph
path cost, and multi-relation path cost. Entity-ID alignment is a secondary
proxy only because SEABED does not provide official correspondence labels.

FEA is diagnostic here. A value below 1 means that the path representation and
the provided GED label do not share the same cost semantics; it does not mean
that the path found a cost below the true optimum.

## Run

Run the six path-only unit tests:

```bash
cd /data/projects/GEDRanker-main/experiments/seabed_versions/v2_edit_path_audit
/home/vermouth/miniconda3/envs/gedranker/bin/python test_path_evaluator.py
```

Recommended four-dataset smoke audit, 100 pairs per dataset and `k=5`:

```bash
cd /data/projects/GEDRanker-main
SMOKE=1 bash experiments/seabed_versions/v2_edit_path_audit/run.sh
```

Run one smoke dataset:

```bash
SMOKE=1 DATASETS=SWDF \
  bash experiments/seabed_versions/v2_edit_path_audit/run.sh
```

After smoke acceptance, run the full frozen-checkpoint audit with `k=100`:

```bash
DATASETS="LUBM SWDF YAGO WIKIDATA" \
  bash experiments/seabed_versions/v2_edit_path_audit/run.sh
```

All pairs contribute to aggregate metrics. By default only the first 100 full
paths per dataset are stored in JSONL. Save all paths with:

```bash
MAX_SAVED_PATHS=0 DATASETS=SWDF \
  bash experiments/seabed_versions/v2_edit_path_audit/run.sh
```

Disable path JSONL while retaining all aggregate validation and metrics with
`SAVE_PATHS=0`.

## Acceptance rule

V2 smoke is accepted only when both representations have:

```text
mapping_validity_rate = 1.0
cost_consistency_rate = 1.0
replay_success_rate   = 1.0
```

The evaluator chosen for later repair/training must additionally have FEA 1.0
under the target label definition. If simple and multi-relation FEA differ by
dataset, the benchmark targets are semantically inconsistent and must be
reported as separate benchmark and KG-aware tracks rather than hidden behind a
dataset-specific implementation switch.

No 200-epoch retraining should begin until this audit is recorded in
`RESULTS.md`.
