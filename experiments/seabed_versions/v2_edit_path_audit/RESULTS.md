# V2 Results

Status: complete. Functional, smoke, and full `k=100` evaluations finished.

## Configuration

```text
training changed: no
checkpoint source: V0 200-epoch checkpoints
mapping source: frozen V1 deterministic_dense_v4 + two_swap
path revision: dual_executable_path_v1
GED label: column 3
```

## Smoke results

Functional test:

```text
dataset=SWDF, pairs=10, k=2
simple path:        MAE=0.300, ACC=0.700, FEA=1.000
multirelation path: MAE=4.200, ACC=0.000, FEA=1.000
all six executable-path invariants passed
```

Command:

```bash
SMOKE=1 DATASETS=SWDF MAX_TEST_PAIRS=10 TEST_K_SMOKE=2 \
  bash experiments/seabed_versions/v2_edit_path_audit/run.sh
```

Four-dataset smoke command:

```bash
SMOKE=1 DATASETS="LUBM SWDF YAGO WIKIDATA" MAX_TEST_PAIRS=100 \
  TEST_K_SMOKE=5 MAX_SAVED_PATHS=100 \
  bash experiments/seabed_versions/v2_edit_path_audit/run.sh
```

Required acceptance fields for both path representations:

| Dataset | View | Pairs | MAE | ACC | FEA | Mapping valid | Cost consistent | Replay success | Mean excess | Dense disagreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | simple | 100 | 0.210 | 0.790 | 1.000 | 1.000 | 1.000 | 1.000 | 0.210 | 0.00% |
| LUBM | multirelation | 100 | 0.210 | 0.790 | 1.000 | 1.000 | 1.000 | 1.000 | 0.210 | 0.00% |
| SWDF | simple | 100 | 0.390 | 0.660 | 1.000 | 1.000 | 1.000 | 1.000 | 0.390 | 13.00% |
| SWDF | multirelation | 100 | 3.470 | 0.010 | 1.000 | 1.000 | 1.000 | 1.000 | 3.470 | 98.00% |
| YAGO | simple | 100 | 0.930 | 0.830 | 1.000 | 1.000 | 1.000 | 1.000 | 0.930 | 0.00% |
| YAGO | multirelation | 100 | 0.930 | 0.830 | 1.000 | 1.000 | 1.000 | 1.000 | 0.930 | 0.00% |
| WIKIDATA | simple | 100 | 1.110 | 0.720 | 1.000 | 1.000 | 1.000 | 1.000 | 1.110 | 7.00% |
| WIKIDATA | multirelation | 100 | 1.170 | 0.690 | 1.000 | 1.000 | 1.000 | 1.000 | 1.170 | 11.00% |

## Evaluator audit

| Dataset | Simple vs V1 dense | Multi vs V1 dense | Simple vs multi | Mean multi - simple |
| --- | ---: | ---: | ---: | ---: |
| LUBM | 0.00% | 0.00% | 0.00% | 0.00 |
| SWDF | 13.00% | 98.00% | 98.00% | 3.08 |
| YAGO | 0.00% | 0.00% | 0.00% | 0.00 |
| WIKIDATA | 7.00% | 11.00% | 5.00% | 0.06 |

The simple executable path corrected the below-label V1 dense costs in this
sample: its FEA is 1.0 on all datasets, while the dense reference FEA is 0.94 on
SWDF and 0.95 on WIKIDATA. SWDF column 3 is clearly a simple-graph target: the
multi-relation path adds 3.08 operations on average and disagrees on 98% of
pairs. LUBM and YAGO contain no effective difference in this sample. WIKIDATA
has a smaller but nonzero multi-relation difference.

## Semantic proxy

| Dataset | Shared entities | Aligned shared entities | Alignment rate |
| --- | ---: | ---: | ---: |
| LUBM | 43 | 28 | 65.12% |
| SWDF | 1 | 0 | 0.00% |
| YAGO | 2050 | 156 | 7.61% |
| WIKIDATA | 2700 | 157 | 5.81% |

SWDF has too little entity overlap for this proxy. LUBM also has a small
denominator. YAGO and WIKIDATA provide strong evidence that low-cost structural
paths do not imply semantically faithful entity correspondence.

Path construction is inexpensive relative to inference: 0.00049--0.00180
seconds per pair in this smoke run, versus 0.09649--0.10250 seconds per pair for
frozen V1 inference.

## Full results

| Dataset | Pairs | View | MAE | ACC / optimal path | FEA | Mapping valid | Cost consistent | Replay success |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 10000 | simple | 0.102 | 0.908 | 1.000 | 1.000 | 1.000 | 1.000 |
| LUBM | 10000 | multirelation | 0.102 | 0.908 | 1.000 | 1.000 | 1.000 | 1.000 |
| SWDF | 10000 | simple | 0.234 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| SWDF | 10000 | multirelation | 2.721 | 0.086 | 0.998 | 1.000 | 1.000 | 1.000 |
| YAGO | 6000 | simple | 0.473 | 0.873 | 1.000 | 1.000 | 1.000 | 1.000 |
| YAGO | 6000 | multirelation | 0.474 | 0.873 | 1.000 | 1.000 | 1.000 | 1.000 |
| WIKIDATA | 10000 | simple | 0.349 | 0.881 | 0.973 | 1.000 | 1.000 | 1.000 |
| WIKIDATA | 10000 | multirelation | 0.330 | 0.904 | 1.000 | 1.000 | 1.000 | 1.000 |

Both path representations passed all three executable-path invariants on all
36,000 full-test pairs. Each result saved 100 representative JSONL paths.

Comparison with the unchanged V1 dense score for the same mapping:

| Dataset | V1 dense MAE / ACC / FEA | Simple MAE / ACC / FEA | Multi MAE / ACC / FEA |
| --- | --- | --- | --- |
| LUBM | 0.102 / 0.908 / 1.000 | 0.102 / 0.908 / 1.000 | 0.102 / 0.908 / 1.000 |
| SWDF | 0.258 / 0.773 / 0.973 | 0.234 / 0.800 / 1.000 | 2.721 / 0.086 / 0.998 |
| YAGO | 0.473 / 0.873 / 1.000 | 0.473 / 0.873 / 1.000 | 0.474 / 0.873 / 1.000 |
| WIKIDATA | 0.367 / 0.863 / 0.955 | 0.349 / 0.881 / 0.973 | 0.330 / 0.904 / 1.000 |

The evaluator audit changes only the cost assigned to the frozen V1 mapping.
It improves SWDF simple-path MAE by 9.30% and WIKIDATA multi-relation MAE by
10.08% relative to the V1 dense score, while restoring FEA to 1.0 in the
compatible view.

Full evaluator disagreement:

| Dataset | Simple vs dense | Multi vs dense | Simple vs multi | Mean multi - simple |
| --- | ---: | ---: | ---: | ---: |
| LUBM | 0.00% | 0.00% | 0.00% | 0.0000 |
| SWDF | 6.30% | 90.60% | 90.24% | 2.4836 |
| YAGO | 0.07% | 0.10% | 0.03% | 0.0003 |
| WIKIDATA | 2.04% | 5.52% | 3.61% | 0.0396 |

WIKIDATA reverses the smoke-only conclusion: its simple path is below the label
on 2.73% of full pairs, whereas its multi-relation path has FEA 1.0 and better
MAE/ACC. Therefore the four column-3 targets do not have one uniform edge
semantics. SWDF is simple-graph compatible; WIKIDATA is more consistent with
the raw multi-relation view. This is a benchmark construction fact, not a reason
to add hard-coded dataset branches to the method.

Full semantic proxy as originally loaded:

| Dataset | Shared entities | Aligned shared entities | Alignment rate |
| --- | ---: | ---: | ---: |
| LUBM | 2119 | 1417 | 66.87% |
| SWDF | 884 | 587 | 66.40% |
| YAGO | 119370 | 9028 | 7.56% |
| WIKIDATA | 254550 | 16118 | 6.33% |

These rates are not valid mapping-quality measurements for YAGO/WIKIDATA. The
data-integrity audit below found that their derived graphs attach `node_features`
to a different node order from `edge_indices`. The table is retained because it
was the signal that exposed the issue, but it must not be cited as evidence that
the model learned semantically incorrect correspondence.

## Feature-topology integrity incident

For each raw edge, the audit checked all three identities:

```text
node_features[edge_indices[i][0]].id == KG[i][0]
edge_features[i].id                  == KG[i][1]
node_features[edge_indices[i][1]].id == KG[i][2]
```

| Dataset | Graphs | Consistent edges / edges | Edge consistency | Fully consistent graphs |
| --- | ---: | ---: | ---: | ---: |
| LUBM | 1000 | 6659 / 6659 | 100.00% | 1000 / 1000 |
| SWDF | 1000 | 8124 / 8124 | 100.00% | 1000 / 1000 |
| YAGO | 31000 | 28607 / 841353 | 3.40% | 1000 / 31000 |
| WIKIDATA | 51000 | 37615 / 1695173 | 2.22% | 1000 / 51000 |

In every YAGO/WIKIDATA split, the original base graphs are consistent and the
derived graphs are not. Their `node_features` lists were reordered without the
same permutation being applied to `edge_indices`. Consequently, entity IDs and
embeddings are attached to the wrong topology nodes in the current SEABED files.

This does not invalidate V2's structural path costs or replay checks: both use
the same authoritative `edge_indices` and `edge_features`. It does invalidate
entity-name annotations and shared-entity alignment for the affected derived
graphs. Those fields must be treated as diagnostic-only until reindexing.

Recovery is deterministic. For each `edge_indices[i]=[u,v]` and
`KG[i]=[s,p,o]`, assign entity `s` to node index `u` and entity `o` to node index
`v`, then reorder the existing feature records by ID. On the 100 saved full-run
paths for each affected dataset, an identity mapping under these reconstructed
IDs reached the supplied GT GED in both path views for 100/100 YAGO and 100/100
WIKIDATA pairs. It also had exactly the current simple and multi costs on 89% of
YAGO pairs and 75% of WIKIDATA pairs.

Full result files:

```text
results/result_SEABED_v2_edit_path_audit_LUBM_test_k100_two_swap_20260801_133217.json
results/result_SEABED_v2_edit_path_audit_SWDF_test_k100_two_swap_20260801_134936.json
results/result_SEABED_v2_edit_path_audit_YAGO_test_k100_two_swap_20260801_140641.json
results/result_SEABED_v2_edit_path_audit_WIKIDATA_test_k100_two_swap_20260801_142353.json
```

## Decision

V2 is accepted and closed. Retain both executable views: simple graph for
benchmark projection and multi-relation for the unified KG path. Do not silently
compare a multi-relation SWDF cost with its simple-graph column-3 label, and do
not use a simple WIKIDATA path as proof of feasibility against its raw-edge
label.

The next isolated version must fix feature-topology indexing before adding a
semantic objective. Reconstruct node order from `KG + edge_indices`, reorder the
existing node feature records by entity ID, and assert graph-wide consistency.
This is one universal integrity rule, not a dataset-name branch; it is a no-op on
already consistent LUBM/SWDF graphs.

First evaluate the corrected loader with frozen checkpoints to isolate the data
factor. Because YAGO/WIKIDATA checkpoints were trained on misaligned derived
features, this run diagnoses compatibility and does not establish a final model.
Only after that audit should a later version add dual-cost semantic tie-breaking
or retrain for 200 epochs on corrected features.
