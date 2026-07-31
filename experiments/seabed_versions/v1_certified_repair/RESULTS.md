# V1 Results

Status: LUBM full evaluation complete; SWDF/YAGO/WIKIDATA full evaluations are pending.

## Full result

| Dataset | Pairs | k | Initial MAE | Final MAE | Initial ACC | Final ACC | Improved pairs | Initial evaluator-bound hit | Final evaluator-bound hit | Time/pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 10000 | 100 | 0.110 | 0.102 | 0.901 | 0.908 | 0.0075 | 0.0151 | 0.0152 | 0.17083 s |

LUBM result file:

```text
results/result_SEABED_v1_certified_repair_LUBM_test_k100_two_swap_20260731_203827.json
```

## Cost incident

The first full SWDF run stopped at pair 2778 because the original V1 fast cost
decomposition returned 12 while V0's exact evaluator returned 11. The protective
equality check prevented an inconsistent repair result from being recorded.

Cause: V0 stores one relation label per dense adjacency cell. Reciprocal or
multi-relation edges can overwrite that cell and leave asymmetric relation labels.
The old decomposition assumed this could not affect the unmatched-edge term.

The first correction, `exact_v0_permutation_v2`, made every repair candidate
execute V0's padded upper-triangle comparison. A repeated failure at the same
pair then showed that V0's inference API returned only the first `n1` rows of the
`n2 x n2` solution, losing the exact full permutation used for candidate scoring.

Revision `full_permutation_v3` expanded the V1 parallel
inference path, retains all complete permutations, validates row/column
injectivity, selects with V0's exact evaluator, and passes the selected complete
permutation to repair. A third full SWDF attempt then exposed the actual lowest
level cause at pair 1698: the same complete permutation evaluated to 14 and 15.

Some KG graphs contain multiple relation labels for the same dense adjacency
cell. V0 constructs adjacency on CUDA with advanced-index assignment. Duplicate
index writes have no guaranteed winner on CUDA, so rebuilding the same adjacency
can select a different relation label and change the cost.

The current revision is `deterministic_dense_v4`. It constructs adjacency once
per pair on CPU in edge-list order with explicit `last-write-wins` semantics,
then shares that matrix between candidate selection and repair. Eight tests pass,
including duplicate-cell overwrite order, asymmetric labels, complete-permutation
equivalence, and non-injective mapping rejection. The failing pair 1698 and its
saved permutation were replayed 100 times in scalar and batched modes; all 200
evaluations returned the same cost, 14.

Graph audit:

| Dataset | Graphs | Graphs with asymmetric relation labels |
| --- | ---: | ---: |
| LUBM | 1000 | 0 |
| SWDF | 1000 | 484 |
| YAGO | 31000 | 101 |
| WIKIDATA | 51000 | 9265 |

Graphs with conflicting relation labels for the same undirected endpoint pair:

| Dataset | Graphs | Graphs with conflicting dense cells |
| --- | ---: | ---: |
| LUBM | 1000 | 0 |
| SWDF | 1000 | 678 |
| YAGO | 31000 | 204 |
| WIKIDATA | 51000 | 9265 |

The completed LUBM result remains valid because all LUBM effective adjacency
matrices are symmetric, making the old and exact formulas equivalent for every
mapping. SWDF must restart; YAGO and WIKIDATA must use `deterministic_dense_v4`.

## Preliminary smoke result

These pre-fix exploratory runs tested the isolated V1 change on the first 100 test pairs of each
dataset. They use `test_k=5`, so they must not be compared directly with the V0
`test_k=100` table.
The table is retained as an experiment record, but SWDF/YAGO/WIKIDATA final
columns are not accepted results because they predate `deterministic_dense_v4`.

| Dataset | Pairs | k | Initial MAE | Final MAE | Initial ACC | Final ACC | Improved pairs | Initial evaluator-bound hit | Final evaluator-bound hit | Time/pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 100 | 5 | 0.360 | 0.320 | 0.700 | 0.720 | 0.040 | 0.000 | 0.000 | 0.08876 s |
| SWDF | 100 | 5 | 0.490 | 0.480 | 0.570 | 0.570 | 0.070 | 0.000 | 0.000 | 0.08770 s |
| YAGO | 100 | 5 | 2.140 | 1.030 | 0.570 | 0.820 | 0.350 | 0.570 | 0.820 | 0.20053 s |
| WIKIDATA | 100 | 5 | 4.280 | 1.150 | 0.220 | 0.640 | 0.660 | 0.180 | 0.510 | 0.27830 s |

Additional diagnostics:

```text
average cost reduction:       1.11
maximum cost reduction:       12
average repair iterations:    0.59
average candidates evaluated: 223.27
label equals evaluator bound: 1.00
```

The same quantity is `0.00` for LUBM/SWDF and `0.74` for this WIKIDATA subset.
WIKIDATA's label still equals the raw graph-size bound; the discrepancy comes
from V0 collapsing multiple relation edges into one dense adjacency cell. This
means the certificate is currently a certificate for the V0 evaluator, not yet
for a multi-edge KG edit path.

Command:

```bash
SMOKE=1 DATASETS=YAGO REPAIR_MODE=two_swap \
  bash experiments/seabed_versions/v1_certified_repair/run.sh

SMOKE=1 DATASETS="LUBM SWDF WIKIDATA" REPAIR_MODE=two_swap \
  bash experiments/seabed_versions/v1_certified_repair/run.sh
```

Result files:

```text
results/result_SEABED_v1_certified_repair_YAGO_test_k5_two_swap_20260731_200928.json
results/result_SEABED_v1_certified_repair_LUBM_test_k5_two_swap_20260731_201117.json
results/result_SEABED_v1_certified_repair_SWDF_test_k5_two_swap_20260731_201128.json
results/result_SEABED_v1_certified_repair_WIKIDATA_test_k5_two_swap_20260731_201140.json
```

## Run record

| Field | Value |
| --- | --- |
| Date | 2026-07-31 |
| Git commit | `64bc1dd` plus the uncommitted isolated V1 files |
| Command | `bash experiments/seabed_versions/v1_certified_repair/run.sh` |
| CUDA / device | CUDA for full run; smoke was CPU |
| `test_k` | 100 |
| `repair_max_iterations` | 20 |

## Metrics

| Dataset | Initial MAE | Final MAE | Initial ACC | Final ACC | Improved pairs | Initial bound hit | Final bound hit | Time/pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 0.110 | 0.102 | 0.901 | 0.908 | 0.0075 | 0.0151 | 0.0152 | 0.17083 s |
| SWDF | pending | pending | pending | pending | pending | pending | pending | pending |
| YAGO | pending | pending | pending | pending | pending | pending | pending | pending |
| WIKIDATA | pending | pending | pending | pending | pending | pending | pending | pending |

## Conclusion

Pending.
