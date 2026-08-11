# V11 Results

Status: four-dataset raw-relation 200-epoch training and independent fixed-seed
evaluation complete; relation controls pending.

## Static and unit checks

| Check | Result |
| --- | --- |
| Relation normalization/control tests | passed |
| Generator relation-attribute sensitivity | passed |
| Python compilation | passed |
| Shell syntax | passed |

## Data diagnostics

| Dataset | Graphs | Edges | Relation IDs | Dimension | Nested vectors flattened | Inconsistent IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 1,000 | 6,659 | 17 | 100 | 0 | 0 |
| SWDF | 1,000 | 8,124 | 114 | 100 | 8,124 | 0 |

Every original edge has one relation vector. Forward/reverse copies and zero
self-loop vectors align exactly with the graph edge tensor.

## Functional smoke

Both datasets completed one GED-only epoch on 16 training pairs, saved a V11
checkpoint, and evaluated five raw `k=1` test pairs with `postprocessing=none`.
The smoke values are intentionally not interpreted as performance.

Files:

```text
checkpoints/LUBM_1_GEDRankerSEABED_v11_relation_raw_col3_unit_BPR_20260803_170315.pt
checkpoints/SWDF_1_GEDRankerSEABED_v11_relation_raw_col3_unit_BPR_20260803_170433.pt
training_results/manifest_LUBM_raw_epoch1_20260803_170315.json
training_results/manifest_SWDF_raw_epoch1_20260803_170433.json
```

## Formal results

### Checkpoints

| Dataset | Checkpoint |
| --- | --- |
| LUBM | `LUBM_200_GEDRankerSEABED_v11_relation_raw_col3_unit_BPR_20260803_180220.pt` |
| SWDF | `SWDF_200_GEDRankerSEABED_v11_relation_raw_col3_unit_BPR_20260803_210556.pt` |
| YAGO | `YAGO_200_GEDRankerSEABED_v11_relation_raw_col3_unit_BPR_20260803_225008.pt` |
| WIKIDATA | `WIKIDATA_200_GEDRankerSEABED_v11_relation_raw_col3_unit_BPR_20260804_025643.pt` |

### Training-process preview

These tests ran at the end of each training process after training had consumed
the random-number stream. They verify the full configuration and show the
likely trend, but they are not the authoritative fixed-seed comparison.

| Dataset | MSE | MAE | ACC | Exact-anchor recall |
| --- | ---: | ---: | ---: | ---: |
| LUBM | 0.107 | 0.094 | 0.913 | 59.65% |
| SWDF | 0.309 | 0.266 | 0.754 | 69.80% |
| YAGO | 0.005 | 0.004 | 0.996 | 98.18% |
| WIKIDATA | 0.249 | 0.164 | 0.873 | 98.41% |

Relative to the independent raw baseline, the preview improves both primary
GED datasets: LUBM MAE changes from 0.110 to 0.094 and SWDF from 0.341 to
0.266. WIKIDATA appears to degrade from 0.130 to 0.164 in this non-authoritative
random stream; the independent fixed-seed result below supersedes that preview.

### Independent fixed-seed results

Each result independently reloads its raw 200-epoch checkpoint under seed 0.
The evaluator uses all test pairs, `test_k=100`, column-3 unit cost, and no
postprocessing.

| Dataset | Pairs | Baseline MSE | V11 MSE | Baseline MAE | V11 MAE | Baseline ACC | V11 ACC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 10,000 | 0.135 | **0.107** | 0.110 | **0.092** | 0.901 | **0.916** |
| SWDF | 10,000 | 0.434 | **0.304** | 0.341 | **0.263** | 0.701 | **0.756** |
| YAGO | 6,000 | 0.008 | **0.005** | 0.006 | **0.004** | 0.995 | **0.996** |
| WIKIDATA | 10,000 | 0.186 | **0.177** | 0.130 | **0.125** | 0.892 | **0.897** |

V11 passes both predeclared primary acceptance criteria. LUBM MAE falls by
0.018 (16.36%) and ACC rises by 1.5 percentage points. SWDF MAE falls by 0.078
(22.87%) and ACC rises by 5.5 percentage points. These are raw generator
results; no V1 two-swap, V8 anchor repair, or modified GED cost is involved.

The training-process and independent evaluations agree closely on the two
primary datasets. WIKIDATA is more sensitive to the inference random stream
(`0.164` training-process MAE versus `0.125` fixed-seed MAE), reinforcing its
secondary status and the need to compare methods under the same evaluation
seed.

Result files:

```text
raw_eval_results/result_SEABED_LUBM_test_BPR_gedcol3_unit_20260804_122031.json
raw_eval_results/result_SEABED_SWDF_test_BPR_gedcol3_unit_20260804_124851.json
raw_eval_results/result_SEABED_YAGO_test_BPR_gedcol3_unit_20260804_131713.json
raw_eval_results/result_SEABED_WIKIDATA_test_BPR_gedcol3_unit_20260804_134220.json
```

### Raw correspondence diagnostic

| Dataset | Baseline exact anchors | V11 exact anchors | Baseline recall | V11 recall | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| LUBM | 1,418 / 2,119 | 1,259 / 2,119 | 66.92% | 59.41% | -7.51 pp |
| SWDF | 584 / 884 | 608 / 884 | 66.06% | 68.78% | +2.72 pp |
| YAGO | 116,903 / 119,370 | 117,119 / 119,370 | 97.93% | 98.11% | +0.18 pp |
| WIKIDATA | 250,144 / 254,550 | 250,567 / 254,550 | 98.27% | 98.44% | +0.16 pp |

V11 improves GED on every dataset but does not uniformly improve identity
correspondence. LUBM is the clear tradeoff: relation-aware training finds lower
benchmark-cost mappings while selecting fewer exact-identity anchors. V10 and
V11 therefore provide complementary evidence rather than one dominating the
other.

### Decision

The full `raw relation + GINE` method is accepted as a raw GED improvement over
the corrected GEDRanker baseline. Causal attribution to relation identity is
still pending: the same architecture must be trained with `RELATION_MODE=constant`
on LUBM and SWDF. If constant relations retain the gain, the explanation is the
encoder change rather than relation semantics. If the gain disappears, run the
`shuffled` control to test whether correct relation/topology association is the
effective factor.
