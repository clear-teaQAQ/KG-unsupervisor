# V11 Results

Status: four-dataset raw-relation 200-epoch training complete; independent
fixed-seed checkpoint evaluation pending.

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
0.266. WIKIDATA degrades from 0.130 to 0.164, so relation-aware message passing
is not yet accepted as a uniformly beneficial method.

### Independent fixed-seed results

Pending. The `raw_eval_results` directory contained no result files after
training. Do not copy V1 repaired GED, V10 correspondence, smoke values, or the
training-process preview into the authoritative V11 table. V11 primary results
must independently reload the raw 200-epoch checkpoints under seed 0 with
`test_k=100`, column-3 unit cost, and no postprocessing.
