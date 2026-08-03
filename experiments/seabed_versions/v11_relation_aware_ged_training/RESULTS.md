# V11 Results

Status: implementation and LUBM/SWDF functional smoke complete; formal
200-epoch training pending.

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

Pending. Do not copy V1 repaired GED, V10 correspondence, or smoke values into
the V11 formal table. V11 primary results must be independently evaluated raw
checkpoints trained for 200 epochs with `RELATION_MODE=raw`.
