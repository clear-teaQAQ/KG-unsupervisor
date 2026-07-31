# V0: Clean Baseline

## Hypothesis

This is the frozen reference implementation: GEDRanker is trained and evaluated
with SEABED GED column 3, raw node features, `test_k=100`, and the relation-labeled
undirected `unit` cost.

No code should be changed in this version. The implementation remains in
`src/SEABED/`; later versions load these checkpoints.

## Configuration

```text
ged_column=3
cost_mode=unit
use_raw_features=1
epochs=200
test_k=100
max_test_pairs=0
```

## Results

| Dataset | MAE | ACC | FEA | rho | tau | pk10 | pk20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 0.109 | 0.901 | 1.000 | 0.967 | 0.946 | 0.959 | 0.966 |
| SWDF | 0.338 | 0.702 | 0.936 | 0.912 | 0.864 | 0.879 | 0.932 |
| YAGO | 0.837 | 0.777 | 1.000 | 0.933 | 0.879 | 0.927 | 0.938 |
| WIKIDATA | 0.972 | 0.677 | 0.967 | 0.939 | 0.877 | 0.909 | 0.927 |

The detailed baseline and cost audit are in
`result/seabed_baseline_summary.md` and `result/seabed_cost_audit_full.md`.

## Frozen checkpoints

| Dataset | Checkpoint |
| --- | --- |
| LUBM | `model_save/LUBM_200_GEDRankerSEABED_main_col3_unit_BPR_20260726_215721.pt` |
| SWDF | `model_save/SWDF_200_GEDRankerSEABED_main_col3_unit_BPR_20260727_005926.pt` |
| YAGO | `model_save/YAGO_200_GEDRankerSEABED_main_col3_unit_BPR_20260727_040001.pt` |
| WIKIDATA | `model_save/WIKIDATA_200_GEDRankerSEABED_main_col3_unit_BPR_20260727_062631.pt` |

Checkpoint SHA-256 values:

```text
506e93c118c1c05d31604a332497ecbba5682bba443f409aea86f8b0a57fa9bb  LUBM
a075af9526d850eb70d8e5aab9a0f0d853fd438f99b6a7e969a963e76b6e0d2b  SWDF
7c03cd20901924f8d75389bfa5f3b09d3923732b2fbd2edad1efcadfbc0d7279  YAGO
e50b2a5bd3e6c7596e60db19ed4c35cdca2a1f1842b8c53f385e4e8b2333017c  WIKIDATA
```

## Interpretation

YAGO and WIKIDATA labels equal the graph-size lower bound on every test pair.
Their scalar GED metric is therefore trivial, but finding a feasible matching
whose evaluated cost reaches that bound is still a correspondence problem.

