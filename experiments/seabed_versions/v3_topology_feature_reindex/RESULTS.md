# V3 Results

Status: complete. Unit tests, four-dataset smoke, and affected-dataset full
evaluation passed.

## Fixed configuration

```text
training changed: no
checkpoint source: V0 200-epoch checkpoints
mapping pipeline: frozen V1 deterministic_dense_v4 + two_swap
path evaluator: unchanged V2 dual_executable_path_v1
GED label: column 3
feature revision: kg_edge_topology_reindex_v1
```

## Functional tests

Five topology-reindex unit tests pass. They cover unchanged consistent graphs,
shuffled features, isolated nodes, conflicting topology assignments, and
predicate mismatches.

## Smoke results

Command:

```bash
SMOKE=1 DATASETS="LUBM YAGO" \
  bash experiments/seabed_versions/v3_topology_feature_reindex/run.sh
```

Feature-topology integrity:

| Dataset | Graphs | Changed graphs | Reassigned nodes | Edge consistency before | Edge consistency after | Fully consistent before | Fully consistent after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 1,000 | 0 | 0 | 100.00% | 100.00% | 1,000 | 1,000 |
| SWDF | 1,000 | 0 | 0 | 100.00% | 100.00% | 1,000 | 1,000 |
| YAGO | 31,000 | 30,000 | 677,833 | 3.40% | 100.00% | 1,000 | 31,000 |
| WIKIDATA | 51,000 | 50,000 | 1,392,387 | 2.22% | 100.00% | 1,000 | 51,000 |

Frozen-checkpoint path metrics on 100 pairs with `k=5`:

| Dataset | Version/view | MAE | ACC | FEA | Mapping valid | Cost consistent | Replay success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | V2 simple/multi | 0.210 | 0.790 | 1.000 | 1.000 | 1.000 | 1.000 |
| LUBM | V3 simple/multi | 0.210 | 0.790 | 1.000 | 1.000 | 1.000 | 1.000 |
| SWDF | V2 simple | 0.390 | 0.660 | 1.000 | 1.000 | 1.000 | 1.000 |
| SWDF | V3 simple | 0.390 | 0.660 | 1.000 | 1.000 | 1.000 | 1.000 |
| SWDF | V2 multi | 3.470 | 0.010 | 1.000 | 1.000 | 1.000 | 1.000 |
| SWDF | V3 multi | 3.470 | 0.010 | 1.000 | 1.000 | 1.000 | 1.000 |
| YAGO | V2 simple/multi | 0.930 | 0.830 | 1.000 | 1.000 | 1.000 | 1.000 |
| YAGO | V3 simple/multi | 0.660 | 0.890 | 1.000 | 1.000 | 1.000 | 1.000 |
| WIKIDATA | V2 simple | 1.110 | 0.720 | 1.000 | 1.000 | 1.000 | 1.000 |
| WIKIDATA | V3 simple | 0.410 | 0.860 | 1.000 | 1.000 | 1.000 | 1.000 |
| WIKIDATA | V2 multi | 1.170 | 0.690 | 1.000 | 1.000 | 1.000 | 1.000 |
| WIKIDATA | V3 multi | 0.470 | 0.830 | 1.000 | 1.000 | 1.000 | 1.000 |

LUBM and SWDF are exact no-op controls: feature placement, costs, and
correspondence metrics reproduce V2. On YAGO, correcting the inputs used by the
frozen model reduces smoke MAE by 29.03% and increases ACC by 0.06. On
WIKIDATA, it reduces simple-view MAE by 63.06% and multi-view MAE by 59.83%,
while increasing simple/multi ACC by 0.14. These improvements are encouraging,
but remain smoke results from checkpoints trained on the old misindexed inputs.

The corrected shared-entity alignment rates are 65.12% for LUBM, 81.61% for
YAGO, and 84.96% for WIKIDATA. SWDF has only one shared entity in the smoke
sample and is not informative for this metric. The original YAGO/WIKIDATA rates
of 7.61%/5.81% were invalid because they compared entity IDs using the corrupted
feature order.

## Provenance check

The feature-topology mismatch predates the `05_constant_features` ablation.
That experiment reads graph JSON with `json.load`, creates transformed feature
records in memory with `zeros_like`/`ones_like`, and never writes graph payloads
back to the shared data directory. The YAGO graph files were installed on
2026-07-09 and retain 2024 modification timestamps; the recorded constant-feature
YAGO run started on 2026-07-13.

The mismatch also has a generation-specific pattern: all 1,000 base YAGO graphs
are consistent and all 30,000 derived graphs are inconsistent. This is evidence
of a derived-graph indexing defect in the supplied data, not an ablation-time
feature transformation.

Result files:

```text
results/result_SEABED_v3_topology_feature_reindex_LUBM_test_k5_two_swap_20260801_172208.json
results/result_SEABED_v3_topology_feature_reindex_SWDF_test_k5_two_swap_20260801_173328.json
results/result_SEABED_v3_topology_feature_reindex_YAGO_test_k5_two_swap_20260801_172221.json
results/result_SEABED_v3_topology_feature_reindex_WIKIDATA_test_k5_two_swap_20260801_173341.json
```

## Smoke decision

V3 smoke is accepted. The universal reindex is a no-op on both consistent
datasets, repairs every edge on both affected datasets, and preserves every V2
path invariant. Proceed to full frozen-checkpoint evaluation on the affected
YAGO and WIKIDATA datasets before deciding the corrected-training baseline.

## Full results

YAGO and WIKIDATA were evaluated on all test pairs with `k=100`. LUBM and SWDF
are exact no-op datasets under the reindex, so their completed V2 full results
are retained as the corresponding V3 references.

| Dataset | Pairs | Version/view | MAE | ACC | FEA | Mapping valid | Cost consistent | Replay success |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 10,000 | V2/V3 simple | 0.102 | 0.908 | 1.000 | 1.000 | 1.000 | 1.000 |
| LUBM | 10,000 | V2/V3 multi | 0.102 | 0.908 | 1.000 | 1.000 | 1.000 | 1.000 |
| SWDF | 10,000 | V2/V3 simple | 0.234 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| SWDF | 10,000 | V2/V3 multi | 2.721 | 0.086 | 0.998 | 1.000 | 1.000 | 1.000 |
| YAGO | 6,000 | V2 simple | 0.473 | 0.873 | 1.000 | 1.000 | 1.000 | 1.000 |
| YAGO | 6,000 | V3 simple | 0.282 | 0.922 | 1.000 | 1.000 | 1.000 | 1.000 |
| YAGO | 6,000 | V2 multi | 0.474 | 0.873 | 1.000 | 1.000 | 1.000 | 1.000 |
| YAGO | 6,000 | V3 multi | 0.282 | 0.922 | 1.000 | 1.000 | 1.000 | 1.000 |
| WIKIDATA | 10,000 | V2 simple | 0.349 | 0.881 | 0.973 | 1.000 | 1.000 | 1.000 |
| WIKIDATA | 10,000 | V3 simple | 0.162 | 0.928 | 0.971 | 1.000 | 1.000 | 1.000 |
| WIKIDATA | 10,000 | V2 multi | 0.330 | 0.904 | 1.000 | 1.000 | 1.000 | 1.000 |
| WIKIDATA | 10,000 | V3 multi | 0.137 | 0.954 | 1.000 | 1.000 | 1.000 | 1.000 |

Relative to V2, V3 reduces YAGO simple-view MAE by 40.38% and WIKIDATA
multi-view MAE by 58.48%. ACC increases by 0.049 and 0.050 respectively. The
corrected full shared-entity alignment rates are 84.60% for YAGO and 90.47% for
WIKIDATA, compared with the invalid old-order rates of 7.56% and 6.33%.

WIKIDATA continues to support the multirelation benchmark view: it is feasible
on every pair and has lower MAE/higher ACC than the simple projection. YAGO's
simple and multirelation views remain effectively identical. V3 therefore does
not change V2's conclusion about mixed column-3 edge semantics.

Full result files:

```text
results/result_SEABED_v3_topology_feature_reindex_YAGO_test_k100_two_swap_20260801_174425.json
results/result_SEABED_v3_topology_feature_reindex_WIKIDATA_test_k100_two_swap_20260801_180204.json
```

Each run saved 100 executable paths. All path invariants passed on all 16,000
evaluated pairs.

## Final decision

V3 is accepted and closed. Feature-topology reindexing is a required universal
data-integrity step before any semantic method. It is not presented as a new GED
cost and does not use GED labels or cross-graph correspondence.

The next isolated version must retrain the unchanged GEDRanker generator with
the corrected loader. The V3 checkpoint was trained on the old misindexed
YAGO/WIKIDATA inputs and evaluated on corrected inputs, so V3 is strong evidence
for the repair but is not a train/test-consistent final model. Do not add a
semantic tie-break or embedding cost until that corrected-training baseline is
recorded.
