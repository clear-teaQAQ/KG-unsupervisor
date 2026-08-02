# V8 Results

Status: implementation complete. Four-dataset inference passed, but V9 shows
that YAGO/WIKIDATA GED accuracy is fully size-determined and not matching
evidence.

## Affected-dataset smoke

YAGO and WIKIDATA were run by the user on CUDA with 100 prefix pairs and
`test_k=5`.

| Dataset | Anchors before | Anchors after | Changed pairs | Anchor-improved pairs | Alignment-improved pairs | Harmed pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YAGO | 1,976 | 2,050 | 41 | 41 | 41 | 0 |
| WIKIDATA | 2,680 | 2,700 | 17 | 17 | 17 | 0 |

Both datasets reached 100% shared-entity alignment. Every mapping change added
an exact embedding anchor. Simple and multirelation cost-changed pair counts
were both zero. Both executable paths had mapping validity, cost consistency,
and replay success rates of 1.0.

A direct per-pair comparison against the CUDA V6 smoke found zero mismatches in
graph-pair identity, final mapping, simple cost, or multirelation cost for all
100 YAGO and all 100 WIKIDATA pairs. V8 therefore retains V6 behavior where
identity evidence is available while removing unsupported control-dataset
changes.

## Control-dataset smoke

LUBM and SWDF were run in the Codex tool environment on CPU because CUDA was
not exposed there. This run validates V8 behavior and invariants; its stochastic
GED aggregates must not be mixed with formal CUDA results.

| Dataset | Anchors before | Anchors after | Changed pairs | Anchor-improved pairs | Alignment-improved pairs | Harmed pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 26 | 28 | 2 | 2 | 2 | 0 |
| SWDF | 0 | 0 | 0 | 0 | 0 | 0 |

This is the intended control behavior. Compared with V6's broad cosine smoke,
mapping changes contract from 83 to 2 on LUBM and from 78 to 0 on SWDF. V8
retains only changes backed by an added exact identity anchor.

Both protected cost-changed pair counts remained zero. Both path views passed
all validity, cost-consistency, and replay checks, and all 200 saved operation
breakdowns summed to their recorded total costs.

## Decision

V8 is accepted for full evaluation. It does not claim GED improvement over the
pre-anchor structural mapping. Formal results must report unchanged GED/path
costs and the separate gain in exact-anchor/shared-entity correspondence.

Smoke result files:

```text
results/result_SEABED_v8_exact_embedding_anchor_YAGO_test_k5_two_swap_20260802_132304.json
results/result_SEABED_v8_exact_embedding_anchor_WIKIDATA_test_k5_two_swap_20260802_132441.json
results/result_SEABED_v8_exact_embedding_anchor_LUBM_test_k5_two_swap_20260802_132939.json
results/result_SEABED_v8_exact_embedding_anchor_SWDF_test_k5_two_swap_20260802_132951.json
```

## Affected-dataset full results

The formal runs used `test_k=100`, all test pairs, and the corresponding V4
corrected-training checkpoints.

| Dataset | Pairs | Alignment before | Alignment after | Added aligned entities | Changed/improved pairs | Harmed pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YAGO | 6,000 | 116,915 / 119,370 (97.9434%) | 119,358 / 119,370 (99.9899%) | 2,443 | 1,551 | 0 |
| WIKIDATA | 10,000 | 250,161 / 254,550 (98.2758%) | 254,540 / 254,550 (99.9961%) | 4,379 | 2,220 | 0 |

Only 12 shared-entity observations remain unaligned on YAGO and 10 on
WIKIDATA. The local exact-anchor search does not force these correspondences
when no accepted matched swap or unmatched-target replacement can increase the
anchor count while preserving both path costs.

GED and executable-path results are unchanged from the corresponding V4 full
runs:

| Dataset | Dense MAE / ACC / FEA | Simple MAE / ACC / FEA | Multi MAE / ACC / FEA |
| --- | --- | --- | --- |
| YAGO | 0.001 / 0.999 / 0.999 | 0 / 1 / 1 | 0 / 1 / 1 |
| WIKIDATA | 0.056 / 0.948 / 0.948 | 0.033 / 0.969 / 0.969 | 0.000 / 1.000 / 1.000 |

Both datasets report zero simple and multirelation cost-changed pairs. All
mapping-validity, cost-consistency, and replay-success rates are 1.0. No
alignment was harmed.

The first 100 saved full paths were compared directly with V4. YAGO changes 42
mappings and improves alignment from 1,974 to 2,050; WIKIDATA changes 17 and
improves alignment from 2,680 to 2,700. Both datasets have zero simple-cost and
zero multirelation-cost mismatches, and every saved operation breakdown sums to
its total cost.

Measured exact-anchor search time is 0.081923 seconds per YAGO pair and
0.147965 seconds per WIKIDATA pair, approximately 8.2 and 24.7 minutes over the
full datasets. This is reported separately from model inference and path audit
time.

Full result files:

```text
results/result_SEABED_v8_exact_embedding_anchor_YAGO_test_k100_two_swap_20260802_133436.json
results/result_SEABED_v8_exact_embedding_anchor_WIKIDATA_test_k100_two_swap_20260802_133856.json
```

## Control-dataset full results

The formal control runs used `test_k=100`, all 10,000 test pairs, and the V0
LUBM/SWDF checkpoints. Feature-topology reindexing changed zero graphs in both
datasets.

| Dataset | Alignment before | Alignment after | Added aligned entities | Changed/improved pairs | Harmed pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| LUBM | 1,417 / 2,119 (66.8712%) | 1,495 / 2,119 (70.5521%) | 78 | 69 | 0 |
| SWDF | 587 / 884 (66.4027%) | 624 / 884 (70.5882%) | 37 | 32 | 0 |

GED and executable-path results are unchanged from the corresponding V2 full
runs:

| Dataset | Dense MAE / ACC / FEA | Simple MAE / ACC / FEA | Multi MAE / ACC / FEA |
| --- | --- | --- | --- |
| LUBM | 0.102 / 0.908 / 1.000 | 0.102 / 0.908 / 1.000 | 0.102 / 0.908 / 1.000 |
| SWDF | 0.258 / 0.773 / 0.973 | 0.234 / 0.800 / 1.000 | 2.721 / 0.086 / 0.998 |

SWDF column 3 is compatible with the simple-graph view; its multirelation
numbers are a distinct KG-path target and must not be presented as benchmark
GED error. LUBM's two views agree.

Both protected cost-changed pair counts are zero on both datasets. All path
validity, cost consistency, and replay rates are 1.0. The first 100 saved paths
have zero simple/multirelation cost mismatches against V2. LUBM changes two
mappings and improves alignment from 27 to 29; SWDF changes none in that prefix.

Exact-anchor search costs 0.002521 seconds per LUBM pair and 0.002639 seconds
per SWDF pair, approximately 25 and 26 seconds over each full dataset.

Full control result files:

```text
results/result_SEABED_v8_exact_embedding_anchor_LUBM_test_k100_two_swap_20260802_161634.json
results/result_SEABED_v8_exact_embedding_anchor_SWDF_test_k100_two_swap_20260802_163430.json
```

## Final decision

V8 is retained as the final tested framework version. Across all four
datasets, 3,872 graph pairs gain at least one exact identity anchor, 6,937
shared-entity observations are newly aligned, no pair loses shared-entity
alignment, and no simple or multirelation path cost changes.

The contribution must be stated as GED-equivalent correspondence refinement
and executable explanation, not as additional GED prediction accuracy.
