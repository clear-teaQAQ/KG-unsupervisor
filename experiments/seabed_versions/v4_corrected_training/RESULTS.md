# V4 Results

Status: complete. YAGO and WIKIDATA formal training and full audits finished.

## Fixed configuration

```text
training initialization: from scratch, seed 0
training epochs: 200 for formal results
feature revision: kg_edge_topology_reindex_v1
training objective: unchanged BPR
GED label: column 3
cost: unchanged unit cost
postprocessing: unchanged two-swap + dual executable path audit
```

## Functional smoke

Commands:

```bash
SMOKE=1 DATASETS=YAGO \
  bash experiments/seabed_versions/v4_corrected_training/train.sh

SMOKE=1 DATASETS=YAGO \
  bash experiments/seabed_versions/v4_corrected_training/audit.sh
```

The one-epoch checkpoint trained on 128 pairs, was saved successfully, and was
then discovered and loaded by the independent path-audit script. Its SHA-256 is:

```text
ec09f6b0fd40774d6adf38681deaee2719684d49ddb8de3586e355a5d1bc21d6
```

Pipeline checks:

| Check | Result |
| --- | ---: |
| Graphs changed | 30,000 / 31,000 |
| Edge consistency after | 1.000 |
| Fully consistent graphs after | 31,000 / 31,000 |
| Saved executable paths | 100 |
| Simple mapping/cost/replay | 1.000 / 1.000 / 1.000 |
| Multi mapping/cost/replay | 1.000 / 1.000 / 1.000 |

The one-epoch generator has MAE 37.490 and ACC 0.000. Two-swap reduces its
initial MAE from 37.520 to 16.920, but ACC remains 0.010. These values are not
method results: the smoke uses one epoch and 128 training pairs, whereas V3 uses
a 200-epoch checkpoint trained on the complete split.

Files:

```text
checkpoints/YAGO_1_GEDRankerSEABED_v4_corrected_training_col3_unit_BPR_20260801_192346.pt
training_results/manifest_YAGO_epoch1_20260801_192346.json
audit_results/result_SEABED_v4_corrected_training_YAGO_test_k5_two_swap_20260801_193038.json
```

Functional smoke is accepted. Proceed to formal 200-epoch training.

## Formal results

### YAGO generator

The corrected-input model was trained from scratch for 200 epochs on the full
YAGO training split and evaluated on all 6,000 test pairs with `k=100`.

| Version | Training input | Test input | MAE | ACC | FEA | rho | tau | pk10 | pk20 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 | misindexed | misindexed | 0.837 | 0.777 | 1.000 | 0.933 | 0.879 | 0.927 | 0.938 |
| V3 | misindexed | corrected | 0.282 | 0.922 | 1.000 | 0.975 | 0.954 | 0.979 | 0.973 |
| V4 generator | corrected | corrected | 0.003 | 0.998 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 |

The 99.8% value is exact-GED accuracy from the generator evaluation, not FEA or
path replay. The independent deterministic two-swap and dual-path audit below
loads the saved 200-epoch checkpoint and confirms the result as executable paths.

This result is plausible rather than automatically evidence of leakage. YAGO
derived graphs preserve entity IDs and embeddings from their base graph while
adding edits. Correct reindexing restores a strong within-pair identity signal
that the old files attached to the wrong topology nodes. V4 uses only each
graph's features during model inference and never supplies GED labels or an
explicit cross-graph ID mapping. Nevertheless, the benchmark may become nearly
solved by entity identity, so later reporting must include a constant/shuffled
feature control and explicit correspondence metrics rather than presenting the
GED number alone.

Checkpoint:

```text
checkpoints/YAGO_200_GEDRankerSEABED_v4_corrected_training_col3_unit_BPR_20260801_193451.pt
sha256: 5a73e93c2835d4d03fbf7a4fa60101f9967350718f7d4d944b26b22d25abf133
```

Generator result and manifest:

```text
training_results/result_SEABED_YAGO_test_BPR_gedcol3_unit_20260801_193451.json
training_results/manifest_YAGO_epoch200_20260801_193451.json
```

### YAGO executable-path audit

The independent audit loaded the saved 200-epoch checkpoint and evaluated all
6,000 test pairs with `k=100` and unchanged two-swap repair.

| Stage/view | MAE | ACC | FEA | Mapping valid | Cost consistent | Replay success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial deterministic dense | 0.003 | 0.998 | 0.999 | - | - | - |
| Final dense after two-swap | 0.001 | 0.999 | 0.999 | - | - | - |
| Simple executable path | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Multirelation executable path | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Both path representations reach the supplied GED on all 6,000 pairs. Their
rank metrics and relation-preservation rates are also `1.0`. The remaining
dense below-label cases are representation artifacts resolved by the executable
path evaluator, not infeasible edit paths.

Shared-entity alignment is:

```text
116,915 / 119,370 = 97.94%
```

This distinction is important: GED/path accuracy is 100%, but semantic entity
correspondence is not. Equal-cost structural mappings remain, providing a
well-isolated motivation for a later semantic tie-break that must preserve the
benchmark GED optimum.

Audit result:

```text
audit_results/result_SEABED_v4_corrected_training_YAGO_test_k100_two_swap_20260801_225608.json
```

The run saved 100 paths and validated mapping, cost, and replay invariants on
all 6,000 pairs.

### WIKIDATA generator

The corrected-input model was trained from scratch for 200 epochs on the full
WIKIDATA training split and evaluated on all 10,000 test pairs with `k=100`.

| Version | Training input | Test input | MAE | ACC | FEA | rho | tau | pk10 | pk20 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 | misindexed | misindexed | 0.972 | 0.677 | 0.967 | 0.939 | 0.877 | 0.909 | 0.927 |
| V3 | misindexed | corrected | 0.181 | 0.910 | 0.952 | 0.986 | 0.968 | 0.976 | 0.983 |
| V4 generator | corrected | corrected | 0.118 | 0.899 | 0.946 | 0.995 | 0.985 | 0.976 | 0.987 |

The baseline generator uses the legacy dense evaluator. The independent audit
below is the authoritative mapping/path result and substantially improves its
deterministic initial cost before interpreting the path.

Checkpoint:

```text
checkpoints/WIKIDATA_200_GEDRankerSEABED_v4_corrected_training_col3_unit_BPR_20260801_231901.pt
sha256: 2647c1c7c5d5d4cb937890623f7d6aa04f8f69211d0e332dcbdc812467c7cdd9
```

### WIKIDATA executable-path audit

| Stage/view | MAE | ACC | FEA | Mapping valid | Cost consistent | Replay success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial deterministic dense | 0.056 | 0.948 | 0.949 | - | - | - |
| Final dense after two-swap | 0.056 | 0.948 | 0.948 | - | - | - |
| Simple executable path | 0.033 | 0.969 | 0.969 | 1.000 | 1.000 | 1.000 |
| Multirelation executable path | 0.0004 | 0.9997 | 1.000 | 1.000 | 1.000 | 1.000 |

The multirelation representation remains the benchmark-compatible WIKIDATA
view. Only 0.03% of pairs have positive excess, mean excess is `0.0004`, and
maximum excess is `2`. The simple projection is below the label on 3.09% of
pairs, so it must not be used as WIKIDATA feasibility evidence.

Shared-entity alignment is:

```text
250,161 / 254,550 = 98.28%
```

Audit result:

```text
audit_results/result_SEABED_v4_corrected_training_WIKIDATA_test_k100_two_swap_20260802_110404.json
```

The run saved 100 paths and validated all executable-path invariants on all
10,000 pairs.

## Final decision

V4 is accepted and closed. Corrected training raises YAGO to 100% executable
path accuracy and WIKIDATA multirelation optimal-path rate to 99.97%, while
shared-entity alignment remains 97.94%/98.28%. The remaining research target is
therefore correspondence quality among structurally equal-cost mappings, not a
new benchmark GED cost.

The next version must first measure the equal-cost semantic repair space. A
dataset-independent candidate is accepted only when it preserves both simple
and multirelation path costs, then uses entity identity or embedding similarity
as a secondary ranking signal. Exact identity is an oracle/upper-bound analysis;
embedding similarity is the deployable candidate. No semantic score may be
added to the reported GED.
