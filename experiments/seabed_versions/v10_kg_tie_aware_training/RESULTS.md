# V10 Results

Status: implementation and functional smoke verified; formal training pending.

## Functional checks

| Check | Result |
| --- | --- |
| Lexicographic objective unit tests | 5 / 5 passed |
| One-epoch isolated smoke | passed on LUBM, 128 training pairs |
| Checkpoint saved | yes |
| Raw evaluation contains no postprocessing | confirmed |

The accepted functional smoke command was:

```bash
SMOKE=1 DATASETS=LUBM MAX_TRAIN_PAIRS=128 MAX_VAL_PAIRS=4 \
  MAX_TEST_PAIRS=5 TEST_K=1 \
  bash experiments/seabed_versions/v10_kg_tie_aware_training/train.sh
```

Training diagnostics:

| Candidates | Strict GED updates | Equal-GED anchor updates | Available train anchors | Best pseudo-label anchors |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 41 | 1 | 18 | 2 |

The nonzero equal-GED update confirms that the KG secondary objective reaches
the persistent pseudo-label and therefore later mapping-loss updates. It does
not establish model quality after one epoch.

The five-pair raw result had zero selected anchors and zero anchor recall. This
is retained rather than repaired: the smoke verifies plumbing only and must not
be presented as a trained result.

An independent raw-evaluator smoke loaded the old LUBM 200-epoch checkpoint via
the same code path. It recorded the checkpoint path, selected mappings through
unchanged first-minimum-GED best-of-k inference, and set `postprocessing` to
`none`.

Smoke files:

```text
checkpoints/LUBM_1_GEDRankerSEABED_v10_kg_tie_aware_col3_unit_BPR_20260802_180235.pt
training_results/training_diagnostics_LUBM_20260802_180235.json
training_results/result_SEABED_LUBM_test_BPR_gedcol3_unit_20260802_180235.json
raw_eval_results/result_SEABED_LUBM_test_BPR_gedcol3_unit_20260802_180256.json
```

## Formal results

No formal 200-epoch V10 result has been produced yet. Do not copy V8 repaired
alignment or V9 shortcut numbers into this section; they are not trained V10
outputs.
