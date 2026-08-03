# V10 Results

Status: complete on all four datasets after 200-epoch training and independent
fixed-seed raw evaluation. V10 improves raw correspondence everywhere; SWDF
exposes a measurable GED tradeoff that must be addressed by the next version.

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

### LUBM training dynamics

The full LUBM training run processed 30,000 training pairs per epoch for 200
epochs. KG-aware tie updates remained active throughout training:

| Epoch | Strict GED updates | Equal-GED anchor updates | Best anchors | Available anchors | Best pseudo-label anchor recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15,784 | 166 | 1,533 | 7,617 | 20.13% |
| 10 | 1,471 | 57 | 4,412 | 7,617 | 57.92% |
| 50 | 77 | 8 | 5,081 | 7,617 | 66.71% |
| 100 | 12 | 10 | 5,373 | 7,617 | 70.54% |
| 150 | 5 | 0 | 5,558 | 7,617 | 72.97% |
| 200 | 1 | 1 | 5,619 | 7,617 | 73.77% |

Cumulative updates over 200 epochs:

```text
strict GED updates:       90,589
equal-GED anchor updates:  2,415
```

The persistent semantic updates and monotonic long-term anchor-recall trend
confirm that the secondary objective changes the pseudo-label population. Small
single-epoch decreases are expected when a lower-GED candidate replaces a
higher-anchor mapping, because GED remains lexicographically primary.

### Formal raw comparison

Both rows below independently load their saved checkpoint, reset seed 0, and
run the same unchanged raw best-of-100 evaluator over all 10,000 test pairs.
Neither row uses structural or semantic postprocessing.

| Model | MSE | MAE | ACC | FEA | rho | tau | Selected / available anchors | Raw anchor recall | Perfect-anchor pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 baseline | 0.135 | 0.110 | 0.901 | 1.000 | 0.969 | 0.949 | 1,418 / 2,119 | 66.92% | 1,023 |
| V10 | 0.129 | 0.109 | 0.900 | 1.000 | 0.967 | 0.948 | 1,565 / 2,119 | 73.86% | 1,161 |

Formal delta:

```text
selected exact anchors: +147
raw anchor recall:      +6.94 percentage points
perfect-anchor pairs:   +138
GED MSE:                -0.006
GED MAE:                -0.001
GED ACC:                -0.001
```

V10 therefore produces a clear learned correspondence gain while retaining
essentially the same GED quality. MSE and MAE improve slightly; ACC, rho, and
tau fluctuate down by 0.001-0.002. The supported conclusion is correspondence
improvement without material GED degradation, not uniform improvement of every
GED/ranking metric.

This is qualitatively different from V8: V10 reports the model's unmodified
matching selected by unchanged minimum-GED best-of-k inference. Both result
JSON files record `postprocessing: none`; neither two-swap nor exact-anchor
repair ran.

Formal LUBM files:

```text
checkpoints/LUBM_200_GEDRankerSEABED_v10_kg_tie_aware_col3_unit_BPR_20260802_181515.pt
training_results/training_diagnostics_LUBM_20260802_181515.json
training_results/result_SEABED_LUBM_test_BPR_gedcol3_unit_20260802_181515.json
raw_eval_results/result_SEABED_LUBM_test_BPR_gedcol3_unit_20260802_181430.json
raw_eval_results/result_SEABED_LUBM_test_BPR_gedcol3_unit_20260802_215317.json
```

Checkpoint SHA-256:

```text
edecf3ab0ee9f1e6c00c1f3f6338c31f21d8a61f07cb63f58431181c7faf3458
```

LUBM is accepted.

### Four-dataset raw correspondence

All rows below come from independently loaded checkpoints evaluated with seed
0, unchanged best-of-100 minimum-GED selection, and no postprocessing.

| Dataset | Baseline anchors | V10 anchors | Baseline recall | V10 recall | Delta | Baseline perfect pairs | V10 perfect pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 1,418 / 2,119 | 1,565 / 2,119 | 66.92% | 73.86% | +6.94 pp | 1,023 | 1,161 |
| SWDF | 584 / 884 | 640 / 884 | 66.06% | 72.40% | +6.33 pp | 250 | 307 |
| YAGO | 116,903 / 119,370 | 119,352 / 119,370 | 97.93% | 99.98% | +2.05 pp | 4,437 | 5,986 |
| WIKIDATA | 250,144 / 254,550 | 254,493 / 254,550 | 98.27% | 99.98% | +1.71 pp | 7,778 | 9,970 |

V10 adds 7,001 aligned exact-identity observations across the four test sets.
The aggregate is dominated by YAGO/WIKIDATA, so per-dataset deltas are the
authoritative presentation.

### Four-dataset GED behavior

| Dataset | Baseline MSE | V10 MSE | Baseline MAE | V10 MAE | Baseline ACC | V10 ACC | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LUBM | 0.135 | 0.129 | 0.110 | 0.109 | 0.901 | 0.900 | GED retained |
| SWDF | 0.434 | 0.460 | 0.341 | 0.359 | 0.701 | 0.686 | GED tradeoff |
| YAGO | 0.008 | 0.004 | 0.006 | 0.004 | 0.995 | 0.997 | shortcut-affected |
| WIKIDATA | 0.186 | 0.189 | 0.130 | 0.132 | 0.892 | 0.892 | shortcut-affected, stable ACC |

SWDF is the decisive limitation. Raw anchor recall rises by 6.33 percentage
points, but MAE rises by 0.018 and ACC falls by 0.015. The independent result
agrees with the training-end evaluation (`MAE=0.358`, `ACC=0.687`), so this is
not a checkpoint-selection or evaluation-mode error.

The update rule remains locally GED-primary: a worse-GED candidate can never
replace the current archive because of anchors. However, accepting a different
equal-GED pseudo-label changes the generator and its later rollout trajectory.
On SWDF, the final training best-GED sum is 291,819 versus 291,265 for the
baseline run. Local lexicographic safety therefore does not guarantee identical
finite-training GED convergence.

YAGO and WIKIDATA GED values remain secondary because V9 proved their labels
are completely determined by graph-size differences. Their valid V10 evidence
is the raw correspondence increase to nearly 100%, not their GED delta.

### Training mechanism

| Dataset | Equal-GED anchor updates | Initial pseudo-label anchor recall | Final pseudo-label anchor recall |
| --- | ---: | ---: | ---: |
| LUBM | 2,415 | 20.13% | 73.77% |
| SWDF | 444 | 21.12% | 65.62% |
| YAGO | 9,559 | 11.04% | 100.00% |
| WIKIDATA | 15,840 | 14.68% | 100.00% |

These 28,258 semantic tie updates establish that the raw test gains are linked
to changed training pseudo-labels rather than test-time selection.

### Checkpoints

```text
LUBM
edecf3ab0ee9f1e6c00c1f3f6338c31f21d8a61f07cb63f58431181c7faf3458

SWDF
9f2e90ac02789bffdfe1299163270183dd629b7f33deb2ac8c5bda5d121b291c

YAGO
f8c62e76ee435c5b39563db5564b1445d889ce1f2df9dc34f4db236feac1da07

WIKIDATA
979b87861f425d3431d5177770b2dc5ab1fc6ddcfd3f55820c61e1bb9ef6df8a
```

### Decision

V10 is accepted as proof that KG identity evidence can shape GEDRanker's learned
correspondence distribution without changing the evaluator or using inference
repair. It is not accepted as the final method because SWDF demonstrates that
the single semantic pseudo-label archive can alter structural convergence.

The next version should preserve a separate GED-only structural archive while
using a KG-aware archive or auxiliary loss to teach correspondence. Its success
criterion is retaining the V10 alignment gain while recovering SWDF GED to the
baseline range.
