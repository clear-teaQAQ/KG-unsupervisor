# GEDRanker-SEABED Shortcut-Aware Consolidated Results

> **Critical benchmark caveat:** a parameter-free size formula exactly predicts
> 100% of YAGO and WIKIDATA GED labels. Their near-perfect model/path scores do
> not demonstrate nontrivial graph matching. LUBM and SWDF are the valid
> model-based GED evaluations in the current four-dataset suite.

## Scope

The results have two distinct surfaces that must not be collapsed:

```text
V8: frozen inference, exact structural repair, executable edit paths, and
    GED-equivalent identity refinement

V10: training-time KG-aware pseudo-label selection, evaluated on raw generator
     mappings without V1/V8 repair

V11: relation-aware GINE generator/discriminator training, evaluated on raw
     minimum-GED best-of-100 mappings without V1/V8 repair
```

V8 remains the executable-path/explanation framework. V10 tests learned
identity correspondence; V11 is the latest raw-GED result and tests
relation-aware graph encoding.

Checkpoint policy:

```text
LUBM/SWDF:     original V0 200-epoch checkpoints; reindex is a verified no-op
YAGO/WIKIDATA: V4 200-epoch checkpoints trained on corrected feature order
```

All formal evaluations use SEABED GED column 3, unit costs, `test_k=100`, and
all test pairs.

## Learned Raw GED

V11 replaces node-only GIN message passing with GINE message passing over the
released relation embeddings in both generator and discriminator. It restores
the baseline strict-GED pseudo-label update and uses no graph-size input,
modified edit cost, or inference repair.

| Dataset | Baseline MSE | V11 MSE | Baseline MAE | V11 MAE | Baseline ACC | V11 ACC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 0.135 | **0.107** | 0.110 | **0.092** | 0.901 | **0.916** |
| SWDF | 0.434 | **0.304** | 0.341 | **0.263** | 0.701 | **0.756** |
| YAGO | 0.008 | **0.005** | 0.006 | **0.004** | 0.995 | **0.996** |
| WIKIDATA | 0.186 | **0.177** | 0.130 | **0.125** | 0.892 | **0.897** |

The LUBM/SWDF improvements are the primary evidence because their GED labels
are not size-determined. YAGO/WIKIDATA remain secondary shortcut-dominated
results. A constant-relation GINE control is still required before attributing
the gain specifically to relation identity rather than the encoder change.

## Learned Raw Correspondence

V10 retains GED as the primary pseudo-label objective and uses exact embedding
identity only to select among equal-GED training mappings. Test-time selection
remains unchanged minimum-GED best-of-100 inference with no postprocessing.

| Dataset | Baseline raw anchors | V10 raw anchors | Baseline recall | V10 recall | Delta | GED MAE baseline -> V10 | GED ACC baseline -> V10 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LUBM | 1,418 / 2,119 | 1,565 / 2,119 | 66.92% | 73.86% | +6.94 pp | 0.110 -> 0.109 | 0.901 -> 0.900 |
| SWDF | 584 / 884 | 640 / 884 | 66.06% | 72.40% | +6.33 pp | 0.341 -> 0.359 | 0.701 -> 0.686 |
| YAGO | 116,903 / 119,370 | 119,352 / 119,370 | 97.93% | 99.98% | +2.05 pp | 0.006 -> 0.004 | 0.995 -> 0.997 |
| WIKIDATA | 250,144 / 254,550 | 254,493 / 254,550 | 98.27% | 99.98% | +1.71 pp | 0.130 -> 0.132 | 0.892 -> 0.892 |

The raw correspondence gain reproduces on all four datasets and adds 7,001
aligned exact-identity observations. SWDF is a real limitation: correspondence
improves while GED MAE rises by 0.018 and ACC falls by 0.015. V10 is therefore
evidence that KG-aware pseudo-labels teach correspondence, but not yet the final
multi-objective training design.

YAGO/WIKIDATA correspondence is nearly saturated because derived graphs retain
identical embeddings for shared entities. Their GED deltas remain unsuitable as
matching evidence because of the size shortcut.

## Nontrivial GED Results

| Dataset | Pairs | Compatible path view | Path MAE | Path ACC | Path FEA | Size-only exact rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| LUBM | 10,000 | Simple = multirelation | 0.102 | 0.908 | 1.000 | 1.52% |
| SWDF | 10,000 | Simple | 0.234 | 0.800 | 1.000 | 3.17% |

These are the current executable-path results. Both use original 200-epoch V0
checkpoints followed by exact structural repair and are distinct from V11's raw
generator results above. They are not untrained systems.

## Shortcut-Dominated Results

The benchmark-compatible path view differs because SEABED column 3 does not
have one consistent edge representation across all four datasets.

| Dataset | Pairs | Compatible path view | Path MAE | Path ACC | Size-only exact rate | Alignment before | Alignment after |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| YAGO | 6,000 | Simple = multirelation | 0.000 | 1.000 | 100.00% | 97.9434% | 99.9899% |
| WIKIDATA | 10,000 | Multirelation | 0.000 | 1.000 | 100.00% | 98.2758% | 99.9961% |

YAGO/WIKIDATA were retrained for 200 epochs in V4 after feature-order repair,
but training is unnecessary to recover their GED labels: the size-only formula
already has 100% exact accuracy. WIKIDATA's rounded multirelation MAE is 0.000;
its mean excess cost is 0.0004 and exact optimal-path rate is 99.97%.

## Anchor Effect

| Dataset | Newly aligned entities | Improved pairs | Harmed pairs | Simple cost changes | Multi cost changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| LUBM | 78 | 69 | 0 | 0 | 0 |
| SWDF | 37 | 32 | 0 | 0 | 0 |
| YAGO | 2,443 | 1,551 | 0 | 0 | 0 |
| WIKIDATA | 4,379 | 2,220 | 0 | 0 | 0 |
| **Total** | **6,937** | **3,872** | **0** | **0** | **0** |

Entity IDs are not used for mapping selection. They are used only to evaluate
shared-entity alignment. V7 established that a copied entity has an exactly
equal embedding in every derived graph, is always the unique strict cosine
top-1 in the audited data, and has no incorrect exact-vector collision.

## Path Guarantees

For all 36,000 formal V8 graph pairs and both path views:

```text
mapping validity rate   = 1.0
cost consistency rate   = 1.0
replay success rate     = 1.0
```

The simple path represents an undirected endpoint graph with one retained
predicate per endpoint. The multirelation path preserves the complete predicate
multiset. Both are executable; only the dataset-compatible view should be used
for benchmark GED comparison.

## Interpretation

V8 does not improve GED over its own pre-anchor structural mapping. It selects
a more identity-consistent correspondence only inside the set of mappings with
exactly equal simple and multirelation path costs. Its defensible contribution
is therefore:

```text
unsupervised GED-driven correspondence optimization
topology-correct KG entity features
dual executable edit-path explanations
identity-preserving GED-equivalent correspondence refinement
```

Do not claim that V8 adds semantic node costs, learns direction/type costs, or
improves GED accuracy through the anchor operation. Those targets are not
supported by SEABED column 3.

Do not use YAGO/WIKIDATA accuracy as evidence for learned GED matching. Their
role is limited to implementation scalability, executable-path invariants, and
identity-correspondence analysis under explicit construction leakage.

V10 adds a separate defensible training result:

```text
GED-primary, KG-aware pseudo-label self-training improves raw identity
correspondence on all four datasets without test-time semantic selection
```

Do not claim that its per-update lexicographic rule guarantees unchanged final
GED. SWDF shows that equal-GED pseudo-label choices alter later rollout and
optimization trajectories. The next training version must retain a separate
GED-only structural archive or equivalent preservation objective.

V11 adds the first learned raw-GED improvement on both nontrivial datasets:

```text
relation-aware GINE training improves raw column-3 GED without graph-size
features, modified edit costs, or test-time repair
```

The claim is currently about the complete `raw relation + GINE` method. It must
not yet be narrowed to “relation identity causes the gain” until the constant
and shuffled relation controls are complete. V11 also does not dominate V10 on
identity correspondence: LUBM exact-anchor recall falls from the 66.92%
baseline to 59.41%, even as MAE improves to 0.092. A later combined method must
jointly preserve V11's GED gain and V10's identity-alignment gain.
