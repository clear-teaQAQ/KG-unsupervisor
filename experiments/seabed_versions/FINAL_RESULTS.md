# GEDRanker-SEABED Shortcut-Aware Results

> **Critical benchmark caveat:** a parameter-free size formula exactly predicts
> 100% of YAGO and WIKIDATA GED labels. Their near-perfect model/path scores do
> not demonstrate nontrivial graph matching. LUBM and SWDF are the valid
> model-based GED evaluations in the current four-dataset suite.

## Scope

The final tested framework is V8. It combines corrected topology-feature
alignment, frozen GEDRanker inference, exact structural two-swap repair, dual
executable edit paths, and dual-cost-preserving exact embedding anchors.

Checkpoint policy:

```text
LUBM/SWDF:     original V0 200-epoch checkpoints; reindex is a verified no-op
YAGO/WIKIDATA: V4 200-epoch checkpoints trained on corrected feature order
```

All formal evaluations use SEABED GED column 3, unit costs, `test_k=100`, and
all test pairs.

## Nontrivial GED Results

| Dataset | Pairs | Compatible path view | Path MAE | Path ACC | Path FEA | Size-only exact rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| LUBM | 10,000 | Simple = multirelation | 0.102 | 0.908 | 1.000 | 1.52% |
| SWDF | 10,000 | Simple | 0.234 | 0.800 | 1.000 | 3.17% |

These are the current results that can support a GED matching claim. Both use
original 200-epoch V0 checkpoints followed by exact structural repair. They are
not untrained systems.

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
