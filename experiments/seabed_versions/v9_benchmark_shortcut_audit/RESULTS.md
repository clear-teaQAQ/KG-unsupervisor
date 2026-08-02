# V9 Benchmark Shortcut Results

Status: complete. The YAGO/WIKIDATA GED task is fully determined by graph size.

## Size-only baseline

| Dataset | Test pairs | GED equals `abs(delta V) + abs(delta E)` | Ratio |
| --- | ---: | ---: | ---: |
| YAGO | 6,000 | 6,000 | 100.00% |
| WIKIDATA | 10,000 | 10,000 | 100.00% |
| LUBM | 10,000 | 152 | 1.52% |
| SWDF | 10,000 | 317 | 3.17% |

The baseline has no learned parameters and does not inspect correspondence,
topology, node embeddings, or relation labels.

## Independent supporting evidence

SEABED's existing YAGO constant-feature ablation sets all node and edge
embeddings to zero yet reports exact-GED accuracy between 98.45% and 99.58%
across its recorded runs. Its existing `08_count_mlp` formula result reports
`MAE=0`, `ACC=1`, and no checkpoint.

The data generator explains this result. It deep-copies each source graph,
applies monotone node/edge additions, and records the number of generated edits
as GED. Retained entity embeddings are copied or reloaded from one global
entity-ID table.

## Decision

YAGO and WIKIDATA cannot support a claim that GEDRanker, structural repair, or
V8 learned nontrivial GED matching on this benchmark. Their near-perfect path
GED is expected from data construction. They may be retained only as:

```text
pipeline and scalability checks
executable-path validity checks
identity-correspondence case studies with an explicit leakage caveat
```

LUBM and SWDF do not have the same size shortcut and remain the valid datasets
for model-based GED comparison. V8's anchor operation still does not improve
GED on any dataset; it changes only equal-cost correspondence.

