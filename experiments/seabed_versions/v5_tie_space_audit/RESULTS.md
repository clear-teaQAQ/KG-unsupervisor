# V5 Tie-Space Results

Status: complete. Four-dataset 100-path offline audit passed.

This version is diagnostic and does not change model training, GED costs, or
reported executable paths.

## Functional check

The first saved YAGO path starts with 18/20 shared entities aligned. Both the
ID-guided local oracle and embedding-cosine tie-break reach 20/20 while
preserving simple and multirelation costs exactly. No model inference was run.

## Affected-dataset audit

The audit used the 100 paths saved by each V4 full run.

| Dataset | Shared entities | Initial alignment | ID local oracle | Cosine tie-break | Improved pairs | Harmed pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YAGO | 2,050 | 1,974 (96.29%) | 2,050 (100%) | 2,050 (100%) | 42 / 100 | 0 / 100 |
| WIKIDATA | 2,700 | 2,680 (99.26%) | 2,700 (100%) | 2,700 (100%) | 17 / 100 | 0 / 100 |

Equal-cost local neighbors exist for 98/100 YAGO pairs and 84/100 WIKIDATA
pairs. An ID-improving neighbor exists for every pair improved by cosine. On
this sample, embedding cosine exactly matches the ID-guided local oracle.

Protected invariants:

```text
simple cost unchanged rate        = 1.0
multirelation cost unchanged rate = 1.0
```

This is stronger than merely preserving the benchmark-compatible view: the
same implementation cannot degrade either reported path representation. The
sample rates differ from full V4 semantic rates because V5 analyzes the first
100 saved paths rather than all test pairs.

Results:

```text
results/tie_space_YAGO.json
results/tie_space_WIKIDATA.json
```

The cosine candidate is retained. Run the same offline safety audit on LUBM and
SWDF before integrating it into full inference.

## Control-dataset audit

| Dataset | Shared entities | Initial alignment | ID local oracle | Cosine tie-break | Improved pairs | Harmed pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 43 | 27 (62.79%) | 29 (67.44%) | 29 (67.44%) | 2 / 100 | 0 / 100 |
| SWDF | 1 | 0 | 0 | 0 | 0 / 100 | 0 / 100 |

LUBM again shows exact agreement between cosine and the ID-guided local oracle.
SWDF has insufficient shared-entity overlap for a semantic conclusion; its role
is a safety control. Equal-cost neighbors exist for 97/100 LUBM and 94/100 SWDF
pairs, and both protected costs remain unchanged on every analyzed path.

Results:

```text
results/tie_space_LUBM.json
results/tie_space_SWDF.json
```

## Final decision

V5 is accepted and closed. Across 400 saved paths, embedding cosine improves 61
pairs, harms no shared-entity alignment, and exactly matches the ID-guided local
oracle wherever that oracle improves the mapping. Simple and multirelation costs
remain unchanged on every pair.

Proceed to a separate full-inference version. It must load V0 checkpoints for
the no-op LUBM/SWDF datasets and V4 corrected-training checkpoints for
YAGO/WIKIDATA, apply the same dual-cost-preserving cosine repair, emit executable
paths, and report before/after semantic alignment plus runtime. No retraining or
semantic GED cost is justified.
