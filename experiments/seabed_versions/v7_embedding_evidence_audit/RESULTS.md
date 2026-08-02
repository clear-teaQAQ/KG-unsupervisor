# V7 Results

Status: complete. Four-dataset offline audit passed and resolved the V6
interpretation question.

## Results

| Dataset | Pairs | Unique sources | Shared observations | Unique shared IDs | Exact vectors | Strict cosine top-1 | Incorrect exact collisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 100 | 69 | 43 | 3 | 43/43 | 43/43 | 0 |
| SWDF | 100 | 55 | 1 | 1 | 1/1 | 1/1 | 0 |
| YAGO | 100 | 4 | 2,050 | 79 | 2,050/2,050 | 2,050/2,050 | 0 |
| WIKIDATA | 100 | 2 | 2,700 | 54 | 2,700/2,700 | 2,700/2,700 | 0 |

Every observed shared entity has a raw embedding that is exactly equal in the
source and target. It is also the unique cosine top-1 and reciprocal top-1 in
every case. Correct-entity cosine is exactly 1.0 throughout.

Separation from the best incorrect target remains positive:

| Dataset | Minimum margin | P05 margin | Median margin |
| --- | ---: | ---: | ---: |
| LUBM | 0.516025 | 0.520289 | 0.520289 |
| SWDF | 0.746460 | 0.746460 | 0.746460 |
| YAGO | 0.005079 | 0.021637 | 0.172058 |
| WIKIDATA | 0.059249 | 0.064271 | 0.271247 |

LUBM and SWDF remain dataset-specific semantic evidence gaps because they have
fewer than 100 shared observations. In particular, SWDF has 688 nonshared
source observations and only one shared observation.

## Data-generation evidence

This behavior is guaranteed by SEABED construction rather than being an
accident of the smoke prefix. `dataset/generate_newgraph.py` deep-copies the
source `node_features` into each derived graph, then assigns every node from a
global `statistics[id]["embedding"]` lookup. A retained entity therefore keeps
the same serialized vector in every derived graph.

## Decision

The evidence supports an identity-preserving interpretation, not a broad
semantic-similarity claim. All V6 improvements that can be checked against
entity IDs are exact identity recovery. V6 also changes many LUBM/SWDF mappings
for nonshared entities, but those changes have no available correspondence
ground truth and cannot be claimed as better explanations.

Do not retain unrestricted cosine maximization as the final method. The next
isolated version should accept a dual-cost-equivalent move only when it
increases exact cross-graph embedding anchors. This keeps one universal rule,
recovers copied KG entities, and leaves unsupported nonshared mappings alone.

Result files:

```text
results/embedding_evidence_LUBM.json
results/embedding_evidence_SWDF.json
results/embedding_evidence_YAGO.json
results/embedding_evidence_WIKIDATA.json
```
