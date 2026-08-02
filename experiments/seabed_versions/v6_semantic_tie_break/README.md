# V6: Dual-Cost Semantic Tie-Break

## Single change from V4

V6 adds an inference-only local semantic repair after the unchanged structural
two-swap. It maximizes the sum of cross-graph node-embedding cosine similarities
using matched 2-swaps and unmatched-target replacements.

A move is accepted only if it preserves both executable-path costs exactly:

```text
simple cost after == simple cost before
multirelation cost after == multirelation cost before
```

Entity IDs are never used to select a mapping. They are used after selection to
measure shared-entity alignment.

Checkpoint policy:

```text
LUBM/SWDF:     retained V0 checkpoints (feature reindex is a no-op)
YAGO/WIKIDATA: V4 corrected-training checkpoints
```

No model is trained and no semantic value is added to GED.

## Smoke

Run 100 pairs with `k=5` on all four datasets:

```bash
cd /data/projects/GEDRanker-main
SMOKE=1 bash experiments/seabed_versions/v6_semantic_tie_break/run.sh
```

Run affected datasets first:

```bash
SMOKE=1 DATASETS="YAGO WIKIDATA" \
  bash experiments/seabed_versions/v6_semantic_tie_break/run.sh
```

## Acceptance

Both path representations must retain mapping validity, cost consistency,
replay success, and exactly the V4/V2 per-pair costs. Shared-entity alignment
must not decrease. The result records semantic mapping changes, improved/harmed
pairs, candidate counts, iterations, cosine gain, and added runtime.

Do not start a full run until all four smoke datasets pass.
