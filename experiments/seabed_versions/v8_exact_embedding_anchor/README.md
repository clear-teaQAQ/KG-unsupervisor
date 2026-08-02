# V8: Exact Embedding Anchor Repair

## Single change from V6

V8 replaces unrestricted cosine maximization with an exact-embedding anchor
objective. A local mapping move is accepted only when both conditions hold:

```text
simple and multirelation executable-path costs remain unchanged
the number of mapped node pairs with exactly equal embeddings increases
```

This decision follows V7: every observed copied entity has an exactly equal
embedding, is the unique strict cosine top-1, and has no incorrect exact-vector
collision. Entity IDs remain diagnostic only and are never used for mapping
selection.

The rule is universal. It has no dataset branch or threshold. Graph pairs with
no recoverable exact anchor retain the structural mapping.

Checkpoint policy remains unchanged:

```text
LUBM/SWDF:     V0 checkpoints
YAGO/WIKIDATA: V4 corrected-training checkpoints
```

No model is trained and no node semantic value is added to GED.

## Smoke

Run affected datasets first:

```bash
cd /data/projects/GEDRanker-main
SMOKE=1 DATASETS="YAGO WIKIDATA" \
  bash experiments/seabed_versions/v8_exact_embedding_anchor/run.sh
```

Then run controls:

```bash
SMOKE=1 DATASETS="LUBM SWDF" \
  bash experiments/seabed_versions/v8_exact_embedding_anchor/run.sh
```

## Acceptance

```text
both cost changed-pair counts = 0
all path validity/consistency/replay rates = 1
anchors_after >= anchors_before
alignment_after >= alignment_before
mapping changes only on anchor-improved pairs
SWDF mapping changes = 0 unless an exact anchor can be added
```

