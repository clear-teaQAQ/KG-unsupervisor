# V7: Embedding Evidence Audit

## Purpose

V6 safely changes mappings inside a dual-cost-equivalent solution set, but its
100-pair control smoke changes many mappings that cannot be judged with shared
entity IDs. V7 determines what evidence the embedding cosine actually carries.

This version is an offline diagnostic. It does not load a checkpoint, run model
inference, change GED, or select a new mapping.

For every shared entity in the 400 saved V6 smoke paths, it measures:

```text
raw embedding exact equality
cosine top-1, unique top-1, and reciprocal top-1 retrieval
rank and reciprocal rank of the correct entity
margin over the most similar incorrect entity
incorrect exact-vector collisions
```

It also reports the best target cosine for source entities that do not occur in
the target graph. Results explicitly include source-graph coverage because the
V6 smoke uses a non-random test-file prefix.

## Run

Unit tests:

```bash
cd /data/projects/GEDRanker-main/experiments/seabed_versions/v7_embedding_evidence_audit
/home/vermouth/miniconda3/envs/gedranker/bin/python test_embedding_evidence.py
```

Audit all four V6 smoke path files:

```bash
cd /data/projects/GEDRanker-main
bash experiments/seabed_versions/v7_embedding_evidence_audit/run.sh
```

## Decision Rule

Interpret the evidence before full V6 inference:

```text
exact-vector and strict top-1 near 100%:
  embeddings behave as entity identity; frame V6 as identity preservation

top-1 high, exact-vector low, positive separation margin:
  embeddings provide semantic-neighbor evidence beyond exact identity

top-1 weak or incorrect collisions common:
  reject or confidence-gate the V6 cosine tie-break

fewer than 100 shared observations:
  dataset-specific semantic conclusions are inconclusive
```

