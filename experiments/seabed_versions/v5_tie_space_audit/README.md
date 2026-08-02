# V5: Semantic Tie-Space Audit

## Purpose

V4 reaches 100% executable-path accuracy on YAGO and 99.97% multirelation
optimal paths on WIKIDATA, while shared-entity alignment remains 97.94% and
98.28%. V5 first measures whether the remaining correspondence errors can be
repaired without changing GED.

This version is an offline diagnostic over the 100 paths already saved by each
full audit. It does not load a checkpoint or run model inference.

For every mapping, local candidates include:

```text
swap two matched target assignments
replace one assignment with an unmatched target
```

A candidate is eligible only when both are preserved exactly:

```text
simple_graph_path.total_cost
multirelation_path.total_cost
```

Two independent secondary objectives are evaluated:

1. Exact entity-ID alignment: a local oracle and diagnostic upper signal.
2. Node-embedding cosine sum: the deployable semantic tie-break candidate.

Neither objective is added to GED.

## Run

Unit tests:

```bash
cd /data/projects/GEDRanker-main/experiments/seabed_versions/v5_tie_space_audit
/home/vermouth/miniconda3/envs/gedranker/bin/python test_semantic_tie.py
```

Analyze the 100 saved full paths for all four datasets:

```bash
cd /data/projects/GEDRanker-main
bash experiments/seabed_versions/v5_tie_space_audit/run.sh
```

Start with YAGO and WIKIDATA only:

```bash
DATASETS="YAGO WIKIDATA" \
  bash experiments/seabed_versions/v5_tie_space_audit/run.sh
```

## Decision rule

Do not integrate a semantic tie-break into full inference unless:

```text
simple cost unchanged rate       = 1.0
multirelation cost unchanged rate = 1.0
embedding tie-break harms no pair
embedding tie-break improves shared-entity alignment
```

If the ID-guided local oracle improves alignment but cosine does not, embedding
similarity is not a justified implementation despite the available tie space.
