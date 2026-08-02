# V9: Benchmark Shortcut Audit

## Purpose

V8 produced near-perfect path GED on YAGO and WIKIDATA. V9 tests whether this
requires graph matching at all.

The primary baseline is the model-free size bound:

```text
b(G1, G2) = abs(|V1| - |V2|) + abs(|E1| - |E2|)
```

It uses no checkpoint, node feature, edge label, topology, or correspondence.

## Run

```bash
/home/vermouth/miniconda3/envs/gedranker/bin/python \
  /data/projects/SEABED-main/SEABED_ablation_runs/03_local_only/dataset/count_size_bound_ged.py \
  --root /data/projects/SEABED-main/data \
  --datasets YAGO WIKIDATA LUBM SWDF \
  --splits test
```

## Interpretation

A dataset where the size bound equals every GED label cannot demonstrate
matching quality or structural GED reasoning. Model results can still test
software invariants and path construction, but they are not evidence that the
model solved a nontrivial correspondence problem.

