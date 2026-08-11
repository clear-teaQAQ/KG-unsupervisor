# V15: Benchmark-Projected Relation Input

## Non-negotiable objective

V15 does not redefine GED. It keeps the V11 task and hard references:

```text
use_raw_features=1
ged_column=3
cost_mode=unit
preference labels=unchanged V11 candidate unit-GED ordering
primary metrics=MAE and ACC

V11 LUBM: MAE 0.092, ACC 0.916
V11 SWDF: MAE 0.263, ACC 0.756
```

## Research question

SWDF column 3 was generated through an undirected `nx.Graph`, which keeps one
relation for each endpoint pair using last-write-wins semantics. V11 GINE,
however, consumes every raw parallel relation edge. V15 tests whether these
benchmark-ignored messages make SWDF node matching harder.

## Single controlled change

V15 maintains two edge views in every pair:

```text
model edge view     raw V11 edges or undirected last-write projection
unit-GED edge view  exact original V11 edges in both modes
```

The projected model view treats `(u,v)` and `(v,u)` as the same undirected
endpoint, keeps the relation and orientation from the last raw occurrence, and
then adds symmetric GINE directions with that same relation. Generator and
discriminator both receive this view.

The candidate evaluator, BPR labels, pseudo-label update, diffusion, node
features, model architecture, and inference selection remain unchanged. This
isolates input representation from evaluator changes.

## Modes

```text
raw              exact V11 GINE and unit-GED edge views
projected_input  projected GINE view, unchanged V11 unit-GED view
```

## Tests

```bash
cd /data/projects/GEDRanker-main
/home/vermouth/miniconda3/envs/gedranker/bin/python \
  experiments/seabed_versions/v15_benchmark_projected_relation_input/test_v15.py
```

## Acceptance

Run paired raw/projected controls with the same seed. LUBM is a no-drop parity
control. SWDF is the effectiveness test. Do not claim improvement from smoke or
a single unpaired historical comparison.
