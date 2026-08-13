# V17: Cross-Graph Sinkhorn Matching

## Objective

V17 preserves the fixed task used by V11/V16:

```text
cost_mode=unit
ged_column=3
use_raw_features=1
ground truth unchanged
BPR preference definition unchanged
primary metrics=MAE and ACC
```

The model change is isolated to the mapping generator. V17 replaces the
diffusion denoiser's graph-local mapping logits with a relation-aware GINE
encoder followed by cross-graph attention and a learned pair scorer:

```text
G1/G2 relation-aware encoding
        -> cross-graph query-key compatibility
        -> node-pair assignment logits
        -> existing differentiable Gumbel-Sinkhorn rollout
```

The existing trainer still computes candidate GED with the V16 official graph
view and trains the unchanged BPR ranking critic. No semantic distance is
added to GED and no mapping labels are redefined.

## Validation

The V17 unit test, Python compilation, shell syntax check, and one-epoch LUBM
and SWDF smoke runs pass. Smoke metrics use 20 test pairs and `test_k=3`; they
are plumbing checks, not model results.

## Pilot

Use a 20-epoch pilot on LUBM and SWDF before a 200-epoch run. The pilot should
be compared with V16 under the same seed and configuration.
