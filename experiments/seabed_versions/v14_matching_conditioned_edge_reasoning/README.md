# V14: Matching-Conditioned Edge Reasoning

## Non-negotiable objective

V14 does not redefine GED. It fixes the same V11 task:

```text
use_raw_features=1
ged_column=3
cost_mode=unit
preference label=strict ordering by the existing unit-GED evaluator
primary metrics=MAE and ACC
```

The research question is whether a discriminator can rank the same unit-GED
candidates better when it explicitly sees the edge consequences induced by
each candidate matching.

## Difference from V11

V11 uses relation-aware GINE representations before candidate evaluation.
V14 additionally conditions on the candidate mapping itself. For every real
edge on both graph sides, it derives three differentiable events:

```text
exact relation edge preserved
topology preserved but relation label differs
edge broken or extra under the candidate
```

Exact means exact pair-local relation ID equality. No embedding cosine
similarity enters the event definition.

The events are encoded per edge, pooled in both graph directions, and passed
through a small learned scorer:

```text
D_v14(candidate) = D_v11(candidate) + tanh(gate) * edge_reasoner(candidate)
```

`V14_GATE_INIT=0.0`, so the residual is initially disabled. The BPR labels,
generator, diffusion, pseudo-label update, exploration schedule, and evaluator
are inherited unchanged from V11.

## Modes

```text
baseline      exact V11 generator and discriminator
matched_edge  V11 plus the zero-gated edge-reasoning discriminator residual
```

The extra module preserves the V11 RNG stream during construction. Baseline
and matched-edge runs with the same `SEED` therefore start with the same V11
parameters, initial mappings, and data order until the learned residual changes
the optimization trajectory.

## Tests

```bash
cd /data/projects/GEDRanker-main
/home/vermouth/miniconda3/envs/gedranker/bin/python \
  experiments/seabed_versions/v14_matching_conditioned_edge_reasoning/test_v14.py
```

Tests cover exact/wrong/broken events, reconstruction of hard unit edge cost,
soft-matching gradients, and exact zero-gate V11 score parity.

## Smoke

```bash
cd /data/projects/GEDRanker-main
SMOKE=1 EPOCHS=1 DATASETS=LUBM V14_MODE=matched_edge \
  PYTHON_BIN=/home/vermouth/miniconda3/envs/gedranker/bin/python \
  bash experiments/seabed_versions/v14_matching_conditioned_edge_reasoning/train.sh
```

Smoke results validate plumbing only and must not be interpreted as model
performance.

## Acceptance order

1. Run paired LUBM baseline and matched-edge with the same seed.
2. Independently evaluate both checkpoints with the same seed and `test_k=100`.
3. Continue to SWDF only if matched-edge improves MAE or ACC without a material
   regression in the other primary metric.
4. Run at least three paired seeds before making an effectiveness claim.

The V11 independent fixed-seed references remain:

```text
LUBM: MAE 0.092, ACC 0.916
SWDF: MAE 0.263, ACC 0.756
```
