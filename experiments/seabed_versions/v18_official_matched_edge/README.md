# V18: Official-Graph Matching-Conditioned Edge Reasoning

## Fixed objective

V18 changes what the discriminator sees, not what the model learns:

```text
cost_mode=unit
ged_column=3
ground truth unchanged
BPR preference=strict ordering by the existing unit-GED evaluator
primary metrics=MAE and ACC
```

## Model

V18 combines the successful parts of V14 and V16:

```text
V16 undirected last-write official graph
+ V11 relation-aware diffusion generator
+ V11 preference discriminator
+ zero-gated matching-conditioned exact-edge residual
```

`baseline` disables the new discriminator. `matched_edge` uses the learned
residual. Both modes use the same official graph, generator, unit-GED cost, BPR
labels, training schedule, and evaluator.

## Runtime policy

The V14 edge reasoner rebuilt adjacency matrices and looped over every pair on
every discriminator call. V18 caches edge-correspondence indices once and
evaluates all pairs in a batch with gather and pooling operations. It also:

- disables the V14 in-training preference audit by default;
- turns candidate-vs-ground-truth checks into counters, never aborts;
- removes V16's per-pair runtime consistency walk from the training path;
- skips the discriminator forward after the original exploration weight is 0.

The last optimization is objective-preserving: epochs in the second half have
`alpha=0`, so GEDRanker already multiplies that discriminator term by zero.

## Smoke

```bash
cd /data/projects/GEDRanker-main
SMOKE=1 DATASETS="LUBM SWDF" EPOCHS=1 V18_MODE=matched_edge \
PYTHON_BIN=/home/vermouth/miniconda3/envs/gedranker/bin/python \
bash experiments/seabed_versions/v18_official_matched_edge/train.sh
```

## Acceptance

Screen on LUBM and SWDF under a paired seed. Continue to seeds 1 and 2 only if
the full method preserves both primary metrics relative to the same-path V18
baseline and materially improves at least one. Independent evaluation must use
all test pairs and `test_k=100`.

