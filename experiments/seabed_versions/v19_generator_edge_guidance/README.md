# V19: Generator Edge Guidance

## Fixed objective

V19 does not change the task definition:

```text
cost_mode=unit
ged_column=3
ground truth unchanged
BPR preference unchanged
primary metrics=MAE and ACC
official graph=V16 undirected simple last-write view
```

## Controlled model change

V14/V18 supplied matching-conditioned relation evidence only to the
discriminator. V19 supplies the same kind of exact, wrong-relation and missing
edge evidence to the diffusion generator at every noisy matching state. A
small batched network converts five per-candidate features into a residual:

```text
V19 logits = V18 generator logits + tanh(gate) * edge residual
gate starts at zero
```

`baseline` retains the V18 matched-edge discriminator and original generator.
`generator_edge` changes only the generator. Both use identical unit-GED,
preference labels, official graph, optimizer and training schedule.

The edge evidence reuses V18's cached correspondence combinations and uses GPU
gather plus `index_add_`; it does not build a relation-by-node 3-D matrix and
does not loop over graph pairs during a forward pass.

## Smoke test

```bash
cd /data/projects/GEDRanker-main

SMOKE=1 \
DATASETS="LUBM SWDF" \
EPOCHS=1 \
V19_MODE=generator_edge \
PYTHON_BIN=/home/vermouth/miniconda3/envs/gedranker/bin/python \
bash experiments/seabed_versions/v19_generator_edge_guidance/train.sh
```

Run paired seed-0 formal training only after unit tests and smoke timing pass.

