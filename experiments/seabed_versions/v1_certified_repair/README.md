# V1: Certified Exact Local Repair

## Single change from V0

V1 loads a frozen V0 checkpoint and applies exact hill-climbing repair to the
best matching produced by the existing parallel diffusion sampler.

One local move either:

- swaps the target assignments of two matched source nodes; or
- swaps a matched target node with an unmatched target node.

Every candidate is scored with exactly the same undirected, relation-labeled
`unit` cost as V0. V1 does not change training, node features, the GED label,
direction handling, predicate costs, or node semantic costs.

The current repair cost revision is `deterministic_dense_v4`. V1 retains and
validates every complete `n2 x n2` candidate permutation during parallel
inference. It builds relation-labeled adjacency once per graph pair on CPU with
explicit edge-list-order `last-write-wins` semantics, then shares those exact
matrices between candidate selection and repair. This avoids CUDA's undefined
write order when several relation labels target the same dense adjacency cell.
Every candidate runs the same padded upper-triangle comparison as V0 intends.

The repair stops when no strict improvement exists, when `repair_max_iterations`
is reached, or when the exact lower bound of V0's effective adjacency evaluator
is reached:

```text
|V2 - V1| + |E2 - E1|
```

Reaching the bound certifies that the returned hard matching is globally optimal
under the current evaluator. The bound is never used as a predicted GED value.

V0 stores one relation label per dense adjacency cell. Parallel KG edges sharing
the same endpoint pair can therefore collapse. V1 reports both the raw graph-size
bound and the effective-evaluator bound, but only the latter is used for certified
termination. This limitation is inherited from V0 and is not silently repaired in
V1 because doing so would change a second experimental factor.

## Run

Fast smoke evaluation on 100 YAGO pairs:

```bash
SMOKE=1 DATASETS=YAGO bash experiments/seabed_versions/v1_certified_repair/run.sh
```

Full four-dataset evaluation using the frozen V0 checkpoints:

```bash
bash experiments/seabed_versions/v1_certified_repair/run.sh
```

Useful overrides:

```bash
DATASETS="LUBM SWDF" TEST_K=100 REPAIR_MAX_ITERATIONS=20 \
  bash experiments/seabed_versions/v1_certified_repair/run.sh
```

To verify that the isolated entry point reproduces V0 inference without repair:

```bash
SMOKE=1 DATASETS=YAGO REPAIR_MODE=none \
  bash experiments/seabed_versions/v1_certified_repair/run.sh
```

JSON outputs are written only to this version's `results/` directory. Aggregate
repair diagnostics include initial/final MAE, lower-bound hit rate, improvement
rate, average cost reduction, and average number of repair iterations.

The run script defaults to
`/home/vermouth/miniconda3/envs/gedranker/bin/python`. Override `PYTHON_BIN` if
the environment moves.

## Acceptance rule

Promote the idea to V2 only if V1 gives a meaningful correspondence improvement
without unacceptable inference cost. Otherwise keep the certificate as a cheap
termination/reporting device and discard local repair.
