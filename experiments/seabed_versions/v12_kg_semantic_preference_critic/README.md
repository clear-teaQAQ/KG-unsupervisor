# V12: KG Semantic Preference Critic

## Hypothesis

Can GEDRanker preserve its current SEABED gains while moving from a purely
relation-aware encoder to an explicit KG semantic preference critic?

This version is intentionally isolated from V10/V11:

- its own entry points
- its own parser
- its own control modes
- its own checkpoints and result directories

## Mode map

- `full`: semantic critic + keep loss + adaptive explore + sparse diagnostics
- `raw`: raw relation mode with full control stack
- `constant`: constant relation control
- `shuffled`: relation-topology shuffle control
- `no_critic`: critic disabled
- `no_keep`: keep-loss disabled
- `no_adapt`: adaptive exploration disabled
- `no_sparse`: sparse diagnostics disabled

## Current status

The isolated trainer, semantic critic, relation controls, sparse candidate
selection, checkpoint format, and raw evaluator are implemented. A one-epoch
LUBM smoke run has completed successfully. The result is not a method claim:
run the attribution controls below before starting the expensive full run.

## Reproducible execution

Run from the repository root. The default scripts use the `gedranker` conda
environment and the external dataset root `/data/projects/SEABED-main/data`.
Set `PYTHON_BIN` and `DATA_ROOT` when those paths differ.

First verify the complete pipeline on a small sample:

```bash
SMOKE=1 DATASETS=LUBM \
  bash experiments/seabed_versions/v12_kg_semantic_preference_critic/train.sh
```

Then run the four-run attribution matrix (full critic, critic ablation,
constant relations, and shuffled relations):

```bash
SMOKE=1 DATASETS="LUBM SWDF" \
  bash experiments/seabed_versions/v12_kg_semantic_preference_critic/run_controls.sh
```

For the formal experiment, use the same matrix with `SMOKE=0`. Each full run
trains for 200 epochs, evaluates all test pairs with `test_k=100`, and writes
its checkpoint and manifest under this directory:

```bash
SMOKE=0 DATASETS="LUBM SWDF" \
  bash experiments/seabed_versions/v12_kg_semantic_preference_critic/run_controls.sh
```

To evaluate a saved checkpoint explicitly:

```bash
CHECKPOINT_PATH=/absolute/path/to/checkpoint.pt \
  DATASETS=LUBM SMOKE=0 \
  bash experiments/seabed_versions/v12_kg_semantic_preference_critic/evaluate.sh
```

The control matrix should be completed before updating `RESULTS.md` or making
causal claims about relation semantics.
