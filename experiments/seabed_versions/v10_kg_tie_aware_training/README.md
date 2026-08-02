# V10: KG Tie-Aware Training

## Research question

Can KG identity evidence improve the correspondence distribution learned by
GEDRanker without changing the SEABED GED definition or repairing mappings at
inference time?

V10 changes training, not evaluation. The benchmark GED remains the primary
pseudo-label objective. Exact cross-graph embedding identity is used only when
two candidate mappings have equal benchmark GED:

```text
candidate replaces best if
  candidate GED < best GED
or
  candidate GED == best GED and candidate exact anchors > best exact anchors
```

The accepted mapping becomes `best_mapping_label`, so later diffusion training
steps imitate the KG-better correspondence. A worse-GED mapping can never win
because of its anchors.

## Scope

Fixed from V4:

```text
model architecture and optimizer
BPR discriminator objective
column-3 unit GED evaluator
topology-feature reindex
200 formal epochs
```

Explicitly excluded from primary V10 evaluation:

```text
V1 two-swap repair
V6 cosine tie-break
V8 exact-anchor repair
edit-path cost replacement
semantic/type/direction edit costs
```

Exact embedding equality is identity evidence on these derived SEABED graphs,
as established by V7. V10 therefore tests identity-aware training; it does not
claim broad open-world semantic alignment.

## Tests

```bash
cd /data/projects/GEDRanker-main
/home/vermouth/miniconda3/envs/gedranker/bin/python \
  experiments/seabed_versions/v10_kg_tie_aware_training/test_kg_objective.py
```

## Smoke

One epoch, 128 training pairs, 100 raw test pairs, and `k=1`:

```bash
cd /data/projects/GEDRanker-main
SMOKE=1 DATASETS=LUBM \
  bash experiments/seabed_versions/v10_kg_tie_aware_training/train.sh
```

The smoke is accepted only when training records both update types, saves a
checkpoint, and writes raw correspondence metrics without postprocessing. A
one-epoch score is not a method result.

## Formal training

Run one dataset at a time so failures and result provenance remain isolated:

```bash
cd /data/projects/GEDRanker-main
DATASETS=LUBM \
  bash experiments/seabed_versions/v10_kg_tie_aware_training/train.sh
```

Repeat with `SWDF`, `YAGO`, and `WIKIDATA`. Each command trains from scratch for
200 epochs and then performs unchanged raw best-of-100 inference.

## Fair raw baseline

Evaluate the old V0/V4 checkpoints through exactly the same raw evaluator:

```bash
cd /data/projects/GEDRanker-main
SOURCE=BASELINE DATASETS=LUBM \
  bash experiments/seabed_versions/v10_kg_tie_aware_training/evaluate.sh
```

Re-evaluate a V10 checkpoint independently:

```bash
SOURCE=V10 DATASETS=LUBM \
  bash experiments/seabed_versions/v10_kg_tie_aware_training/evaluate.sh
```

The raw evaluator selects the first minimum-GED sample returned by the original
best-of-k inference. It does not use anchors to select among test-time ties.

## Decision metrics

Primary comparison:

```text
raw GED MAE / ACC
raw exact-anchor recall
strict GED pseudo-label updates
equal-GED anchor pseudo-label updates
best training pseudo-label anchor recall
```

V10 succeeds as a training mechanism only if raw anchor recall improves under
the same evaluator while GED does not materially degrade. YAGO/WIKIDATA GED ACC
is shortcut-affected, so correspondence is the primary result there; LUBM and
SWDF remain the nontrivial GED controls.
