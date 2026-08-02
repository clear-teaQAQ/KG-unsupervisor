# V4: Corrected-Feature Training

## Single change from V3

V4 trains the unchanged V0 GEDRanker generator from scratch after applying the
accepted V3 topology-feature reindex to every train, validation, and test graph.

Fixed factors:

```text
GEDRanker architecture and optimizer     unchanged
BPR objective and unit GED cost          unchanged
raw embedding dimension                  unchanged
GED column                               3
formal training epochs                   200
V1 two-swap repair                       unchanged
V2 simple/multirelation paths            unchanged
```

No semantic edit cost or tie-break is added. V4 isolates train/test consistency
after the data-integrity repair.

Only YAGO and WIKIDATA require new training because V3 is an exact no-op on
LUBM and SWDF.

## Functional smoke

Train one epoch on 128 YAGO pairs and evaluate 100 pairs with `k=5`:

```bash
cd /data/projects/GEDRanker-main
SMOKE=1 DATASETS=YAGO \
  bash experiments/seabed_versions/v4_corrected_training/train.sh
```

Then load the saved one-epoch checkpoint into the unchanged repair/path audit:

```bash
SMOKE=1 DATASETS=YAGO \
  bash experiments/seabed_versions/v4_corrected_training/audit.sh
```

The one-epoch numbers are not method results. Acceptance requires a checkpoint,
a training result and manifest, complete feature-topology consistency, and all
six simple/multirelation path invariants equal to `1.0`.

## Formal run

After smoke acceptance, train corrected YAGO and WIKIDATA for 200 epochs:

```bash
DATASETS="YAGO WIKIDATA" \
  bash experiments/seabed_versions/v4_corrected_training/train.sh
```

Audit the final checkpoints:

```bash
DATASETS="YAGO WIKIDATA" \
  bash experiments/seabed_versions/v4_corrected_training/audit.sh
```

Training checkpoints, generator results, and path-audit results are isolated in
this directory. The scripts automatically select the newest checkpoint matching
the requested dataset, epoch, and V4 model name.
