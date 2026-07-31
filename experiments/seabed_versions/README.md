# SEABED Experiment Versions

This directory is the experiment registry for the GEDRanker-to-SEABED work.
Each version has one isolated hypothesis, its own entry point, run script, and
result record. A completed version is not edited when the next version starts.

## Version protocol

1. Change one method factor per version.
2. Reuse the same V0 checkpoints unless the version explicitly changes training.
3. Keep SEABED GED column 3 and the benchmark-faithful `unit` evaluator fixed.
4. Run a smoke evaluation before a full evaluation.
5. Record the exact command, checkpoint, aggregate metrics, and conclusion in the
   version's `RESULTS.md` before creating the next version.

## Registry

| Version | Hypothesis | Training changed | Status |
| --- | --- | --- | --- |
| [V0](v0_clean_baseline/README.md) | Clean GEDRanker baseline | Yes, original training | Complete |
| [V1](v1_certified_repair/README.md) | Exact local repair improves generated correspondence | No | Smoke passed; full pending |

## Historical naming note

`src/SEABED_v1/` predates this registry and is the old no-edge-label variant.
It is not V1 in the experiment sequence above and is intentionally left unchanged.
