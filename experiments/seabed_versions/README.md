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
| [V1](v1_certified_repair/README.md) | Exact local repair improves generated correspondence | No | Complete; retained |
| [V2](v2_edit_path_audit/README.md) | Audit frozen V1 mappings as executable simple and multi-relation paths | No | Complete |
| [V3](v3_topology_feature_reindex/README.md) | Reindex node features to topology entities before semantic work | No | Complete; retained |
| [V4](v4_corrected_training/README.md) | Retrain unchanged GEDRanker on topology-aligned features | Yes, corrected inputs | Complete; retained |
| [V5](v5_tie_space_audit/README.md) | Measure dual-cost-preserving semantic repair space | No | Complete; retained |
| [V6](v6_semantic_tie_break/README.md) | Apply dual-cost-preserving embedding-cosine tie-break | No | Smoke passed; broad cosine not retained |
| [V7](v7_embedding_evidence_audit/README.md) | Determine whether node embeddings carry semantic or identity evidence | No | Complete; identity evidence confirmed |
| [V8](v8_exact_embedding_anchor/README.md) | Recover exact embedding identity anchors inside the dual-cost tie space | No | Complete; retained |
| [V9](v9_benchmark_shortcut_audit/README.md) | Test whether GED labels require matching beyond graph-size differences | No | Complete; YAGO/WIKIDATA shortcut confirmed |
| [V10](v10_kg_tie_aware_training/README.md) | Train the generator on GED-primary, exact-anchor tie-resolved pseudo-labels | Yes, pseudo-label selection | Complete; correspondence improves on all four, SWDF GED tradeoff |
| [V11](v11_relation_aware_ged_training/README.md) | Encode relation embeddings used by column-3 GED in generator and discriminator message passing | Yes, model input/encoder | Smoke verified on LUBM/SWDF; formal training pending |

## Current decision

V10 is the latest completed training version. V1-V9 remain frozen engineering,
explainability, data-integrity, and benchmark audits; their post-hoc improvements
are not treated as learned correspondence quality. V10 keeps the column-3 unit
GED evaluator fixed and changes only which equal-GED pseudo-label the generator
learns during training. Its primary evaluation uses raw generator mappings with
no V1/V8 repair.

V10 raises raw exact-anchor recall from 66.92% to 73.86% on LUBM, 66.06% to
72.40% on SWDF, 97.93% to 99.98% on YAGO, and 98.27% to 99.98% on WIKIDATA.
LUBM retains GED quality, while SWDF MAE changes from 0.341 to 0.359 and ACC
from 0.701 to 0.686. This SWDF tradeoff prevents treating V10 as the final
method and motivates a separate structural archive in the next training version.

V11 is the active GED-improvement experiment. It restores the baseline GED-only
pseudo-label rule and isolates relation-aware GINE message passing. This targets
the labeled-edge mismatch that affects 97.56% of LUBM and 99.49% of SWDF test
pairs without exposing graph-size deltas to the model.

The benchmark interpretation remains constrained by V9. The size-only formula
`abs(delta V) + abs(delta E)` equals
100% of YAGO and WIKIDATA test GED labels, so their near-perfect scores are not
evidence of learned matching. LUBM and SWDF match that formula on only 1.52%
and 3.17% of test pairs and remain the nontrivial GED evaluations.

YAGO/WIKIDATA use V4 checkpoints trained after topology-feature correction;
LUBM/SWDF retain V0 checkpoints because the same correction is a verified
no-op. All four datasets produce executable simple and multirelation paths with
100% mapping validity, cost consistency, and replay success.

The V8 anchor rule still has a valid narrow result: across 36,000 pairs it
newly aligns 6,937 shared-entity observations, harms none, and changes no path
cost. This is identity-preserving explanation refinement, not GED improvement.
See [FINAL_RESULTS.md](FINAL_RESULTS.md) for shortcut-aware tables.

## Historical naming note

`src/SEABED_v1/` predates this registry and is the old no-edge-label variant.
It is not V1 in the experiment sequence above and is intentionally left unchanged.
