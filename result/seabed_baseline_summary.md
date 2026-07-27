# GEDRanker on SEABED: Existing Results

This note records the existing GEDRanker-SEABED result files found in this
workspace. Use the "SEABED main GED" block as the preliminary baseline for the
SEABED task. The "strict/no-edge-label" block is an ablation because LUBM and
SWDF use column 4 of `*_GEDINFO.json`, not the SEABED main GED column.

## SEABED Main GED

These runs use column 3 of `pairs_info`, which is the SEABED main GED label.

| Dataset | Result file | MSE | MAE | ACC | FEA | rho | tau | pk10 | pk20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | `result_SEABED_LUBM_test_BPR_20260629_223750.json` | 0.146 | 0.120 | 0.892 | 1.000 | 0.964 | 0.942 | 0.958 | 0.966 |
| SWDF | `result_SEABED_SWDF_test_BPR.json` | n/a | 0.348 | 0.695 | 0.936 | 0.908 | 0.860 | 0.942 | 0.884 |
| YAGO | `result_SEABED_YAGO_test_BPR.json` | n/a | 1.192 | 0.709 | 1.000 | 0.905 | 0.837 | 0.934 | 0.939 |
| WIKIDATA | `result_SEABED_WIKIDATA_test_BPR_20260629_235213.json` | 8.741 | 1.349 | 0.605 | 0.973 | 0.919 | 0.841 | 0.878 | 0.903 |

Do not use `result_SEABED_WIKIDATA_test_BPR.json` as the main result; it is an
older outlier with MAE 44.943.

## Clean Rerun Results

These are the clean reruns with explicit config recording:

- `--ged-column 3`
- `--cost-mode unit`
- `--use-raw-features 1`
- `--test-k 100`
- full test split (`max_test_pairs=0`)

| Dataset | Result file | MSE | MAE | ACC | FEA | rho | tau | pk10 | pk20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | `result_SEABED_LUBM_test_BPR_gedcol3_unit_20260726_215721.json` | 0.129 | 0.109 | 0.901 | 1.000 | 0.967 | 0.946 | 0.959 | 0.966 |
| SWDF | `result_SEABED_SWDF_test_BPR_gedcol3_unit_20260727_005926.json` | 1.345 | 0.338 | 0.702 | 0.936 | 0.912 | 0.864 | 0.879 | 0.932 |
| YAGO | `result_SEABED_YAGO_test_BPR_gedcol3_unit_20260727_040001.json` | 2.749 | 0.837 | 0.777 | 1.000 | 0.933 | 0.879 | 0.927 | 0.938 |
| WIKIDATA | `result_SEABED_WIKIDATA_test_BPR_gedcol3_unit_20260727_062631.json` | 9.175 | 0.972 | 0.677 | 0.967 | 0.939 | 0.877 | 0.909 | 0.927 |

## Strict / No-Edge-Label Ablation

These runs came from `src/GEDRanker_SEABED`. For LUBM and SWDF, that loader uses
column 4 of `pairs_info`, which SEABED's own strict conversion notes call the
no-edge-label GED column.

| Dataset | Result file | MAE | ACC | FEA | rho | tau | pk10 | pk20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | `result_GEDRanker_LUBM_test_BPR_20260702_015842.json` | 0.203 | 0.901 | 1.000 | 0.956 | 0.926 | 0.955 | 0.975 |
| SWDF | `result_GEDRanker_SWDF_test_BPR_20260701_144221.json` | 2.328 | 0.122 | 1.000 | 0.837 | 0.740 | 0.892 | 0.907 |

## Clean Rerun Protocol

Use `scripts/run_seabed_clean_baseline.sh` to rerun all four datasets under the
same protocol:

- `src/SEABED/main.py`
- `--ged-column 3`
- `--cost-mode unit`
- `--use-raw-features 1`
- `--test-k 100`
- `--model-epoch-end 200`

New result files include `gedcol{N}` and `cost_mode` in their filenames, and the
result JSON includes a `config` block.

For quick smoke/debug runs, use:

```bash
SMOKE=1 bash scripts/run_seabed_clean_baseline.sh
```

Smoke mode defaults to 1 epoch, `test-k=5`, 128 training pairs, 64 validation
pairs, and 100 testing pairs per dataset. You can also limit datasets:

```bash
SMOKE=1 DATASETS=LUBM bash scripts/run_seabed_clean_baseline.sh
```

## Result Collection

Collect result JSON files into a Markdown table:

```bash
python3 scripts/collect_seabed_results.py --result-dir result
```

Optionally write CSV:

```bash
python3 scripts/collect_seabed_results.py --result-dir result --output result/seabed_results.csv
```

## Cost Audit

Run a quick no-training cost audit on a small test subset:

```bash
python3 scripts/audit_seabed_costs.py --dataset LUBM --max-pairs 200
```

Run all four datasets with the default lightweight audit:

```bash
python3 scripts/audit_seabed_costs.py --dataset all --max-pairs 200
```

The audit reports simple non-model baselines:

- `size_delta`: `|V1 - V2| + |E1 - E2|`
- `random_unit`: current relation-labeled unit cost under random matching
- `entity_id_unit`: current relation-labeled unit cost under same-entity-ID matching
- `feature_greedy_unit`: current relation-labeled unit cost under greedy node-embedding matching

The full test-split audit is saved at `result/seabed_cost_audit_full.md`.

| Dataset | Method | Pairs | MAE | ACC | FEA |
| --- | --- | ---: | ---: | ---: | ---: |
| LUBM | size_delta | 10000 | 4.5729 | 0.0152 | 0.0152 |
| LUBM | random_unit | 10000 | 4.9551 | 0.0041 | 1.0000 |
| LUBM | entity_id_unit | 10000 | 3.6328 | 0.0323 | 1.0000 |
| LUBM | feature_greedy_unit | 10000 | 4.1557 | 0.0227 | 1.0000 |
| SWDF | size_delta | 10000 | 4.7799 | 0.0317 | 0.0610 |
| SWDF | random_unit | 10000 | 4.2563 | 0.0052 | 1.0000 |
| SWDF | entity_id_unit | 10000 | 2.9668 | 0.0468 | 1.0000 |
| SWDF | feature_greedy_unit | 10000 | 3.6474 | 0.0259 | 1.0000 |
| YAGO | size_delta | 6000 | 0.0000 | 1.0000 | 1.0000 |
| YAGO | random_unit | 6000 | 39.6858 | 0.0000 | 1.0000 |
| YAGO | entity_id_unit | 6000 | 39.3618 | 0.0000 | 1.0000 |
| YAGO | feature_greedy_unit | 6000 | 39.3618 | 0.0000 | 1.0000 |
| WIKIDATA | size_delta | 10000 | 0.0000 | 1.0000 | 1.0000 |
| WIKIDATA | random_unit | 10000 | 53.1767 | 0.0003 | 1.0000 |
| WIKIDATA | entity_id_unit | 10000 | 52.8581 | 0.0001 | 1.0000 |
| WIKIDATA | feature_greedy_unit | 10000 | 52.8581 | 0.0001 | 1.0000 |
