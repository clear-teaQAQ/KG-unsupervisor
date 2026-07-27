#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
COST_MODE="${COST_MODE:-unit}"
GED_COLUMN="${GED_COLUMN:-3}"
USE_RAW_FEATURES="${USE_RAW_FEATURES:-1}"
DATASETS="${DATASETS:-LUBM SWDF YAGO WIKIDATA}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS="${EPOCHS:-1}"
  TEST_K="${TEST_K:-5}"
  MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-128}"
  MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-64}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-100}"
else
  EPOCHS="${EPOCHS:-200}"
  TEST_K="${TEST_K:-100}"
  MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-0}"
  MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-0}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-0}"
fi

for DATASET in ${DATASETS}; do
  python3 src/SEABED/main.py \
    --dataset "${DATASET}" \
    --dataset-root "${DATA_ROOT}/${DATASET}" \
    --model-train 1 \
    --model-epoch-start 0 \
    --model-epoch-end "${EPOCHS}" \
    --test-k "${TEST_K}" \
    --cost-mode "${COST_MODE}" \
    --ged-column "${GED_COLUMN}" \
    --use-raw-features "${USE_RAW_FEATURES}" \
    --max-train-pairs "${MAX_TRAIN_PAIRS}" \
    --max-val-pairs "${MAX_VAL_PAIRS}" \
    --max-test-pairs "${MAX_TEST_PAIRS}" \
    --model-name "GEDRankerSEABED_main_col${GED_COLUMN}_${COST_MODE}"
done
