#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASETS="${DATASETS:-LUBM SWDF YAGO WIKIDATA}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
MODEL_NAME="${MODEL_NAME:-GEDRankerSEABED_v10_kg_tie_aware_col3_unit}"
MODEL_PATH="experiments/seabed_versions/v10_kg_tie_aware_training/checkpoints"
RESULT_PATH="experiments/seabed_versions/v10_kg_tie_aware_training/training_results"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS="${EPOCHS:-1}"
  TEST_K="${TEST_K:-1}"
  MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-128}"
  MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-32}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-100}"
else
  EPOCHS="${EPOCHS:-200}"
  TEST_K="${TEST_K:-100}"
  MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-0}"
  MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-0}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-0}"
fi

for DATASET in ${DATASETS}; do
  "${PYTHON_BIN}" experiments/seabed_versions/v10_kg_tie_aware_training/main.py \
    --dataset "${DATASET}" \
    --dataset-root "${DATA_ROOT}/${DATASET}" \
    --model-train 1 \
    --model-epoch-start 0 \
    --model-epoch-end "${EPOCHS}" \
    --test-k "${TEST_K}" \
    --topk-approach parallel \
    --cost-mode unit \
    --ged-column 3 \
    --use-raw-features 1 \
    --max-train-pairs "${MAX_TRAIN_PAIRS}" \
    --max-val-pairs "${MAX_VAL_PAIRS}" \
    --max-test-pairs "${MAX_TEST_PAIRS}" \
    --model-name "${MODEL_NAME}" \
    --model-path "${MODEL_PATH}" \
    --result-path "${RESULT_PATH}"
done
