#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASETS="${DATASETS:-LUBM SWDF}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
CONTROL_MODE="${CONTROL_MODE:-full}"
RELATION_MODE="${RELATION_MODE:-raw}"
MODEL_NAME="${MODEL_NAME:-GEDRankerSEABED_v12_semantic_${CONTROL_MODE}_${RELATION_MODE}}"

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
  CONTROL_MODE="${CONTROL_MODE}" RELATION_MODE="${RELATION_MODE}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" \
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
      --control-mode "${CONTROL_MODE}" \
      --relation-mode "${RELATION_MODE}"
done
