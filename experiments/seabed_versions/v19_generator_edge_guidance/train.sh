#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASETS="${DATASETS:-LUBM SWDF}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
V19_MODE="${V19_MODE:-generator_edge}"
MODEL_PATH="${SCRIPT_DIR}/checkpoints"
RESULT_PATH="${SCRIPT_DIR}/training_results"

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS="${EPOCHS:-1}"
  TEST_K="${TEST_K:-3}"
  MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-32}"
  MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-8}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-20}"
else
  EPOCHS="${EPOCHS:-200}"
  TEST_K="${TEST_K:-100}"
  MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-0}"
  MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-0}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-0}"
fi

for DATASET in ${DATASETS}; do
  echo "V19 training: dataset=${DATASET} mode=${V19_MODE} epochs=${EPOCHS}"
  SEED="${SEED:-0}" \
    V19_MODE="${V19_MODE}" \
    V19_GATE_INIT="${V19_GATE_INIT:-0.0}" \
    V19_EDGE_HIDDEN_DIM="${V19_EDGE_HIDDEN_DIM:-32}" \
    V18_GATE_INIT="${V18_GATE_INIT:-0.0}" \
    V18_EDGE_HIDDEN_DIM="${V18_EDGE_HIDDEN_DIM:-32}" \
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
      --model-name "GEDRankerSEABED_v19_${V19_MODE}" \
      --model-path "${MODEL_PATH}" \
      --result-path "${RESULT_PATH}"
done

