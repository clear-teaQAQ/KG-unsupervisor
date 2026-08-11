#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASETS="${DATASETS:-LUBM SWDF}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
CONTROL_MODE="${CONTROL_MODE:-full}"
RELATION_MODE="${RELATION_MODE:-raw}"
EPOCHS="${EPOCHS:-200}"
MODEL_NAME="${MODEL_NAME:-GEDRankerSEABED_v12_semantic_${CONTROL_MODE}_${RELATION_MODE}}"

latest_match() {
  local pattern="$1"
  local matches=()
  shopt -s nullglob
  matches=( ${pattern} )
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 0 ]]; then
    return 1
  fi
  ls -1t "${matches[@]}" | sed -n '1p'
}

for DATASET in ${DATASETS}; do
  if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    CHECKPOINT="${CHECKPOINT_PATH}"
  else
    CHECKPOINT="$(latest_match "${SCRIPT_DIR}/checkpoints/${DATASET}_${EPOCHS}_${MODEL_NAME}_*.pt" || true)"
  fi
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "V12 checkpoint not found for ${DATASET}: ${CHECKPOINT}" >&2
    exit 2
  fi

  if [[ "${SMOKE:-0}" == "1" ]]; then
    TEST_K="${TEST_K:-1}"
    MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-100}"
  else
    TEST_K="${TEST_K:-100}"
    MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-0}"
  fi

  echo "V12 raw evaluation: dataset=${DATASET} control=${CONTROL_MODE} relation=${RELATION_MODE} checkpoint=${CHECKPOINT}"
  CHECKPOINT_PATH="${CHECKPOINT}" CONTROL_MODE="${CONTROL_MODE}" RELATION_MODE="${RELATION_MODE}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_main.py" \
      --dataset "${DATASET}" \
      --dataset-root "${DATA_ROOT}/${DATASET}" \
      --model-train 0 \
      --model-epoch-start 0 \
      --model-epoch-end "${EPOCHS}" \
      --test-k "${TEST_K}" \
      --topk-approach parallel \
      --cost-mode unit \
      --ged-column 3 \
      --use-raw-features 1 \
      --max-train-pairs 1 \
      --max-val-pairs 1 \
      --max-test-pairs "${MAX_TEST_PAIRS}" \
      --model-name "${MODEL_NAME}" \
      --control-mode "${CONTROL_MODE}" \
      --relation-mode "${RELATION_MODE}"
done
