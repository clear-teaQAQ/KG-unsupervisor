#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASETS="${DATASETS:-LUBM SWDF}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
RELATION_MODE="${RELATION_MODE:-raw}"
V13_MODE="${V13_MODE:-baseline}"
V13_ANCHOR_GATE_INIT="${V13_ANCHOR_GATE_INIT:-0.0}"
EPOCHS="${EPOCHS:-200}"
RESULT_PATH="${SCRIPT_DIR}/raw_eval_results"

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
    CHECKPOINT="$(latest_match "${SCRIPT_DIR}/checkpoints/${DATASET}_${EPOCHS}_GEDRankerSEABED_v13_${V13_MODE}_${RELATION_MODE}_col3_unit_BPR_*.pt" || true)"
  fi
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "V13 checkpoint not found for ${DATASET}: ${CHECKPOINT}" >&2
    exit 2
  fi

  if [[ "${SMOKE:-0}" == "1" ]]; then
    TEST_K="${TEST_K:-1}"
    MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-100}"
  else
    TEST_K="${TEST_K:-100}"
    MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-0}"
  fi

  echo "V13 raw evaluation: dataset=${DATASET} mode=${V13_MODE}/${RELATION_MODE} checkpoint=${CHECKPOINT}"
  CHECKPOINT_PATH="${CHECKPOINT}" \
    RELATION_MODE="${RELATION_MODE}" \
    V13_MODE="${V13_MODE}" \
    V13_ANCHOR_GATE_INIT="${V13_ANCHOR_GATE_INIT}" \
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
      --model-name "GEDRankerSEABED_v13_eval_${V13_MODE}_${RELATION_MODE}" \
      --model-path "${SCRIPT_DIR}/checkpoints" \
      --result-path "${RESULT_PATH}"
done

