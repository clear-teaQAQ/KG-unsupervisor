#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASETS="${DATASETS:-LUBM SWDF}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
V18_MODE="${V18_MODE:-matched_edge}"
EPOCHS="${EPOCHS:-200}"
RESULT_PATH="${SCRIPT_DIR}/raw_eval_results"

latest_generator() {
  find "${SCRIPT_DIR}/checkpoints" -maxdepth 1 -type f \
    -name "${1}_${EPOCHS}_GEDRankerSEABED_v18_${V18_MODE}_BPR_*.pt" \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

for DATASET in ${DATASETS}; do
  GENERATOR="${CHECKPOINT_PATH:-$(latest_generator "${DATASET}")}"
  BASENAME="$(basename "${GENERATOR}" .pt)"
  DISCRIMINATOR="${DISCRIMINATOR_CHECKPOINT_PATH:-${SCRIPT_DIR}/checkpoints/discriminators/${BASENAME}_discriminator.pt}"
  if [[ ! -f "${GENERATOR}" || ! -f "${DISCRIMINATOR}" ]]; then
    echo "V18 checkpoint pair not found for ${DATASET}." >&2
    exit 2
  fi
  if [[ "${SMOKE:-0}" == "1" ]]; then
    TEST_K="${TEST_K:-3}"
    MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-20}"
  else
    TEST_K="${TEST_K:-100}"
    MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-0}"
  fi

  CHECKPOINT_PATH="${GENERATOR}" \
    DISCRIMINATOR_CHECKPOINT_PATH="${DISCRIMINATOR}" \
    SEED="${SEED:-0}" \
    V18_MODE="${V18_MODE}" \
    V18_GATE_INIT="${V18_GATE_INIT:-0.0}" \
    V18_EDGE_HIDDEN_DIM="${V18_EDGE_HIDDEN_DIM:-32}" \
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
      --model-name "GEDRankerSEABED_v18_${V18_MODE}" \
      --model-path "${SCRIPT_DIR}/checkpoints" \
      --result-path "${RESULT_PATH}"
done

