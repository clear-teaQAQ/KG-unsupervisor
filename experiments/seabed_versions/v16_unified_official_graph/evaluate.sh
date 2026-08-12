#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASET="${DATASET:-SWDF}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
EPOCHS="${EPOCHS:-200}"
RESULT_PATH="${SCRIPT_DIR}/raw_eval_results"

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  CHECKPOINT_PATH="$(
    find "${SCRIPT_DIR}/checkpoints" -maxdepth 1 -type f \
      -name "${DATASET}_${EPOCHS}_GEDRankerSEABED_v16_unified_official_graph_BPR_*.pt" \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "V16 checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  TEST_K="${TEST_K:-3}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-20}"
else
  TEST_K="${TEST_K:-100}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-0}"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  SEED="${SEED:-0}" \
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
    --model-name GEDRankerSEABED_v16_unified_official_graph \
    --model-path "${SCRIPT_DIR}/checkpoints" \
    --result-path "${RESULT_PATH}"

