#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
RESULT_PATH="${SCRIPT_DIR}/frozen_mapping_audit_results"
DEFAULT_CHECKPOINT="${SCRIPT_DIR}/checkpoints/SWDF_200_GEDRankerSEABED_v15_projected_input_relation_raw_col3_unit_BPR_20260812_011747.pt"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT}}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Frozen V15 checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  TEST_K="${TEST_K:-1}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-10}"
else
  TEST_K="${TEST_K:-100}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-0}"
fi

echo "V15 frozen mapping audit: checkpoint=${CHECKPOINT_PATH} test_k=${TEST_K} max_test_pairs=${MAX_TEST_PAIRS}"
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  RELATION_MODE=raw \
  V15_MODE=projected_input \
  SEED="${SEED:-0}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_frozen_audit_main.py" \
    --dataset SWDF \
    --dataset-root "${DATA_ROOT}/SWDF" \
    --model-train 0 \
    --model-epoch-start 0 \
    --model-epoch-end 200 \
    --test-k "${TEST_K}" \
    --topk-approach parallel \
    --cost-mode unit \
    --ged-column 3 \
    --use-raw-features 1 \
    --max-train-pairs 1 \
    --max-val-pairs 1 \
    --max-test-pairs "${MAX_TEST_PAIRS}" \
    --model-name GEDRankerSEABED_v15_frozen_mapping_audit \
    --model-path "${SCRIPT_DIR}/checkpoints" \
    --result-path "${RESULT_PATH}"

