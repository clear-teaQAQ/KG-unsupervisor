#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASETS="${DATASETS:-YAGO WIKIDATA}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
MODEL_NAME="${MODEL_NAME:-GEDRankerSEABED_v4_corrected_training_col3_unit}"
MODEL_PATH="experiments/seabed_versions/v4_corrected_training/checkpoints"

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS="${EPOCHS:-1}"
  TEST_K="${TEST_K:-5}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-100}"
else
  EPOCHS="${EPOCHS:-200}"
  TEST_K="${TEST_K:-100}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-0}"
fi

for DATASET in ${DATASETS}; do
  CHECKPOINT_PATTERN="${DATASET}_${EPOCHS}_${MODEL_NAME}_BPR_*.pt"
  CHECKPOINT="$({ find "${MODEL_PATH}" -maxdepth 1 -type f -name "${CHECKPOINT_PATTERN}" -printf '%T@ %p\n' || true; } | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "${CHECKPOINT}" ]]; then
    echo "No V4 checkpoint found for ${DATASET}, epoch ${EPOCHS}." >&2
    exit 2
  fi

  "${PYTHON_BIN}" experiments/seabed_versions/v4_corrected_training/audit_main.py \
    --dataset "${DATASET}" \
    --dataset-root "${DATA_ROOT}/${DATASET}" \
    --checkpoint-path "${CHECKPOINT}" \
    --test-k "${TEST_K}" \
    --max-test-pairs "${MAX_TEST_PAIRS}" \
    --repair-mode two_swap \
    --save-paths 1 \
    --max-saved-paths 100
done
