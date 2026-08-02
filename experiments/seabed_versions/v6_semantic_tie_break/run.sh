#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASETS="${DATASETS:-LUBM SWDF YAGO WIKIDATA}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
SEMANTIC_MAX_ITERATIONS="${SEMANTIC_MAX_ITERATIONS:-20}"
SAVE_PATHS="${SAVE_PATHS:-1}"
MAX_SAVED_PATHS="${MAX_SAVED_PATHS:-100}"
V4_CHECKPOINT_DIR="experiments/seabed_versions/v4_corrected_training/checkpoints"

latest_v4_checkpoint() {
  local dataset="$1"
  { find "${V4_CHECKPOINT_DIR}" -maxdepth 1 -type f \
      -name "${dataset}_200_GEDRankerSEABED_v4_corrected_training_col3_unit_BPR_*.pt" \
      -printf '%T@ %p\n' || true; } | sort -n | tail -1 | cut -d' ' -f2-
}

if [[ "${SMOKE:-0}" == "1" ]]; then
  TEST_K="${TEST_K:-5}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-100}"
else
  TEST_K="${TEST_K:-100}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-0}"
fi

for DATASET in ${DATASETS}; do
  case "${DATASET}" in
    LUBM)
      CHECKPOINT="model_save/LUBM_200_GEDRankerSEABED_main_col3_unit_BPR_20260726_215721.pt"
      ;;
    SWDF)
      CHECKPOINT="model_save/SWDF_200_GEDRankerSEABED_main_col3_unit_BPR_20260727_005926.pt"
      ;;
    YAGO|WIKIDATA)
      CHECKPOINT="$(latest_v4_checkpoint "${DATASET}")"
      ;;
    *)
      echo "Unsupported dataset: ${DATASET}" >&2
      exit 2
      ;;
  esac
  if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found for ${DATASET}: ${CHECKPOINT}" >&2
    exit 2
  fi

  "${PYTHON_BIN}" experiments/seabed_versions/v6_semantic_tie_break/main.py \
    --dataset "${DATASET}" \
    --dataset-root "${DATA_ROOT}/${DATASET}" \
    --checkpoint-path "${CHECKPOINT}" \
    --test-k "${TEST_K}" \
    --max-test-pairs "${MAX_TEST_PAIRS}" \
    --semantic-max-iterations "${SEMANTIC_MAX_ITERATIONS}" \
    --save-paths "${SAVE_PATHS}" \
    --max-saved-paths "${MAX_SAVED_PATHS}"
done
