#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASETS="${DATASETS:-LUBM SWDF YAGO WIKIDATA}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
TEST_K="${TEST_K:-100}"
REPAIR_MODE="${REPAIR_MODE:-two_swap}"
REPAIR_MAX_ITERATIONS="${REPAIR_MAX_ITERATIONS:-20}"
REPAIR_CANDIDATE_BATCH_SIZE="${REPAIR_CANDIDATE_BATCH_SIZE:-2048}"
SAVE_PAIR_DETAILS="${SAVE_PAIR_DETAILS:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  TEST_K="${TEST_K_SMOKE:-5}"
  MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-100}"
else
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
    YAGO)
      CHECKPOINT="model_save/YAGO_200_GEDRankerSEABED_main_col3_unit_BPR_20260727_040001.pt"
      ;;
    WIKIDATA)
      CHECKPOINT="model_save/WIKIDATA_200_GEDRankerSEABED_main_col3_unit_BPR_20260727_062631.pt"
      ;;
    *)
      echo "Unsupported dataset: ${DATASET}" >&2
      exit 2
      ;;
  esac

  "${PYTHON_BIN}" experiments/seabed_versions/v1_certified_repair/main.py \
    --dataset "${DATASET}" \
    --dataset-root "${DATA_ROOT}/${DATASET}" \
    --checkpoint-path "${CHECKPOINT}" \
    --test-k "${TEST_K}" \
    --max-test-pairs "${MAX_TEST_PAIRS}" \
    --repair-mode "${REPAIR_MODE}" \
    --repair-max-iterations "${REPAIR_MAX_ITERATIONS}" \
    --repair-candidate-batch-size "${REPAIR_CANDIDATE_BATCH_SIZE}" \
    --save-pair-details "${SAVE_PAIR_DETAILS}"
done
