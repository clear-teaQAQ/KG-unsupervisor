#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASETS="${DATASETS:-LUBM SWDF YAGO WIKIDATA}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
SOURCE="${SOURCE:-V10}"
EPOCHS="${EPOCHS:-200}"
RESULT_PATH="experiments/seabed_versions/v10_kg_tie_aware_training/raw_eval_results"

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

baseline_checkpoint() {
  local dataset="$1"
  case "${dataset}" in
    LUBM|SWDF)
      latest_match "model_save/${dataset}_200_GEDRankerSEABED_main_col3_unit_BPR_*.pt"
      ;;
    YAGO|WIKIDATA)
      latest_match "experiments/seabed_versions/v4_corrected_training/checkpoints/${dataset}_200_GEDRankerSEABED_v4_corrected_training_col3_unit_BPR_*.pt"
      ;;
    *)
      return 1
      ;;
  esac
}

for DATASET in ${DATASETS}; do
  if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    CHECKPOINT="${CHECKPOINT_PATH}"
  elif [[ "${SOURCE}" == "BASELINE" ]]; then
    CHECKPOINT="$(baseline_checkpoint "${DATASET}")"
  elif [[ "${SOURCE}" == "V10" ]]; then
    CHECKPOINT="$(latest_match "experiments/seabed_versions/v10_kg_tie_aware_training/checkpoints/${DATASET}_${EPOCHS}_GEDRankerSEABED_v10_kg_tie_aware_col3_unit_BPR_*.pt")"
  else
    echo "Unknown SOURCE=${SOURCE}; use V10 or BASELINE." >&2
    exit 2
  fi

  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found for ${DATASET}: ${CHECKPOINT}" >&2
    exit 2
  fi

  if [[ "${SMOKE:-0}" == "1" ]]; then
    TEST_K="${TEST_K:-1}"
    MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-100}"
  else
    TEST_K="${TEST_K:-100}"
    MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-0}"
  fi

  echo "Raw evaluation: dataset=${DATASET} source=${SOURCE} checkpoint=${CHECKPOINT}"
  CHECKPOINT_PATH="${CHECKPOINT}" \
    "${PYTHON_BIN}" experiments/seabed_versions/v10_kg_tie_aware_training/evaluate_main.py \
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
      --model-name "GEDRankerSEABED_v10_raw_${SOURCE}" \
      --model-path "experiments/seabed_versions/v10_kg_tie_aware_training/checkpoints" \
      --result-path "${RESULT_PATH}"
done
