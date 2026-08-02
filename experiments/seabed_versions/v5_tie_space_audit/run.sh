#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASETS="${DATASETS:-LUBM SWDF YAGO WIKIDATA}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
MAX_PAIRS="${MAX_PAIRS:-100}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
RESULT_DIR="experiments/seabed_versions/v5_tie_space_audit/results"

latest_path_file() {
  local directory="$1"
  local pattern="$2"
  { find "${directory}" -maxdepth 1 -type f -name "${pattern}" -printf '%T@ %p\n' || true; } \
    | sort -n | tail -1 | cut -d' ' -f2-
}

for DATASET in ${DATASETS}; do
  case "${DATASET}" in
    LUBM|SWDF)
      PATH_FILE="$(latest_path_file \
        experiments/seabed_versions/v2_edit_path_audit/results \
        "paths_SEABED_v2_edit_path_audit_${DATASET}_test_k100_two_swap_*.jsonl")"
      ;;
    YAGO|WIKIDATA)
      PATH_FILE="$(latest_path_file \
        experiments/seabed_versions/v4_corrected_training/audit_results \
        "paths_SEABED_v4_corrected_training_${DATASET}_test_k100_two_swap_*.jsonl")"
      ;;
    *)
      echo "Unsupported dataset: ${DATASET}" >&2
      exit 2
      ;;
  esac
  if [[ -z "${PATH_FILE}" ]]; then
    echo "No full saved-path file found for ${DATASET}." >&2
    exit 2
  fi

  "${PYTHON_BIN}" experiments/seabed_versions/v5_tie_space_audit/analyze_saved_paths.py \
    --dataset "${DATASET}" \
    --dataset-root "${DATA_ROOT}/${DATASET}" \
    --path-file "${PATH_FILE}" \
    --output "${RESULT_DIR}/tie_space_${DATASET}.json" \
    --max-pairs "${MAX_PAIRS}" \
    --max-iterations "${MAX_ITERATIONS}"
done
