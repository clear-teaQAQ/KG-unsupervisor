#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
DATASETS="${DATASETS:-LUBM SWDF YAGO WIKIDATA}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
MAX_PAIRS="${MAX_PAIRS:-100}"
V6_RESULT_DIR="experiments/seabed_versions/v6_semantic_tie_break/results"
RESULT_DIR="experiments/seabed_versions/v7_embedding_evidence_audit/results"

latest_v6_path() {
  local dataset="$1"
  { find "${V6_RESULT_DIR}" -maxdepth 1 -type f \
      -name "paths_SEABED_v6_semantic_tie_break_${dataset}_test_k5_two_swap_*.jsonl" \
      -printf '%T@ %p\n' || true; } | sort -n | tail -1 | cut -d' ' -f2-
}

for DATASET in ${DATASETS}; do
  PATH_FILE="$(latest_v6_path "${DATASET}")"
  if [[ -z "${PATH_FILE}" || ! -f "${PATH_FILE}" ]]; then
    echo "V6 smoke path file not found for ${DATASET}." >&2
    exit 2
  fi
  "${PYTHON_BIN}" \
    experiments/seabed_versions/v7_embedding_evidence_audit/audit_embeddings.py \
    --dataset "${DATASET}" \
    --dataset-root "${DATA_ROOT}/${DATASET}" \
    --path-file "${PATH_FILE}" \
    --output "${RESULT_DIR}/embedding_evidence_${DATASET}.json" \
    --max-pairs "${MAX_PAIRS}"
done

