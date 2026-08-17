#!/usr/bin/env bash
set -euo pipefail

# Run this watcher in the same host/tmux environment as the active GPU job.
# It waits for the formal SWDF result, then starts the remaining datasets.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="${LOG_PATH:-${SCRIPT_DIR}/v19_generator_edge_seed0_200.log}"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/projects/SEABED-main/data}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
SEED="${SEED:-0}"

while true; do
  if [[ -f "${LOG_PATH}" ]] && grep -qE \
    "Saved V19 manifest: .*manifest_SWDF_generator_edge_epoch200_" \
    "${LOG_PATH}"; then
    break
  fi
  sleep "${WAIT_SECONDS}"
done

echo "Detected completed formal V19 SWDF run at $(date -Is)."

set -o pipefail
PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=0 \
DATASETS="YAGO WIKIDATA" \
EPOCHS=200 \
TEST_K=100 \
SEED="${SEED}" \
V19_MODE=generator_edge \
V19_GATE_INIT=0.0 \
V19_EDGE_HIDDEN_DIM=32 \
V18_GATE_INIT=0.0 \
V18_EDGE_HIDDEN_DIM=32 \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${SCRIPT_DIR}/train.sh" \
2>&1 | tee "${SCRIPT_DIR}/v19_yago_wikidata_seed${SEED}_200.log"
