#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
DATASETS="${DATASETS:-LUBM SWDF}"
SMOKE="${SMOKE:-1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

run_mode() {
  local mode="$1"
  echo
  echo "=== V13 mode=${mode} relation=raw ==="
  DATASETS="${DATASETS}" \
    RELATION_MODE="raw" \
    V13_MODE="${mode}" \
    SMOKE="${SMOKE}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/train.sh"
}

# Baseline must be checked before any improvement claim. Gated anchor is the
# first unit-cost-aligned residual candidate.
run_mode baseline
run_mode gated_anchor

