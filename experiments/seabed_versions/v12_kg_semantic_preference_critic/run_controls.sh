#!/usr/bin/env bash
set -euo pipefail

# Run the minimum V12 attribution matrix. Override DATASETS, SMOKE, or
# PYTHON_BIN when running on a different machine or with a longer budget.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/vermouth/miniconda3/envs/gedranker/bin/python}"
DATASETS="${DATASETS:-LUBM SWDF}"
SMOKE="${SMOKE:-1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

run_control() {
  local control_mode="$1"
  local relation_mode="$2"
  echo
  echo "=== V12 control=${control_mode} relation=${relation_mode} ==="
  DATASETS="${DATASETS}" \
    CONTROL_MODE="${control_mode}" \
    RELATION_MODE="${relation_mode}" \
    SMOKE="${SMOKE}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/train.sh"
}

# Full/raw is the proposed method. The remaining runs separate critic and
# relation-topology effects before any full 200-epoch claim is made.
run_control full raw
run_control no_critic raw
run_control full constant
run_control full shuffled
