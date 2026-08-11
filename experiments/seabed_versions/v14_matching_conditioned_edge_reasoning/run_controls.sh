#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS="${DATASETS:-LUBM SWDF}"

for MODE in baseline matched_edge; do
  echo "V14 control: mode=${MODE} datasets=${DATASETS}"
  DATASETS="${DATASETS}" \
    V14_MODE="${MODE}" \
    V14_GATE_INIT=0.0 \
    bash "${SCRIPT_DIR}/train.sh"
done
