#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS="${DATASETS:-SWDF}"

for MODE in raw projected_input; do
  echo "V15 evaluation control: mode=${MODE} datasets=${DATASETS}"
  DATASETS="${DATASETS}" \
    V15_MODE="${MODE}" \
    bash "${SCRIPT_DIR}/evaluate.sh"
done
