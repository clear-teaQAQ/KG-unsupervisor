#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS="${DATASETS:-SWDF}"

for MODE in raw projected_input; do
  echo "V15 control: mode=${MODE} datasets=${DATASETS}"
  DATASETS="${DATASETS}" \
    V15_MODE="${MODE}" \
    RELATION_MODE=raw \
    bash "${SCRIPT_DIR}/train.sh"
done
