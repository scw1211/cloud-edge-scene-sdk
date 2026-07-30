#!/usr/bin/env bash
set -euo pipefail
# 只重新计算/导出指标，适合已有 predictions.csv 后复查。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLASS_NAME="${1:?usage: evaluate_metrics.sh capsule|screw}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
IMG_SIZE="${IMG_SIZE:-160}"
RESULT_DIR="${RESULT_DIR:-}"
if [ -z "${RESULT_DIR}" ]; then
  RESULT_DIR="$(find "${ROOT_DIR}/results/${CLASS_NAME}" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
fi
test -n "${RESULT_DIR}" || { echo "No result run found under ${ROOT_DIR}/results/${CLASS_NAME}"; exit 2; }
"${PYTHON_BIN}" "${ROOT_DIR}/tools/verify_metrics.py" \
  --predictions "${RESULT_DIR}" \
  --base "${ROOT_DIR}/baseline/Infra" \
  --class-name "${CLASS_NAME}" --img-size "${IMG_SIZE}" \
  --output "${RESULT_DIR}/metrics.csv" --no-gate
