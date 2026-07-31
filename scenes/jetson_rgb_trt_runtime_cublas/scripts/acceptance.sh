#!/usr/bin/env bash
set -euo pipefail
# 一键验收脚本：运行预处理推理，检查 RSS/延迟门禁，再计算指标门禁。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLASS_NAME="${1:?usage: acceptance.sh capsule|screw}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to MulSen_AD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/${CLASS_NAME}/${RUN_ID}}"
TIME_LOG="${RESULT_DIR}/time.txt"
mkdir -p "${RESULT_DIR}"
failures=()

# 使用 /usr/bin/time -v 采集整个推理命令的最大常驻内存，作为 TX2 部署 RSS 门禁。
if OUTPUT_DIR="${RESULT_DIR}" RUN_ID="${RUN_ID}" /usr/bin/time -v -o "${TIME_LOG}" "${ROOT_DIR}/scripts/run_predict_preprocessed.sh" "${CLASS_NAME}"; then
  echo "Prediction command passed"
else
  status=$?
  echo "Prediction command failed: exit ${status}"
  failures+=("prediction command exit ${status}")
fi

if [ -f "${TIME_LOG}" ]; then
  RSS_KB="$(awk -F: '/Maximum resident set size/ {gsub(/ /, "", $2); print $2}' "${TIME_LOG}")"
else
  RSS_KB=""
fi
if [ -z "${RSS_KB}" ]; then
  echo "RSS gate failed: cannot read maximum resident set size from ${TIME_LOG}"
  failures+=("RSS missing")
elif [ "${RSS_KB}" -gt 512000 ]; then
  echo "RSS gate failed: ${RSS_KB} KB > 512000 KB"
  failures+=("RSS ${RSS_KB} KB > 512000 KB")
else
  echo "RSS gate passed: ${RSS_KB} KB <= 512000 KB"
fi

# latency.csv 由 C++ runtime 写出：end_to_end_ms 是端到端单图耗时，cold_start_to_result_ms 只在第一张图记录冷启动耗时。
LATENCY_CSV="${RESULT_DIR}/latency.csv"
if [ -f "${LATENCY_CSV}" ]; then
  E2E_MAX_MS="$(awk -F, 'NR == 1 { for (i = 1; i <= NF; ++i) if ($i == "end_to_end_ms") c = i; next } NR > 1 && $c > max { max = $c } END { print max + 0 }' "${LATENCY_CSV}")"
  COLD_START_MS="$(awk -F, 'NR == 1 { for (i = 1; i <= NF; ++i) if ($i == "cold_start_to_result_ms") c = i; next } NR == 2 { print $c }' "${LATENCY_CSV}")"
else
  E2E_MAX_MS=""
  COLD_START_MS=""
fi
if [ -z "${E2E_MAX_MS}" ]; then
  echo "End-to-end latency gate failed: cannot read ${LATENCY_CSV}"
  failures+=("end-to-end latency missing")
elif awk "BEGIN { exit !(${E2E_MAX_MS} >= 50.0) }"; then
  echo "End-to-end latency gate failed: ${E2E_MAX_MS} ms >= 50 ms"
  failures+=("end-to-end latency ${E2E_MAX_MS} ms >= 50 ms")
else
  echo "End-to-end latency gate passed: ${E2E_MAX_MS} ms < 50 ms"
fi
if [ -z "${COLD_START_MS}" ]; then
  echo "Cold-start latency gate failed: cannot read first-image cold start from ${LATENCY_CSV}"
  failures+=("cold-start latency missing")
elif awk "BEGIN { exit !(${COLD_START_MS} >= 50.0) }"; then
  echo "Cold-start latency gate failed: ${COLD_START_MS} ms >= 50 ms"
  failures+=("cold-start latency ${COLD_START_MS} ms >= 50 ms")
else
  echo "Cold-start latency gate passed: ${COLD_START_MS} ms < 50 ms"
fi

# 指标门禁委托 Python verifier，和 base/RGB 中的权威基线按比例比较。
if "${PYTHON_BIN}" "${ROOT_DIR}/tools/verify_metrics.py" --predictions "${RESULT_DIR}" --base "${ROOT_DIR}/baseline/RGB" --class-name "${CLASS_NAME}" --img-size 160 --output "${RESULT_DIR}/metrics.csv"; then
  echo "Metric gate passed"
else
  status=$?
  echo "Metric gate failed: verifier exit ${status}"
  failures+=("metric gate exit ${status}")
fi

if [ "${#failures[@]}" -ne 0 ]; then
  echo "Acceptance failed:"
  for failure in "${failures[@]}"; do
    echo "- ${failure}"
  done
  exit 1
fi

echo "Acceptance passed: RSS=${RSS_KB} KB, max end-to-end=${E2E_MAX_MS} ms, cold start=${COLD_START_MS} ms, RGB Pixel AUROC > 80% baseline, RGB Pixel F1 > 80% baseline"
