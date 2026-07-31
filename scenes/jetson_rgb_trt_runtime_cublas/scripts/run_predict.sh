#!/usr/bin/env bash
set -euo pipefail
# 直接用 PNG 运行推理；会把解码/resize/normalize 放进 C++ 进程，主要用于功能验证。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLASS_NAME="${1:?usage: run_predict.sh capsule|screw}"
# 指定数据集路径
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to MulSen_AD}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/${CLASS_NAME}/${RUN_ID}}"
BANK_PATH="${BANK_PATH:-${ROOT_DIR}/assets_compressed/rows384/${CLASS_NAME}.pcbank}"
test -f "${BANK_PATH}" || { echo "Missing memory bank: ${BANK_PATH}"; exit 2; }
# 限制 BLAS/OpenMP 线程数，避免部署端产生额外 CPU/RSS 抖动。
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# 参数顺序：engine、类别记忆库、原始图片目录、输出目录。
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BANK_PATH=${BANK_PATH}"
"${ROOT_DIR}/build/rgb_patchcore_trt" "${ROOT_DIR}/assets/vit_small_patch8_160_fp16.engine" "${BANK_PATH}" "${DATASET_PATH}/${CLASS_NAME}/RGB/test" "${OUTPUT_DIR}"
