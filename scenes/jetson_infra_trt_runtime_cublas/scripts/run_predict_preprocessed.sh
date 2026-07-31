#!/usr/bin/env bash
set -euo pipefail
# 正式低延迟推理入口：读取已预处理 .nchw.f32，并把红外原始 PNG 根目录传给 C++ 用于写回路径。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLASS_NAME="${1:?usage: run_predict_preprocessed.sh capsule|screw}"
# 指定数据集路径
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to MulSen_AD}"
# 指定预处理数据集路径
PREPROCESSED_DIR="${ROOT_DIR}/preprocessed_new/${CLASS_NAME}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/${CLASS_NAME}/${RUN_ID}}"
BANK_PATH="${BANK_PATH:-${ROOT_DIR}/assets/${CLASS_NAME}.pcbank}"
ENGINE_PATH="${ENGINE_PATH:-${ROOT_DIR}/assets/vit_small_patch8_160_fp16.engine}"
# 没有预处理输入时直接提示先运行 prepare_dataset.sh，避免 C++ 静默处理空目录。
test -d "${PREPROCESSED_DIR}" || { echo "Missing prepared inputs: ${PREPROCESSED_DIR}. Run scripts/prepare_dataset.sh first."; exit 2; }
test -f "${BANK_PATH}" || { echo "Missing memory bank: ${BANK_PATH}"; exit 2; }
test -f "${ENGINE_PATH}" || { echo "Missing TensorRT engine: ${ENGINE_PATH}. Run scripts/build_engine.sh first."; exit 2; }
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# --preprocessed 后的路径是源 PNG 根目录，用于 predictions.csv 中保留可追溯原图路径。
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BANK_PATH=${BANK_PATH}"
echo "ENGINE_PATH=${ENGINE_PATH}"
"${ROOT_DIR}/build/infra_patchcore_trt" "${ENGINE_PATH}" "${BANK_PATH}" "${PREPROCESSED_DIR}" "${OUTPUT_DIR}" --preprocessed "${DATASET_PATH}/${CLASS_NAME}/Infrared/test"
