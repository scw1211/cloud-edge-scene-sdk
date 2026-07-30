#!/usr/bin/env bash
set -euo pipefail
# 将指定类别的 Infrared/test PNG 预处理为 .nchw.f32，移出实时推理计时路径。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLASS_NAME="${1:?usage: prepare_dataset.sh capsule|screw}"
# 指定数据集路径
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to MulSen_AD}"
# 指定预处理数据集路径
PREPROCESSED_DIR="${ROOT_DIR}/preprocessed_new/${CLASS_NAME}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
IMG_SIZE="${IMG_SIZE:-160}"
# 输出目录默认按类别分开，run_predict_preprocessed.sh 会读取同一路径。
"${PYTHON_BIN}" "${ROOT_DIR}/tools/prepare_inputs.py" \
  --input-dir "${DATASET_PATH}/${CLASS_NAME}/Infrared/test" \
  --output-dir "${PREPROCESSED_DIR}" --img-size "${IMG_SIZE}"
