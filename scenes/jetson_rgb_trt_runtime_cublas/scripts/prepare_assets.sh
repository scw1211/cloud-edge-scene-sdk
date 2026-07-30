#!/usr/bin/env bash
set -euo pipefail
# 从相邻的 jetson_rgb_runtime 重新生成 ONNX 和 .pcbank；正常离线包已包含资产时无需执行。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_RUNTIME="${SOURCE_RUNTIME:-${ROOT_DIR}/../jetson_rgb_runtime}"
ASSET_DIR="${ROOT_DIR}/assets"

mkdir -p "${ASSET_DIR}"
# 导出 backbone ONNX，输入尺寸固定 160，输出名 patch_features 要与 C++ binding 查找一致。
"${PYTHON_BIN}" "${ROOT_DIR}/tools/export_onnx.py" --runtime-root "${SOURCE_RUNTIME}" --checkpoint "${SOURCE_RUNTIME}/checkpoints/vit_small_patch8_224_dino.pth" --backbone vit_small_patch8_224_dino --output "${ASSET_DIR}/vit_small_patch8_160.onnx" --img-size 160
for class_name in capsule screw; do
  # 每个类别有独立 PatchCore 记忆库，转换为部署端读取的紧凑 FP16 格式。
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/convert_memory_bank.py" --input "${SOURCE_RUNTIME}/memory_banks/${class_name}_RGB_vit_small_patch8_224_dino_img160_fc0.05_eps0.9_seed42.pt" --output "${ASSET_DIR}/${class_name}.pcbank"
done
