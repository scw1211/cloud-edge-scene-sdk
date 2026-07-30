#!/usr/bin/env bash
set -euo pipefail
# 在当前 Jetson/TensorRT 环境中把 ONNX 编译为 FP16 engine。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRTEXEC="${TRTEXEC:-trtexec}"
TRT_WORKSPACE="${TRT_WORKSPACE:-1024}"
# outputIOFormats 强制 patch_features 输出为 FP16 CHW，匹配 C++/CUDA 后处理的 __half 输入。
"${TRTEXEC}" --onnx="${ROOT_DIR}/assets/vit_small_patch8_160.onnx" --saveEngine="${ROOT_DIR}/assets/vit_small_patch8_160_fp16.engine" --fp16 --outputIOFormats=fp16:chw --workspace="${TRT_WORKSPACE}"
