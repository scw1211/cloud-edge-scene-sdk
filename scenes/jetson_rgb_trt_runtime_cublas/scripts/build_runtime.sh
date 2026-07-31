#!/usr/bin/env bash
set -euo pipefail
# 编译 C++/CUDA 推理程序 rgb_patchcore_trt。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
# Ubuntu 18.04 自带 CMake 3.10，早于 -S/-B 语法，所以使用传统 in-build-dir 调用方式。
cmake "${ROOT_DIR}" -DCMAKE_BUILD_TYPE=Release
# TX2 资源有限，默认 -j2 避免编译时占用过高内存。
cmake --build . -- -j2
