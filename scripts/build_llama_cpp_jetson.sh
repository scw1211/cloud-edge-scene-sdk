#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${LLAMA_CPP_SOURCE_DIR:-${SDK_ROOT}/runtime/llama.cpp}"
BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-${SOURCE_DIR}/build-cuda}"
INSTALL_DIR="${LLAMA_CPP_INSTALL_DIR:-${SDK_ROOT}/runtime/bin}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-b9859}"
BUILD_JOBS="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

for command_name in git cmake c++ nvcc; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "缺少构建命令：${command_name}" >&2
    echo "请先安装 git、cmake、C++ 编译器和 JetPack CUDA Toolkit。" >&2
    exit 2
  fi
done

if [[ -z "${CUDA_ARCH:-}" ]]; then
  CUDA_ARCH="$(
    python3 - <<'PY'
try:
    import torch
    major, minor = torch.cuda.get_device_capability(0)
    print("{}{}".format(major, minor))
except Exception:
    print("")
PY
  )"
fi
if [[ -z "${CUDA_ARCH}" ]]; then
  echo "无法自动识别 CUDA 架构，请设置 CUDA_ARCH，例如 Orin Nano 使用 87。" >&2
  exit 2
fi

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${SOURCE_DIR}")"
  git clone --depth 1 --branch "${LLAMA_CPP_REF}" \
    https://github.com/ggml-org/llama.cpp.git "${SOURCE_DIR}"
else
  git -C "${SOURCE_DIR}" fetch --depth 1 origin "refs/tags/${LLAMA_CPP_REF}:refs/tags/${LLAMA_CPP_REF}"
  git -C "${SOURCE_DIR}" checkout --detach "${LLAMA_CPP_REF}"
fi

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF
cmake --build "${BUILD_DIR}" --target llama-server -j "${BUILD_JOBS}"

mkdir -p "${INSTALL_DIR}"
cp "${BUILD_DIR}/bin/llama-server" "${INSTALL_DIR}/llama-server"
chmod +x "${INSTALL_DIR}/llama-server"
"${INSTALL_DIR}/llama-server" --version
echo "llama-server 已安装到 ${INSTALL_DIR}/llama-server"
