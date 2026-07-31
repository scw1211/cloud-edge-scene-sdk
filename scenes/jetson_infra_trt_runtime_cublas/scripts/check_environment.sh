#!/usr/bin/env bash
set -euo pipefail

# 检查 TX2 上构建/运行所需的系统命令、TensorRT 文件和 Python 预处理依赖。
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRTEXEC="${TRTEXEC:-trtexec}"
failed=0

require_command() {
  # 找得到命令就打印实际路径；找不到时标记 failed，最后统一退出。
  if command -v "$1" >/dev/null 2>&1; then
    printf 'OK      %s: %s\n' "$1" "$(command -v "$1")"
  else
    printf 'MISSING %s\n' "$1"
    failed=1
  fi
}

require_command cmake
require_command nvcc
require_command "${TRTEXEC}"

# TensorRT 头文件在 Jetson aarch64 系统上常见于 /usr/include/aarch64-linux-gnu。
if [ -f /usr/include/NvInfer.h ] || [ -f /usr/include/aarch64-linux-gnu/NvInfer.h ]; then
  echo "OK      TensorRT headers"
else
  echo "MISSING TensorRT headers (NvInfer.h)"
  failed=1
fi

# ldconfig 能看到 libnvinfer.so，说明运行时链接库已安装。
if ldconfig -p 2>/dev/null | grep -q 'libnvinfer\.so'; then
  echo "OK      TensorRT runtime library"
else
  echo "MISSING TensorRT runtime library (libnvinfer.so)"
  failed=1
fi

# prepare_inputs.py 需要 Python OpenCV/NumPy；正式 C++ 运行不依赖 PyTorch。
if "${PYTHON_BIN}" -c 'import cv2, numpy; print("OK      Python preprocessing: cv2=%s numpy=%s" % (cv2.__version__, numpy.__version__))'; then
  :
else
  echo "MISSING Python OpenCV/NumPy; set PYTHON_BIN to the deployment environment"
  failed=1
fi

# 任一依赖缺失都阻止继续构建/运行，避免后续错误更难定位。
if [ "${failed}" -ne 0 ]; then
  echo "Environment is incomplete. Build/run is intentionally blocked until every requirement is available."
  exit 1
fi
echo "Environment check passed."
