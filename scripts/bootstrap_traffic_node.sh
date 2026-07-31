#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || ("$1" != "edge" && "$1" != "cloud") ]]; then
  echo "用法：bash scripts/bootstrap_traffic_node.sh edge|cloud" >&2
  exit 2
fi

ROLE="$1"
SDK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${SDK_ROOT}/.venv-${ROLE}}"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 8):
    raise SystemExit("需要 Python 3.8 或更高版本")
print("Python", sys.version.split()[0])
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  if [[ "${ROLE}" == "edge" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
  else
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip install -r "${SDK_ROOT}/requirements.txt"
"${VENV_PYTHON}" -m pip install -e "${SDK_ROOT}"
"${VENV_PYTHON}" -m pip install \
  -r "${SDK_ROOT}/scenes/freeway_traffic/requirements-runtime.txt"
"${VENV_PYTHON}" -m pip install \
  -e "${SDK_ROOT}/scenes/freeway_traffic" --no-deps

if [[ "${ROLE}" == "edge" ]]; then
  if [[ -n "${JETSON_TORCH_WHEEL_URL:-}" ]]; then
    "${VENV_PYTHON}" -m pip install "${JETSON_TORCH_WHEEL_URL}"
  fi
  "${VENV_PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "当前 torch 无法使用 CUDA。请先安装与 JetPack 对应的 NVIDIA PyTorch，"
        "或通过 JETSON_TORCH_WHEEL_URL 指定 wheel。"
    )
print("torch", torch.__version__, "CUDA", torch.cuda.get_device_name(0))
PY

  LLAMA_SERVER_PATH="${LLAMA_SERVER_PATH:-${SDK_ROOT}/runtime/bin/llama-server}"
  if [[ ! -x "${LLAMA_SERVER_PATH}" ]]; then
    bash "${SDK_ROOT}/scripts/build_llama_cpp_jetson.sh"
    LLAMA_SERVER_PATH="${SDK_ROOT}/runtime/bin/llama-server"
  fi
  "${VENV_PYTHON}" "${SDK_ROOT}/scenes/freeway_traffic/install_full_assets.py" --edge
  "${VENV_PYTHON}" "${SDK_ROOT}/scenes/freeway_traffic/deploy_node.py" check \
    --role edge \
    --llama-binary "${LLAMA_SERVER_PATH}" \
    --device cuda
  echo "边缘节点安装完成。llama-server=${LLAMA_SERVER_PATH}"
else
  "${VENV_PYTHON}" "${SDK_ROOT}/scenes/freeway_traffic/deploy_node.py" check \
    --role cloud
  if [[ "${WITH_CLOUD_QWEN9B:-0}" == "1" ]]; then
    "${VENV_PYTHON}" -m model_bundle.install_models --cloud
  fi
  echo "云端节点安装完成。"
fi

echo "Python 环境：${VENV_PYTHON}"
