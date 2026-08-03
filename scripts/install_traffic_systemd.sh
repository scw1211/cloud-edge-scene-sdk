#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || ("$1" != "edge" && "$1" != "cloud") ]]; then
  echo "用法：bash scripts/install_traffic_systemd.sh edge|cloud" >&2
  exit 2
fi

ROLE="$1"
SDK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${RUN_USER:-$(id -un)}"
SERVICE_NAME="cloud-edge-traffic-${ROLE}"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ "${ROLE}" == "edge" ]]; then
  CLOUD_URL="${CLOUD_URL:-}"
  LLAMA_SERVER_PATH="${LLAMA_SERVER_PATH:-${SDK_ROOT}/runtime/bin/llama-server}"
  if [[ -z "${CLOUD_URL}" ]]; then
    echo "边缘节点必须设置 CLOUD_URL，例如 http://192.168.31.160:18100" >&2
    exit 2
  fi
  if [[ ! -x "${LLAMA_SERVER_PATH}" ]]; then
    echo "llama-server 不存在或不可执行：${LLAMA_SERVER_PATH}" >&2
    exit 2
  fi
  PYTHON_PATH="${VENV_DIR:-${SDK_ROOT}/.venv-edge}/bin/python"
  EXEC_START="${PYTHON_PATH} ${SDK_ROOT}/scenes/freeway_traffic/deploy_node.py run --role edge --cloud-url ${CLOUD_URL} --llama-binary ${LLAMA_SERVER_PATH} --parallel 1 --device cuda"
else
  PYTHON_PATH="${VENV_DIR:-${SDK_ROOT}/.venv-cloud}/bin/python"
  EXEC_START="${PYTHON_PATH} ${SDK_ROOT}/scenes/freeway_traffic/deploy_node.py run --role cloud"
fi

if [[ ! -x "${PYTHON_PATH}" ]]; then
  echo "Python环境不存在：${PYTHON_PATH}，请先运行 bootstrap_traffic_node.sh。" >&2
  exit 2
fi

TEMP_UNIT="$(mktemp)"
trap 'rm -f "${TEMP_UNIT}"' EXIT
{
  echo "[Unit]"
  echo "Description=Cloud Edge Traffic ${ROLE} Node"
  echo "After=network-online.target"
  echo "Wants=network-online.target"
  echo
  echo "[Service]"
  echo "Type=simple"
  echo "User=${RUN_USER}"
  echo "WorkingDirectory=${SDK_ROOT}"
  echo "ExecStart=${EXEC_START}"
  echo "Restart=on-failure"
  echo "RestartSec=2"
  echo "TimeoutStopSec=15"
  echo
  echo "[Install]"
  echo "WantedBy=multi-user.target"
} > "${TEMP_UNIT}"

sudo install -m 0644 "${TEMP_UNIT}" "${UNIT_PATH}"
sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"
sudo systemctl --no-pager --full status "${SERVICE_NAME}"
