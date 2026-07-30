#!/usr/bin/env bash
set -euo pipefail
# 生成离线部署包：拷贝源码、脚本、文档、基线和资产，去掉无用 Python 字节码。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_NAME="jetson_infra_trt_runtime_cublas"
PACKAGE_DIR="${ROOT_DIR}/dist/${PACKAGE_NAME}"
rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}"
cp -a "${ROOT_DIR}/CMakeLists.txt" "${ROOT_DIR}/README.md" "${ROOT_DIR}/src" "${ROOT_DIR}/tools" "${ROOT_DIR}/scripts" "${ROOT_DIR}/baseline" "${ROOT_DIR}/docs" "${PACKAGE_DIR}/"
mkdir -p "${PACKAGE_DIR}/assets"
cp -a "${ROOT_DIR}/assets/vit_small_patch8_160.onnx" "${ROOT_DIR}/assets/capsule.pcbank" "${ROOT_DIR}/assets/screw.pcbank" "${PACKAGE_DIR}/assets/"
# __pycache__ 与宿主 Python 版本绑定，离线运行包不需要携带。
find "${PACKAGE_DIR}" -type d -name __pycache__ -prune -exec rm -rf {} +
ARCHIVE_PATH="${ROOT_DIR}/dist/${PACKAGE_NAME}.tar"
TEMP_ARCHIVE="${ARCHIVE_PATH}.tmp"
rm -f "${TEMP_ARCHIVE}"
# 先写临时 tar 并用 tar -tf 校验，再原子替换正式包。
tar -C "${ROOT_DIR}/dist" -cf "${TEMP_ARCHIVE}" "${PACKAGE_NAME}"
tar -tf "${TEMP_ARCHIVE}" >/dev/null
mv "${TEMP_ARCHIVE}" "${ARCHIVE_PATH}"
echo "Created ${ARCHIVE_PATH}"
