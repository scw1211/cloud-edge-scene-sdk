# Jetson TX2 Environment Setup

> 本文档说明如何在 Jetson TX2/JetPack 4.x 环境中安装依赖、检查 TensorRT/CUDA、
> 构建 engine/runtime，并用预处理输入跑一次正式验证。

These commands target Ubuntu 18.04 with JetPack 4.x, the usual TX2 software
stack. Run them on the TX2 while it has access to the NVIDIA JetPack APT
repository.

## 1. Confirm the JetPack base

> 先确认系统自带的 JetPack/CUDA/TensorRT 是否已经可用，避免不必要地重装 NVIDIA 组件。

```bash
cat /etc/nv_tegra_release
nvcc --version
trtexec --version
```

If `nvcc` and `trtexec` already work, do not reinstall CUDA or TensorRT.

## 2. Install development and preprocessing dependencies

> 这里安装的是编译 C++/CUDA、构建 TensorRT engine、以及离线预处理 PNG 所需的软件包。

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake pkg-config \
  libopencv-dev python3-opencv python3-numpy \
  cuda-toolkit-10-2 \
  tensorrt libnvinfer-bin libnvinfer-dev libnvinfer-plugin-dev \
  libnvonnxparsers-dev libnvparsers-dev
```

`cuda-toolkit-10-2` is the JetPack 4 CUDA toolkit package. If APT reports that
it is already installed, that is expected. If TensorRT package names are not
found, the JetPack APT source is missing; repair/reinstall JetPack with NVIDIA
SDK Manager instead of installing an x86 TensorRT package.

## 3. Verify dependencies

> `check_environment.sh` 会把缺失项一次性列出来，比等到 CMake 或运行时失败更容易排查。

From the extracted `jetson_rgb_trt_runtime` directory:

```bash
PYTHON_BIN=/usr/bin/python3 ./scripts/check_environment.sh
```

Expected result: every line starts with `OK` and the final line is
`Environment check passed.`

## 4. Build assets and runtime on the TX2

> ONNX 可以随包携带，但 TensorRT engine 和 CUDA runtime 建议在目标 TX2 上构建，
> 以匹配本机 TensorRT 版本、GPU 架构和库路径。

```bash
./scripts/build_engine.sh
./scripts/build_runtime.sh
```

The TensorRT engine must be built on this TX2. Its `patch_features` output is
explicitly configured as FP16 for the PatchCore CUDA distance kernel. Do not copy an engine produced
on a cloud x86 GPU to the TX2.

## 5. Prepare inputs outside the real-time path

> 预处理会生成 `.nchw.f32`，正式推理只读取这些张量，从而把实时路径收缩到
> H2D 拷贝、TensorRT、CUDA PatchCore 和结果写出。

```bash
DATASET_PATH=~/deploy/jetson_rgb_runtime/datasets/MulSen_AD \
  PYTHON_BIN=/usr/bin/python3 ./scripts/prepare_dataset.sh capsule
```

This makes normalized `NCHW float32` tensors once. The real-time command then
uses only those tensors:

```bash
DATASET_PATH=~/deploy/jetson_rgb_runtime/datasets/MulSen_AD \
  ./scripts/run_predict_preprocessed.sh capsule
```

For formal validation, use `./scripts/acceptance.sh capsule`. It checks RSS,
preprocessed single-image latency, and the agreed metric thresholds.
