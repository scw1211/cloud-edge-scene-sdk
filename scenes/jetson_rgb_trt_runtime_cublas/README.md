# Jetson RGB TensorRT Runtime cuBLAS

> 这个目录是一个独立的 Jetson TX2 部署包，用 TensorRT/CUDA/C++ 替代原来的
> Python/PyTorch RGB PatchCore 推理流程。当前版本把 PatchCore 最近邻搜索替换为 cuBLAS GEMM 后端，用于和普通 CUDA kernel 版本做时延对比。
>`assets/` 放 ONNX、TensorRT engine 和类别记忆库，`src/` 是实时推理程序，
>`tools/` 是资产转换与指标校验工具，`scripts/` 是构建、预处理、推理、验收和打包入口。

Independent low-memory RGB PatchCore deployment runtime for Jetson TX2.
It replaces the Python/PyTorch prediction process with:

```text
OpenCV decode and preprocessing
-> TensorRT FP16 ViT-small/patch8 engine at 160 x 160
-> cuBLAS GEMM PatchCore memory-bank distance search
-> C++ image score and raw 160 x 160 score map output
```

Its acceptance gate is the authoritative `base/RGB` baseline:

| Class | Image AUROC | Pixel AUROC | Pixel F1 | Pixel AUPR | Pixel AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Capsule | 0.952 | 0.996 | 0.644 | 0.686 | 0.686 |
| Screw | 1.000 | 0.990 | 0.312 | 0.270 | 0.270 |

## TX2 Prerequisites

> 正常离线推理不需要 PyTorch；只有重新导出 ONNX 或转换原始 `.pt` 记忆库时才需要源 runtime 的 Python 环境。

- JetPack with TensorRT, CUDA, OpenCV development headers, `trtexec`, CMake,
  and GNU `time`.
- The archive already includes the cloud-validated ONNX model and compact
  FP16 memory banks. PyTorch is not required for the normal runtime flow.

## Build and Acceptance

> 推荐顺序是先检查环境，再构建 TensorRT engine 和 C++ runtime，然后按类别预处理数据，最后跑 acceptance 门禁。

For a TX2 environment setup guide, including the Ubuntu/JetPack package
commands, see [docs/jetson_tx2_setup.md](docs/jetson_tx2_setup.md).

From the extracted project directory on TX2:

```bash
PYTHON_BIN=/usr/bin/python3 ./scripts/check_environment.sh
./scripts/build_engine.sh
./scripts/build_runtime.sh
DATASET_PATH=~/deploy/jetson_rgb_runtime/datasets/MulSen_AD PYTHON_BIN=/usr/bin/python3 ./scripts/prepare_dataset.sh capsule
DATASET_PATH=~/deploy/jetson_rgb_runtime/datasets/MulSen_AD PYTHON_BIN=/usr/bin/python3 ./scripts/prepare_dataset.sh screw
DATASET_PATH=~/deploy/jetson_rgb_runtime/datasets/MulSen_AD ./scripts/acceptance.sh capsule
DATASET_PATH=~/deploy/jetson_rgb_runtime/datasets/MulSen_AD ./scripts/acceptance.sh screw
```

> `prepare_assets.sh` 会依赖相邻的 `jetson_rgb_runtime`，
> 用于重新生成 `assets/*.onnx` 和 `assets/*.pcbank`；离线包已经包含这些文件时不要重复跑。

`scripts/prepare_assets.sh` is only a fallback for regenerating assets from
the matching `jetson_rgb_runtime` source tree; do not run it for the normal offline package.

`check_environment.sh` requires CMake, CUDA (`nvcc`), TensorRT headers and
runtime, `trtexec`, and a Python environment containing OpenCV and NumPy.

For the latency-critical path, prepare test inputs once. This performs image
decode, RGB conversion, the same PIL bicubic resize used by the reference
runtime, and normalization outside the inference timing:

```bash
DATASET_PATH=~/deploy/jetson_rgb_runtime/datasets/MulSen_AD ./scripts/prepare_dataset.sh capsule
DATASET_PATH=~/deploy/jetson_rgb_runtime/datasets/MulSen_AD ./scripts/prepare_dataset.sh screw
```

> 正式延迟统计看预处理路径，避免 PNG 解码和 resize 把推理耗时放大；
>`run_predict.sh`直接读 PNG，更适合快速功能检查。

Use `scripts/run_predict_preprocessed.sh capsule` for normal preprocessed
inference. It consumes `.nchw.f32` tensors and writes `latency.csv` with
`inference_ms` (GPU input upload through image-score result), `end_to_end_ms`
(prepared-input read through result files written), and the first-image
`cold_start_to_result_ms` (process initialization through result files written).

`acceptance.sh` fails unless all conditions hold:

```text
Maximum resident set size <= 512000 KB
Maximum prepared-input end-to-end single-image latency < 50 ms
First-image cold-start-to-result latency < 50 ms
RGB Pixel AUROC is greater than 80% of the original `baseline/RGB`
RGB Pixel F1 is greater than 80% of the original `baseline/RGB`
Image AUROC, RGB Pixel AUPR, and RGB Pixel AP are reported for comparison only and do not block deployment.
```

> 如果想在 TX2 上观察系统整体的内存、GPU、CPU 状态，可以另外打开一个终端，运行:

For system-level RAM/GPU monitoring, run this in another TX2 terminal:

```bash
sudo tegrastats --interval 500
```

## Outputs

> C++ runtime 的核心输出有三类：`predictions.csv` 记录图像分数，`map_*.f32`
> 是像素级原始异常图，`latency.csv` 用于记录推理延迟。

`results/<class>/predictions.csv` stores image scores and references to raw
float32 `160 x 160` score maps (`.f32`). The Python verifier is evaluation
only; it is not part of the C++ prediction RSS measurement.

`scripts/acceptance.sh` writes GNU time output to `results/<class>/time.txt`.
Use that file for the maximum resident set size instead of a separate per-stage
memory profile.

Run `scripts/evaluate_metrics.sh capsule` after prediction to write
`results/capsule/metrics.csv` with the same metric columns and Mean/Overall
row format as `rgb_module`.

## Notes

> 这些 notes 解释了为什么 runtime 和 verifier 分离、为什么记忆库保留 FP16、
> 为什么engine 使用 memory-mapped 方式加载以及engine 必须在 TX2 上最终构建的原因。

- The C++ runtime uses the original RGB `any = RGB or infrared or pointcloud`
  image-label rule only in the separate verifier, matching `base/RGB`.
- The memory bank remains on the GPU as FP16. Its size is not the main source
  of the original Python process RSS.
- The engine is memory-mapped during deserialization and built with a 16 MB
  TensorRT workspace cap to reduce deployment RSS without changing model inputs.
- Cloud validation exported the ONNX model and validated the matching PyTorch
  configuration. TensorRT/CUDA compilation and the final memory/latency gate
  must run on TX2.
