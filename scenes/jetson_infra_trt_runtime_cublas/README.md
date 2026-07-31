# Jetson Infra TensorRT Runtime cuBLAS

独立的 Jetson TX2 红外单模态 PatchCore TensorRT/CUDA/C++ 部署包，PatchCore 最近邻搜索使用 cuBLAS GEMM 后端。

核心配置：

```text
modality: Infrared
backbone: vit_small_patch8_224_dino
checkpoint: base/jetson_rgb_runtime/checkpoints/vit_small_patch8_224_dino.pth
img_size: 160
f_coreset: 0.02
coreset_eps: 0.9
fp16: enabled
blur_radius: 4.0
patchcore_backend: cuBLAS_GEMM
```

运行链路：

```text
Infrared PNG or prepared .nchw.f32
-> TensorRT FP16 ViT-small/patch8 engine at 160 x 160
-> cuBLAS GEMM PatchCore memory-bank distance search
-> C++ image score
-> f32/ raw 160 x 160 score maps
-> json/ minimal event JSON
-> latency.csv
```

Baseline 使用：

```text
baseline/Infra
```

门禁：

```text
Image_ROCAUC        > 85% of baseline/Infra
Infra_Pixel_ROCAUC > 80% of baseline/Infra
Infra_Pixel_F1     > 80% of baseline/Infra
RSS                 <= 1 GB
end-to-end latency  < 100 ms
```

## TX2 部署顺序

```bash
cd ~/deploy/jetson_infra_trt_runtime

export DATASET_PATH=/home/jetsontx2_3/deploy/jetson_runtime/datasets/MulSen_AD
export TRTEXEC=/usr/src/tensorrt/bin/trtexec

bash scripts/prepare_dataset.sh capsule
bash scripts/prepare_dataset.sh screw
bash scripts/build_engine.sh
bash scripts/build_runtime.sh
bash scripts/run_predict_preprocessed.sh capsule
bash scripts/evaluate_metrics.sh capsule
```

默认脚本使用 `TRT_WORKSPACE=1024` MB；如果 TX2 上仍提示 tactic workspace 不足，可以重试：

```bash
TRT_WORKSPACE=2048 bash scripts/build_engine.sh
```

如果只更新 C++ 代码，通常只需要：

```bash
bash scripts/build_runtime.sh
bash scripts/run_predict_preprocessed.sh capsule
```

## 输出结构

每次运行会按日期时间创建独立目录：

```text
results/capsule/20260727_153000/
  predictions.csv
  latency.csv
  f32/
    map_0.f32
  json/
    0.012345_event_000000.json
```

JSON 只保留一个时间字段：

```json
{
  "data": {
    "sample_id": "sample_000000",
    "modality": "infra",
    "score": 0.012345,
    "raw_uri": "file:///.../Infrared/test/good/0.png",
    "heatmap_uri": "file:///.../f32/map_0.f32",
    "inference_ms": 123.4
  }
}
```

`latency.csv` 保留拆分时延：

```text
input_h2d_ms,trt_ms,patchcore_cuda_ms,d2h_ms,score_ms,inference_ms,end_to_end_ms,cold_start_to_result_ms
```

## 资产

离线包已包含：

```text
assets/vit_small_patch8_160.onnx
assets/capsule.pcbank
assets/screw.pcbank
```

本地 PyTorch 参考精度：

```text
Capsule Image_ROCAUC=0.840, Infra_Pixel_ROCAUC=0.945, Infra_Pixel_F1=0.233
Screw   Image_ROCAUC=0.974, Infra_Pixel_ROCAUC=0.996, Infra_Pixel_F1=0.346
```

默认运行脚本使用 f_coreset=0.02 memory bank：

```text
assets/<class>.pcbank
```

如果要临时指定其他 bank：

```bash
BANK_PATH="$PWD/assets/capsule.pcbank" bash scripts/run_predict_preprocessed.sh capsule
```

## 重新生成资产

正常离线部署不需要运行。只有需要重新导出 ONNX 或转换 `.pt` memory bank 时：

```bash
PYTHON_BIN=python bash scripts/prepare_assets.sh
```
