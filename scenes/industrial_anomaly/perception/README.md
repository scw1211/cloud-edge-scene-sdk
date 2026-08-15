# 工业 RGB/红外真实感知

这里是 `industrial_anomaly` 场景的图像感知层。完整链路为：

```text
MulSen-AD PNG
  -> ImageNet/PIL 160x160 预处理
  -> ViT-small/patch8 ONNX 或目标机 TensorRT engine
  -> 1x384x20x20 patch_features
  -> 按模态独立 PatchCore memory bank 最近邻
  -> image score + 160x160 heatmap + 实测 latency
  -> industrial_anomaly 统一 /decide 接口
  -> 静态 Q5/规则复核 + RGB/红外云端汇聚
```

这与原来只输入受控 `score` 的闭环基准不同：这里会真正解码图片、执行视觉骨干并计算
PatchCore 分数。测试集不会进入 memory bank 构建。

## 数据

数据源是 Hugging Face `orgjy314159/MulSen_AD` 的第三个压缩包
`MulSen_AD_new.zip`。项目历史结果使用 `capsule`，因此当前只提取该产品的 RGB 与
Infrared，共 64+64 张正常训练图和 58+58 张测试图，不提交原始图片到 Git。

```bash
python scenes/industrial_anomaly/perception/download_mulsen_subset.py \
  --product capsule \
  --output /opt/industrial_data/mulsen_ad_new_capsule
```

## 资产与 memory bank

`assets/vit_small_patch8_160.onnx` 是可移植源模型；用户提供的两个 ONNX 经 SHA-256
核对完全相同，所以 RGB/红外当前共用同一 DINO 特征骨干，并各自构建独立 memory
bank。两个用户上传的 `.engine` 也完全相同，但其序列化版本 205 无法被 Nano222 的
TensorRT 8.5.2（版本 232）加载，因此没有提交这份不可部署二进制。仓库中的
`vit_small_patch8_160_fp16.engine` 是从同一 ONNX 在 Nano222/Orin 原机构建并完成
反序列化与推理验证的版本。

```bash
PYTHON=.venv-cloud/bin/python
ROOT=/opt/industrial_data/mulsen_ad_new_capsule/MulSen_AD/capsule
ASSETS=scenes/industrial_anomaly/perception/assets

$PYTHON -m scenes.industrial_anomaly.perception.patchcore_onnx build-bank \
  --onnx "$ASSETS/vit_small_patch8_160.onnx" --product-root "$ROOT" \
  --modality rgb --coreset-fraction 0.05 \
  --output "$ASSETS/capsule_rgb.pcbank" \
  --manifest "$ASSETS/capsule_rgb.bank.json"

$PYTHON -m scenes.industrial_anomaly.perception.patchcore_onnx build-bank \
  --onnx "$ASSETS/vit_small_patch8_160.onnx" --product-root "$ROOT" \
  --modality infrared --coreset-fraction 0.02 \
  --output "$ASSETS/capsule_infrared.pcbank" \
  --manifest "$ASSETS/capsule_infrared.bank.json"
```

两个 coreset 比例复刻现有生产运行时的冻结配方（RGB `fc0.05`、红外 `fc0.02`，
Sparse Random Projection `eps=0.9`、seed 42）。输出 `.pcbank` 与
`scenes/jetson_{rgb,infra}_trt_runtime_cublas` 的 C++ 结构逐字节兼容。

## 新鲜评测

```bash
$PYTHON -m scenes.industrial_anomaly.perception.patchcore_onnx evaluate \
  --onnx "$ASSETS/vit_small_patch8_160.onnx" \
  --bank "$ASSETS/capsule_rgb.pcbank" --product-root "$ROOT" \
  --modality rgb --output /tmp/capsule_rgb_perception.json

$PYTHON -m scenes.industrial_anomaly.perception.patchcore_onnx evaluate \
  --onnx "$ASSETS/vit_small_patch8_160.onnx" \
  --bank "$ASSETS/capsule_infrared.pcbank" --product-root "$ROOT" \
  --modality infrared --output /tmp/capsule_infrared_perception.json
```

报告同时给图像级/像素级 AUROC、AP、最大 F1，以及预处理、ONNX、PatchCore 和完整
图片到分数的延迟。`oracle_threshold` 只用于报告最大 F1，不会回写生产阈值。

## 生成可投递工业事件

`infer-event` 会把真实图片、模型和模态 bank 绑定到同一 CloudEvent，并记录热图
SHA-256 与预处理/推理实测时延。若生产器与边缘服务不在同一主机，必须同时传
`--heatmap-uri`，让事件引用边缘服务实际可读的已同步文件；否则 Outbox 会正确地
fail-closed，而不会静默丢失热图。

```bash
$PYTHON -m scenes.industrial_anomaly.perception.patchcore_onnx infer-event \
  --onnx "$ASSETS/vit_small_patch8_160.onnx" \
  --bank "$ASSETS/capsule_rgb.pcbank" \
  --image "$ROOT/RGB/test/broken_inside/0.png" \
  --product capsule --modality rgb \
  --sample-id capsule_broken_inside_0 \
  --heatmap /tmp/rgb_heatmap.f32 \
  --heatmap-uri file:///edge-visible/evidence/rgb_heatmap.f32 \
  --output /tmp/rgb_event.json
```

先用 `.part`、字节数和 SHA-256 把热图原子同步到 `--heatmap-uri` 对应路径，再把
`{"event": <CloudEvent>}` POST 到统一 `/api/v1/collaboration/decide`。RGB 与红外的
`sample_id` 必须相同，云端才会把它们放入同一跨模态聚合组。
