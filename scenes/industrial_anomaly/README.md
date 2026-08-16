# 工业异常场景（RGB + 红外）

该场景已接入 SDK `0.13.1` 的正式运行时，不再自行创建后台线程或写死云端地址。
RGB、红外视觉层负责从真实图片输出局部异常分数和热图；插件负责阈值语义、动作边界和跨模态策略；
公共框架负责插件路由、Outbox、重试、持久聚合、`partial_final/final` 和结果回填。

## 真实图像感知

真实感知实现位于 [`perception/`](perception/README.md)，不是只有手工 `score`：

```text
MulSen_AD capsule RGB/红外 PNG
  -> PIL bicubic + ImageNet normalization
  -> ViT-small/patch8 ONNX 或 Nano TensorRT FP16
  -> 384x20x20 patch feature
  -> RGB/红外独立 PatchCore memory bank
  -> image score + 160x160 float heatmap
  -> 工业 CloudEvent -> /decide -> 云端跨模态 final
```

数据使用 `orgjy314159/MulSen_AD` 的第三个压缩包 `MulSen_AD_new.zip`，只提取历史实验
对应的 `capsule`：每个模态 64 张正常训练图、58 张测试图；测试图从未进入 memory bank。
用户提供的两份 ONNX 完全相同，两份 engine 也完全相同，所以当前是“共享视觉骨干 + 两套
模态独立 memory bank”，不是两套不同权重。原 engine 的 TensorRT 序列化版本与
Nano222 不兼容，仓库内 engine 已在 Nano222/Orin 上从绑定 ONNX 重建并实测。

新鲜实测摘要在 `evidence/perception_capsule_fresh_summary.json`：

- 原图到分数：RGB/红外均值 `103.96/110.88 ms`；
- Nano TensorRT 58+58：图像 AUROC `0.940/0.833`，tensor-to-event 均值
  `53.35/34.26 ms`；
- 一组真实 RGB+红外图片进入正式 18101 后，本地为 `review/anomaly`；在当前 v4 云端
  重新回放相同视觉模型分数与模型身份后，云端收到 2/2 模态，由 ExtraTrees 输出权威
  `anomaly`，全局一致、无缺失成员；两事件平均图片到 provisional 为 `189.47 ms`。

原始数据不提交 Git；摘要以路径、字节数和 SHA-256 绑定本机逐图报告、Nano 预测热图
及 HTTP 响应。最新真实图片回放证据为
`evidence/industrial_visual_edge_cloud_replay_v4.json`。旧
`current_release_full_matrix_summary.json` 是受控 score 输入下的 180 事件弱网/冲突
矩阵，两类证据的输入口径不混写。

## 统一入口

工业与交通共用：

```text
POST /api/v1/collaboration/decide
```

框架按 CloudEvents 信封的 `scene`、`type`、`dataschema` 三项选择插件：

- `scene=industrial_anomaly`：调用 `IndustrialAnomalyPlugin`；
- `scene=freeway_traffic_management`：调用 `TrafficPlugin`。

这意味着最终演示不需要额外的工业 URL。工业数据返回工业 `normal/review/anomaly`
结果，交通数据返回交通控制结果。

## 工业闭环

```text
TensorRT RGB/红外结果
  -> /decide 严格 Schema 校验
  -> review-band 本地初判（provisional）
  -> 边缘静态 Q4 仅对 review-band 样本做 1-token 复核
  -> 公共调度器选择 edge_only / cloud_async / cloud_sync
  -> 摘要/必要热力图写入公共持久 Outbox
  -> 云端按 product + sample_id 持久等待 RGB/红外
  -> 完整双模态分数进入工业 ExtraTrees 专用协调器
  -> ExtraTrees 输出 normal/anomaly 主判与置信度
  -> 仅预期收益足够的低置信结果交给 Qwen3.5 9B 非权威复核
  -> 公共冲突协调器检查并消解跨边缘动作冲突
  -> 完整两模态为权威 final；超时缺失为非确认 partial_final
  -> results/batch 回填边缘 review 状态
```

这里严格采用“边缘感知不同，感知后的协同链路与交通一致”的边界：工业特有部分只有
RGB/红外 TensorRT 感知、工业事件适配、RGB/红外聚合成员、工业动作/冲突语义和工业
ExtraTrees 特征；进入公共 `SemanticEvent` 后，调度器、证据规划、持久 Outbox、云端
聚合、ExtraTrees 主判、按预期收益选择 9B、全局冲突协调、权威 final 与结果回填都复用
交通的同一套框架实现，不维护第二条工业专用传输或调度链路。

两场景的云端权威关系也完全一致：场景专用 ExtraTrees 是在线主判，全量 Qwen 9B 是
低频、非权威的结构化审计器。Qwen 返回 `accept` 或 `challenge` 都保留 ExtraTrees 的
decision/actions；challenge 只记录 `cloud_llm_advisory_recommendation`，用于监控、人工
审计和后续模型更新，不能在线改写 final。相同 `product/sample_id` 的两条模态决策共用
一次 Qwen 复核。Qwen 失败或输出不合法时仍保留 ExtraTrees 基线，普通业务不依赖大
模型可用性。

当前 `industrial-capsule-cross-modal-extratrees-v1` 只声明支持真实感知已经落地的
`capsule`，其他产品继续使用原确定性安全策略，不伪称跨产品泛化。模型使用配对的
Nano TensorRT RGB/红外分数：40 对拟合，18 对固定分层留出且从未参与拟合；留出集
准确率、平衡准确率和宏 F1 均为 `1.0`。原“一致才裁决”策略在相同留出集只能覆盖
`61.11%`，其余为 `review`。冻结结果见
`evidence/industrial_cloud_extratrees_capsule_v1.json`。正式 `18100` 的成对 HTTP
闭环证据见 `evidence/industrial_cloud_chain_formal_18100_v4.json`：高置信样本由
ExtraTrees 直接完成；低置信样本才选择 Qwen 9B，同一 RGB/红外组只复核一次，且实测
challenge 后 final 仍保持 ExtraTrees 的 `normal`。selected-only 9B 时延单独报告，
不用来替代全体样本平均时延。Nano222 `18101` 到正式云
`18100` 的真实 final 回填见
`evidence/industrial_edge_to_cloud_extratrees_qwen_v4.json`，其中两路 final 都明确记录
`source=industrial_cloud_extratrees_coordinator`、`cloud_llm_baseline_preserved=true`
和 `cloud_llm_review_role=advisory_non_authoritative`，并共用同一份 Qwen 复核。

固定 18 对未参与拟合的留出集也已逐对通过正式 HTTP 接口复测，结果见
`evidence/industrial_cloud_heldout_http_v2.json`：ExtraTrees 准确率 `100%`，仅
`1/18 = 5.56%` 的低置信对调用 Qwen；包含该慢路径后的 final HTTP 总体平均为
`74.92 ms`，云端运行总体平均为 `50.57 ms`。selected-only 最大值和总体 P95 仍在
证据中单独披露，不以平均值掩盖长尾。

可复现训练命令：

```bash
python scenes/industrial_anomaly/train_industrial_cloud_coordinator.py \
  --rgb-predictions /path/to/nano_full_rgb_trt_v1/predictions.csv \
  --infrared-predictions /path/to/nano_full_infrared_trt_v1/predictions.csv \
  --thresholds scenes/industrial_anomaly/industrial_anomaly/review_bands.json \
  --output-model scenes/industrial_anomaly/assets/models/industrial_cloud_extratrees_capsule_v1.joblib \
  --output-metrics /path/to/fresh_metrics.json
```

服务启动后，可复跑与正式证据同口径的两条 HTTP 链路：

```bash
python scenes/industrial_anomaly/benchmark_industrial_cloud_chain.py \
  --cloud-url http://127.0.0.1:18100 \
  --output /path/to/new-industrial-cloud-chain-evidence.json
```

固定留出集总体均值复测：

```bash
python scenes/industrial_anomaly/benchmark_industrial_cloud_heldout_http.py \
  --cloud-url http://127.0.0.1:18100 \
  --rgb-predictions /path/to/rgb/predictions.csv \
  --infrared-predictions /path/to/infrared/predictions.csv \
  --output /path/to/new-industrial-heldout-http-evidence.json
```

该脚本会 fail-closed 核对：双场景服务健康、ExtraTrees 模型身份、高置信快速路径不
调用 Qwen、低置信路径按成本门选择 Qwen、两模态共用一份复核结果，以及 Qwen
challenge 只能形成非权威审计建议、不能改写 ExtraTrees final。

真实 Nano→云端语义对齐复测：

```bash
python scenes/industrial_anomaly/benchmark_industrial_edge_cloud_alignment.py \
  --edge-url http://192.168.31.222:18101 \
  --cloud-url http://127.0.0.1:18100 \
  --output /path/to/new-industrial-edge-cloud-alignment.json
```

阈值已由原 `summary_f1_review_bands.xlsx` 转为不依赖 pandas/openpyxl 的
`industrial_anomaly/review_bands.json`，文件中保留原始 Excel SHA-256。

## 安装与双场景配置

```bash
python -m pip install -e scenes/industrial_anomaly --no-deps
```

- 边缘：`deployment/edge_service_traffic_industrial.json`
- 云端：`deployment/cloud_service_traffic_industrial.json`
- 本机接口旁路：`deployment/sidecar_edge_service_traffic_industrial.json`（19101）和
  `deployment/sidecar_cloud_service_traffic_industrial.json`（19100）。旁路使用无权重交通
  参考插件，只验证统一接口/路由/闭环；正式配置仍使用完整交通 Student/Qwen 插件。
- 工业样例：`samples/rgb_event.json`、`samples/infrared_event.json`

两个服务启动后，可用一个命令验证同一边缘入口同时路由两个场景：

```bash
python scenes/industrial_anomaly/demo_dual_scene.py \
  --edge-url http://127.0.0.1:18101
```

脚本会先后发送工业 RGB/红外和交通两个成员，打印两组本地 `provisional` 与云端
`final`。每次运行生成新的 `event_id/sample_id`，不会与 Outbox 幂等记录冲突。

当前正式版本是同一个静态融合 Q4_K_M 模型同时服务交通和工业，不加载运行时 LoRA。
两个插件在完成 Schema 校验后，各自用冻结 action codec 生成原生 16 位输入；模型只
返回一个动作 token。`llama-server` 的
`/lora-adapters` 在启动、请求和结束后都必须为 `[]`，请求体也不得携带 LoRA 字段。
这样避免双 Adapter 交替带来的调度和内存峰值，同时不改变两个场景已有的输入合同。

正式配置位于 `deployment/joint_static_raw16_formal_v2/`。该 release 绑定：

- release `traffic-industrial-joint-static-v3-q4km-raw16`；
- 静态 Q4 SHA-256 `828f839873c7505005101544d8febb7408bc0443c153c4ba97ec4b3603445526`；
- 规范编码器 `scene-native-decimal16@v1`；
- Nano 500 次严格交替门禁峰值 `998,445,056 B`、平均时延 `73.374464 ms`；
- 交通准确率 `72.8%`、工业准确率/宏 F1/加权 F1 均为 `100%`；
- 运行时 LoRA 数量为 `0`，同一常驻模型直接处理交通和工业 16-token 输入。

正式云端 `0.0.0.0:18100` 与 Nano222 边缘 `18101/18190` 已切换并托管。受控 score
输入下的工业弱网、冲突和并发矩阵摘要位于
`evidence/current_release_full_matrix_summary.json`；该文件保留其测试时 release 身份，
不冒充当前真实图片或当前云端 v4 的重跑结果。当前云端和真实边云结果以本节上方 v4
证据为准。

工业模型的输入是由插件从 `score/review_low/review_high` 提取的 16-token 相对阈值距离，
输出为单 token：`A=normal`、`B=review`、`C=anomaly`。它学习的是既有 review-band
动作策略，不代表 RGB/红外视觉模型本身的异常检测准确率。运行模式包括：

- `disabled`：只使用确定性 review-band；
- `shadow`：模型只旁路记录，不改变决策；
- `corroborate`：模型与规则一致时记录为学习模型路径，不一致、超时或运行失败时安全
  回退到确定性规则；
- `selective`：`normal/anomaly` 直接走确定性快速路径，仅 `review` 调用共享静态 Q4；
  模型调用预算固定不超过 180 ms，超时、运行失败或规则分歧时仍回退到 `review`。

因此接入边缘模型不改变工业的安全动作边界，也不改变公共 Outbox、云端跨模态协调和
final 回填流程。

## 旧版双 LoRA 交替稳定性门禁

以下内容保留用于旧多 Adapter 回归，不是当前推荐部署。若同一 `llama-server` 预载
交通/工业 LoRA，可在 Jetson 本机执行只读资源门禁。默认
严格交替发送交通、工业各 250 条，共 500 条；每 20 条读取 `/proc` 中的
`MemAvailable`、系统 swap、`pgpgin/pswpin` 以及指定 llama PID 的 RSS/VmSwap。
`MemAvailable` 低于 256 MiB 时脚本在下一条请求前停止，并原子保留部分报告：

```bash
python scenes/industrial_anomaly/benchmark_multi_adapter_stability.py \
  --traffic-runtime scenes/industrial_anomaly/deployment/edge_llm_runtime_traffic_multi_adapter_nano222.json \
  --industrial-runtime scenes/industrial_anomaly/deployment/edge_llm_runtime_industrial_nano222.json \
  --traffic-jsonl /path/to/traffic_eval.jsonl \
  --industrial-jsonl /path/to/industrial_eval.jsonl \
  --llama-pid "$(pgrep -n -x llama-server)" \
  --base-gguf /path/to/clean/base.Q6_K.gguf \
  --runtime-adapter /path/to/traffic.F32-LoRA.gguf \
  --runtime-adapter /path/to/industrial.F32-LoRA.gguf \
  --output /path/to/multi_adapter_stability_500.json
```

三项资产参数是强制证据绑定：报告会记录基座及两个按 ID 排序 LoRA 的字节数与
SHA256；缺少这些字段的旧报告不能用于构建 Q6 发布包。

如果仅因压力测试需要放宽 HTTP timeout，可传
`--benchmark-timeout-override-seconds 2.0`。覆盖只作用于当前 benchmark 进程，不会
重写 runtime 文件；原值、有效值和覆盖事实都会写进结果，不能把它解释为正式服务
timeout 已改变。生产验收应显式核对 `--llama-pid`，不要依赖模糊的进程名匹配。
