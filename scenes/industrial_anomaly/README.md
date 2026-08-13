# 工业异常场景（RGB + 红外）

该场景已接入 SDK `0.13.1` 的正式运行时，不再自行创建后台线程或写死云端地址。
RGB、红外模型仍负责输出局部异常分数；插件负责阈值语义、动作边界和跨模态策略；
公共框架负责插件路由、Outbox、重试、持久聚合、`partial_final/final` 和结果回填。

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
  -> 摘要/必要热力图写入公共 Outbox
  -> 云端按 product + sample_id 持久等待 RGB/红外
  -> 一致正常=normal，一致异常=anomaly，其余=review
  -> 完整两模态为权威 final；超时缺失为非确认 partial_final
  -> results/batch 回填边缘 review 状态
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

工业边缘动作模型使用与交通相同的 Qwen3.5-0.8B 文本基座，但使用独立 LoRA。服务
启动时可同时预载交通和工业 Adapter，框架依据已经完成 Schema 校验的场景信封，在
请求中显式选择对应 Adapter；模型不负责猜场景，也不会把两套权重混合起来。

工业模型的输入是由插件从 `score/review_low/review_high` 提取的 16-token 相对阈值距离，
输出为单 token：`A=normal`、`B=review`、`C=anomaly`。它学习的是既有 review-band
动作策略，不代表 RGB/红外视觉模型本身的异常检测准确率。运行模式包括：

- `disabled`：只使用确定性 review-band；
- `shadow`：模型只旁路记录，不改变决策；
- `corroborate`：模型与规则一致时记录为学习模型路径，不一致、超时或运行失败时安全
  回退到确定性规则；
- `selective`：`normal/anomaly` 直接走确定性快速路径，仅 `review` 调用工业 LoRA；
  模型调用预算固定不超过 180 ms，超时、运行失败或规则分歧时仍回退到 `review`。

因此接入 LoRA 不改变工业的安全动作边界，也不改变公共 Outbox、云端跨模态协调和
final 回填流程。

## 双 LoRA 交替稳定性门禁

同一 `llama-server` 预载交通/工业 LoRA 后，可在 Jetson 本机执行只读资源门禁。默认
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
