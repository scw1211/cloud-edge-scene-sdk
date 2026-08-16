# 云边协同感知与决策系统

这套系统面向交通监控和工业异常检测两类业务。边缘节点负责接收原始感知结果、完成轻量推理并在断网时维持安全动作；云端负责跨节点汇聚、全局约束检查和低频大模型复核。两类场景共享同一套事件协议、调度器、持久队列、结果回填和模型发布机制，区别只保留在感知模型、特征编码和业务动作上。

当前正式边缘版本为 `traffic-industrial-joint-static-v3-q4km-raw16`。交通和工业共用一个静态融合的 Q4_K_M 模型，不加载运行时 LoRA，也不在请求间切换 Adapter。两个场景分别把业务状态编码成原生 16 位十进制序列，模型只返回一个受约束动作 token。该设计把常驻模型、KV 缓存和调度图控制在 Nano222 可承受范围内，同时保留场景专用的感知和云端协调模型。

## 当前结果

| 项目 | 实测结果 | 口径 |
| --- | ---: | --- |
| 边缘模型 | Q4_K_M，529,289,216 B | 静态融合，运行时 LoRA 为 0 |
| Nano222 严格内存峰值 | 998,445,056 B | `max(VmHWM, 20 ms 采样的 VmRSS+VmSwap)`，不扣基线 |
| 双场景模型推理 | 73.374 ms | 交通、工业各 250 条严格交替，共 500 条 |
| TTFT | 边缘 73.218 ms，云端 311.942 ms | 同一批 80 个 16-token 输入、同为单 token 输出，下降 76.528% |
| 路由感知端到端 | 100.651 ms | 交通 400 事件与工业 180 事件按事件数加权 |
| 交通动作模型 | accuracy 67.75%，weighted-F1 68.66% | 冻结测试集 2,400 条 |
| 工业动作模型 | accuracy / macro-F1 / weighted-F1 均为 100% | 冻结测试集 960 条 |
| 工业 RGB / 红外图像 AUROC | 0.940 / 0.833 | Nano TensorRT，两个模态各 58 张测试图 |
| 弱网业务保持 | 100% | mild、severe、outage 的本地业务均成功，20/20 断网事件恢复回填 |
| 交通自然冲突 | 4.478% | 134 对关联且活跃的边缘决策，协调后残余冲突为 0 |
| 冲突解决 | 100% | 交通 48 个注入冲突和工业 30 个跨模态分歧全部解决 |

端到端时延按业务终点计算：`cloud_sync` 路由一直计到权威 `final`；`edge_only`、`local_autonomy` 和 `cloud_async` 计到安全的 `provisional` 已返回且摘要已持久化。这样既没有把异步任务强行等待云端的时间算进本地业务响应，也没有把同步任务停在临时结果。当前交通 400 个事件均值为 119.866 ms，工业 180 个事件均值为 57.952 ms，合并 580 个事件后为 100.651 ms。

项目源码位于 `cloud-edge-scene-sdk`。完整模型、TensorRT engine、PatchCore memory bank、正式配置和原始证据集中在 GitHub Release：

<https://github.com/scw1211/cloud-edge-scene-sdk/releases/tag/competition-final-v1.0.0>

## 一、公共框架

### 1. 系统分工

系统按感知、边缘决策、云端协调和反馈更新四层组织。

```text
交通检测器 / RGB相机 / 红外相机
        │
        ▼
场景感知模型：时序风险、ViT + PatchCore
        │  CloudEvents 1.0 场景事件
        ▼
边缘公共入口 :18101
  ├─ Schema 校验与场景插件路由
  ├─ 场景专用轻量模型和安全规则
  ├─ 静态 Q4 单 token 复核
  ├─ 在线调度 edge_only / local_autonomy / cloud_async / cloud_sync
  └─ Durable handoff + SQLite Outbox
        │  紧凑摘要、按需证据
        ▼
云端公共入口 :18100
  ├─ 按 sample/group 持久汇聚
  ├─ 场景专用 ExtraTrees 主判
  ├─ 全局冲突协调与约束检查
  └─ Qwen3.5 9B 低频非权威复核
        │
        ▼
authoritative final / partial_final 回填边缘
```

边缘模型不是直接处理所有原始模态。交通时序模型和工业视觉模型先把大输入压缩为结构化态势；0.8B 模型只处理短编码并输出动作。这样既降低推理时延，也避免把摄像头图像、热力图或完整交通张量持续上传到云端。

云端 Qwen3.5 9B 保留完整参数规模，没有做场景蒸馏。它不替代场景专用 ExtraTrees，也不参与每一个请求。协调器先计算是否存在足够的预期收益，只有低置信、高不确定或显式要求复核的聚合组才调用 9B。9B 的 `accept` 或 `challenge` 会成为审计信息，线上权威动作仍由场景协调器和全局约束决定。这样可以利用大模型的解释和跨域复核能力，又不会让普通业务承担其时延和算力成本。

### 2. 统一事件入口

所有场景都从同一个接口进入：

```text
POST /api/v1/collaboration/decide
```

外部请求使用 CloudEvents 1.0 结构化信封。框架只固定身份、来源、场景、事件类型、时间和 Schema URI，`data` 由场景自己定义。

```json
{
  "specversion": "1.0",
  "id": "industrial-event-0001",
  "source": "urn:edge:industrial-rgb:patchcore",
  "type": "com.example.industrial.anomaly-map.v1",
  "scene": "industrial_anomaly",
  "edgeid": "industrial-rgb",
  "time": "2026-08-16T00:00:00Z",
  "dataschema": "https://cloud-edge.local/schemas/examples/industrial-anomaly-map-v1.json",
  "datacontenttype": "application/json",
  "data": {
    "sample_id": "capsule-0001",
    "product": "capsule",
    "modality": "rgb",
    "score": 0.0076
  }
}
```

插件完成严格 Schema 校验后，把场景字段归一化为内部 `SemanticEvent`。公共调度只读取风险、置信度、候选动作、资源范围、证据级别和时限等统一语义，不猜测业务字段。`scene`、`type` 或 `dataschema` 不匹配时请求会直接失败，不会静默回退到其他场景。

主要接口如下：

| 接口 | 作用 |
| --- | --- |
| `POST /api/v1/collaboration/decide` | 边缘本地决策和路由 |
| `POST /api/v1/collaboration/cloud-decision` | 云端场景模型处理单个事件 |
| `POST /api/v1/collaboration/coordinate` | 多事件全局协调 |
| `POST /api/v1/collaboration/flush-pending` | 手动触发待上传任务重放 |
| `GET /api/v1/collaboration/plugins` | 查看已加载场景和模型状态 |
| `GET /api/v1/collaboration/schema` | 查看协议与 Schema |
| `GET /health` | 服务、队列、模型和聚合健康状态 |

### 3. 四条计算路径

调度器同时考虑网络质量、动作风险、模型不确定性、证据大小、云端排队和剩余时限，输出四类路径：

- `edge_only`：本地结果已经足够，直接完成业务。
- `local_autonomy`：网络不可用或预算不足，保留安全的本地自治结果，待网络恢复后补传摘要。
- `cloud_async`：先返回不越过安全边界的 `provisional`，摘要进入持久队列，云端结果随后回填。
- `cloud_sync`：动作必须等待云端确认，业务终点是权威 `final`。

高风险控制在 `final` 前不会获得执行授权。异步返回只是缩短响应，不是放宽安全规则。网络状态也不是由调用者随意填写：边缘服务持续探测云端可达性、RTT、失败窗口和预计传输时间，调度记录中保留当时的网络快照和选择理由。

### 4. 持久传输和结果生命周期

普通事件先写入带校验和的 durable handoff，随后进入 SQLite Outbox。后台 worker 批量上传摘要；失败任务按退避策略重试，云端以请求指纹和幂等键去重。该链路提供“本地持久接受 + 至少一次重试 + 接收端幂等”，不把它描述成跨机 exactly-once。

一个决策会经历：

```text
edge provisional
  -> queued / inflight
  -> cloud durable acceptance
  -> partial_final 或 authoritative final
  -> edge review record completed
```

云端成员未到齐时只能产生 `partial_final`，不能冒充全局确认。结果回填保留 revision，同一聚合组后来补齐成员时会得到更高 revision 的 `final`。服务重启后，Outbox、待复核记录、幂等表和聚合状态都从 SQLite 恢复。

### 5. 聚合和冲突协调

聚合键由场景插件定义。交通按同一时间窗和道路分区汇聚，工业按 `product + sample_id` 汇聚 RGB 与红外。框架等待预期成员到齐，再把完整组交给场景协调器；到期缺失则保留缺失列表和非权威状态。

冲突分两层处理：

1. 场景协调器处理业务语义分歧，例如 RGB 与红外对同一工件判断不同，或相邻交通分区给出互相冲突的控制建议。
2. 公共协调器检查共享资源、动作互斥、版本不一致和跨区约束，必要时降级为安全动作或请求复核。

自然运行中的冲突率和人工构造冲突的解决率分开统计。没有自然冲突时不会用 `0/0` 推导 100% 解决率。

### 6. 模型发布与回滚

边缘模型发布不是复制一个 GGUF 后直接启动。发布包会锁定：

- 上游文本基座和训练来源；
- 训练、验证、测试数据身份；
- 合并和量化方法；
- GGUF 字节数与 SHA-256；
- 输入编码器、动作映射和最大 token 数；
- 本机完整测试、Nano 内存与稳定性证据；
- 启动探针和场景运行时输出。

`ReleaseStore` 使用不可变 release ID 和绑定指纹。`serve-release` 启动后先核对 `/props`、模型路径、slot 数、`/lora-adapters` 和两场景单 token 探针，全部通过才生成 gate receipt。交通与工业两份 runtime 配置带同一 release revision 和模型 SHA；写入中断时由 transaction journal 恢复，避免两个场景处于不同模型代际。

当前正式模型的关键信息：

| 字段 | 值 |
| --- | --- |
| release | `traffic-industrial-joint-static-v3-q4km-raw16` |
| quantization | `Q4_K_M` |
| bytes | `529289216` |
| SHA-256 | `828f839873c7505005101544d8febb7408bc0443c153c4ba97ec4b3603445526` |
| input contract | `scene-disjoint-decimal16@v2` |
| runtime adapters | `[]` |
| llama.cpp | ctx 128，batch 16，ubatch 16，parallel 1，GPU layers 99 |

## 二、交通场景

### 1. 数据和感知

交通使用 PEMS08。一个在线样本是 `[170, 3, 12]`：170 个检测点，流量、占有率和速度三个通道，连续 12 个五分钟步，对应最近一小时。正式默认路径不等待未来真值，也不在每次请求中加载 PyTorch；常驻 NumPy 感知器直接从最近窗口计算节点和区域风险。

170 个检测点通过冻结的 METIS4 映射分成四个道路区域。每个逻辑边缘只读取自己的 42 或 43 个节点，输出区域摘要、控制能力、Top-10 风险节点和必要的速度历史。完整 `[170,3,12]` 张量不会在在线链路中发往云端。

ASTGCN 未来预测路径仍保留，用于预测类实验和消融；正式低时延链路采用当前态势感知。两者使用同一套事件协议和后续协同流程，但指标不能混写。

### 2. 边缘决策

每个交通事件依次经过：

```text
当前态势感知
  -> 区域 Student 初判
  -> defer / 风险 / 动作约束检查
  -> 验证集增益路由
  -> 必要时调用静态 Q4
  -> 安全过滤
  -> provisional 或 final
```

Student 处理大多数普通事件。静态 Q4 不是按“风险高”简单触发，而是只接管验证集证明有净纠错收益、且时限允许的可观察子群。模型输入为 16 位十进制编码，包含交通态势、Student 决策、规则决策、置信度桶、预测集歧义、网络状态和执行器能力；输出限制为 A-F 中的一个 token。

| token | 交通动作 | 是否要求云确认 |
| --- | --- | --- |
| A | 不动作 | 否 |
| B | 拥堵预警 | 否 |
| C | 可变限速 | 视约束而定 |
| D | 匝道控制 | 视约束而定 |
| E | 区域协同 | 是 |
| F | 绕行 | 是 |

动作映射之后还会经过风险等级、执行器是否存在、跨区资源和云确认要求检查。模型即使输出了不合适的动作，也不能越过安全过滤器。

### 3. 云端协调

同一时间窗的四个区域事件在云端聚合。当前态势专用 ExtraTrees 根据四区特征给出主判，再结合道路拓扑检查边界限速、匝道变化率、替代走廊负荷和策略版本。Qwen3.5 9B 只对预期收益足够的少量结果做结构化复核；普通交通事件不会为了“使用大模型”而强制调用 9B。

交通自然冲突实验覆盖 45 个聚合组、270 个区域对，其中 134 对同时活跃且存在道路边界耦合。协调前检测到 6 对冲突，关联活跃对冲突率为 4.478%，全部在一轮全局协调后消失。另用真实 PEMS08 边界事件构造 48 个压力样本，覆盖边界限速不连续、匝道变化率不连续、替代走廊过载和策略版本不一致，48/48 均检测并解决。

### 4. 交通实测

| 测试 | 结果 |
| --- | --- |
| 静态 Q4 冻结集 | 2,400 条，accuracy 67.75%，weighted-F1 68.66%，合法输出 100% |
| Nano 严格交替中的交通子集 | 250 条，accuracy 70.4%，weighted-F1 71.19% |
| 当前态势云 ExtraTrees | 测试准确率 73.73% |
| 正式连续窗 | 100 个窗口、400 个区域事件，业务完成均值 119.866 ms |
| 自然冲突 | 4.478%，协调后 0 |
| 注入冲突 | 48/48 解决 |

400 个事件中，383 个按本地安全结果和持久摘要完成，17 个 `cloud_sync` 计到权威 final。窗口级四分区全部完成均值高于单事件均值，因此报告中同时保留 event 和 window 两种口径，竞赛的双场景总平均使用 event 加权口径。

### 5. 交通复现

先安装轻量依赖和场景包：

```bash
python -m pip install -r requirements.txt
python -m pip install -e scenes/freeway_traffic --no-deps
```

准备冻结的四分区数据：

```bash
python scenes/freeway_traffic/prepare_metis_partition_data.py
```

运行连续 100 窗口端到端基准：

```bash
python scenes/freeway_traffic/benchmark_real_current_state_e2e.py \
  --project-root . \
  --data-npz scenes/freeway_traffic/assets/downloads/PEMS08_r1_d0_w0_astcgn_multitask.npz \
  --edge-url http://127.0.0.1:18101 \
  --cloud-url http://127.0.0.1:18100 \
  --sample-start 100 \
  --sample-stop 200 \
  --warmup-samples 86,125,0 \
  --aggregation-timeout-ms 150 \
  --top-k 10 \
  --require-qwen-selected \
  --require-qwen-accepted \
  --require-congestion-level-coverage \
  --require-complete-final \
  --output /tmp/traffic-100.json
```

四个独立逻辑边缘进程可用下列命令启动；正式部署时把云地址换成实际地址：

```bash
python scenes/freeway_traffic/run_partitioned_current_state_edges.py \
  --project-root . \
  --manifest scenes/freeway_traffic/runtime/pems08_metis4_partitions/manifest.json \
  --launch-isolated-edge-services \
  --edge-port-base 19101 \
  --cloud-url http://192.168.31.135:18100 \
  --sample-start 100 \
  --sample-stop 110 \
  --output /tmp/traffic-four-edge.json
```

## 三、工业场景

### 1. 数据集和视觉模型

工业数据来自 Hugging Face 数据集 `orgjy314159/MulSen_AD` 的第三个压缩包 `MulSen_AD_new.zip`。当前正式验证产品为 `capsule`，使用 RGB 和 Infrared 两种模态。每个模态取 64 张正常图建立 memory bank，58 张测试图只用于评估，训练和测试严格分离。原始图片不进入 Git，子集清单记录来源 revision、文件大小和 SHA-256。

视觉链路如下：

```text
RGB / Infrared PNG
  -> PIL bicubic resize + ImageNet normalization
  -> ViT-small/patch8, 160×160
  -> 384×20×20 patch feature
  -> 模态独立 PatchCore memory bank
  -> image score + 160×160 heatmap
  -> 工业 CloudEvent
```

RGB 和红外共享同一个视觉骨干，但使用不同 memory bank：RGB 1,280 行，红外 512 行。PC 上使用 ONNX Runtime；Nano222 上使用从同一 ONNX 在目标机重建的 TensorRT 8.5.2 FP16 engine，PatchCore 距离计算由 cuBLAS 完成。原先外部提供的 engine 与 Nano 的 TensorRT 序列化版本不兼容，因此没有直接复用二进制 engine。

视觉模型实测：

| 模态 | 图像 AUROC | 像素 AUROC | 原图到分数均值 | Nano tensor-to-event 均值 |
| --- | ---: | ---: | ---: | ---: |
| RGB | 0.940 | 0.996 | 103.965 ms | 53.350 ms |
| Infrared | 0.833 | 0.952 | 110.880 ms | 34.258 ms |

ONNX 的原图口径包含图片读取、缩放归一化、特征提取、PatchCore 打分和热力图构造；Nano 的 tensor-to-event 口径从冻结 NCHW 输入开始。两者分别报告，不把预处理时间从一个口径移到另一个口径。

### 2. 边缘复核

视觉模型先根据产品和模态对应的 `review_low/review_high` 形成三态初判：

- 分数低于下界：`normal`；
- 分数位于区间：`review`；
- 分数高于上界：`anomaly`。

确定的 `normal` 和 `anomaly` 走快速路径。只有落在 review band 的样本才把相对阈值距离、模态和产品编码成 16 位序列，交给同一静态 Q4 复核。工业动作 token 为 `A=normal`、`B=review`、`C=anomaly`。Q4 超时、输出不合法或与安全规则冲突时，结果保持 `review`，不会把不确定样本自动放行为 normal。

“边缘复核”解决的是阈值附近的决策稳定性，不是重新做一次视觉特征提取。ViT + PatchCore 已经完成图像感知，Q4 只读取紧凑的态势编码。冻结 960 条工业动作测试中，accuracy、macro-F1、weighted-F1 和合法输出率均为 100%。

### 3. 云端跨模态协调

RGB 和红外事件以 `product + sample_id` 为聚合键。云端收到完整两模态后，把两侧分数、阈值位置、局部状态、模型身份和证据完整性编码为 15 维特征，交给工业专用 ExtraTrees。交通与工业分别维护自己的 ExtraTrees，因为两个场景的特征和目标完全不同；它们共享训练、加载、调度、审计和发布框架，不强行合并成一棵模型。

云端主流程是：

```text
RGB + Infrared 完整组
  -> 工业 ExtraTrees 主判 normal / anomaly
  -> 置信度和预期收益计算
  -> 少量低置信组调用 Qwen3.5 9B
  -> 公共冲突协调器检查
  -> authoritative final
  -> 同一 final 回填 RGB 与 Infrared 两条边缘记录
```

ExtraTrees 是线上权威主判。Qwen3.5 9B 是成本门控制的非权威审计器：它的 `challenge` 会记录建议和理由，但不会在线覆盖 ExtraTrees 的 final。这样保证工业动作可重复、可回归，9B 不可用时也不会阻断正常生产链路。

模型使用 40 对样本拟合，18 对固定分层留出。留出集通过正式 HTTP 接口逐对复测，ExtraTrees 18/18 正确；只有 1/18 的低置信组调用 9B，选择率 5.56%。包含这一次慢复核后，18 对 final HTTP 总体均值为 74.919 ms，云端运行总体均值为 50.574 ms。9B selected-only 的长时延单独保留，不能替代总体均值。

一组真实 `broken_inside/0` RGB 和红外图片也完成了在线闭环：本地分别得到 `review` 和 `anomaly`，两事件图片到 provisional 的平均核算时间为 189.471 ms；云端收到 2/2 模态，ExtraTrees 输出 `anomaly`，两条 final 一致，缺失成员和残余冲突均为 0。

### 4. 弱网、通信和冲突

工业完整矩阵覆盖 normal、mild、severe、outage 和并发 1/2/4/8。mild、severe、outage 期间本地业务成功率和动作准确率均为 100%；断网时 20 个事件先保持安全本地结果，网络恢复后 20/20 得到权威 final。8 组并发时本地成功率仍为 100%，没有连接重置和服务重启。

边缘默认上传紧凑摘要，不上传 160×160 float32 热力图。原热力图为 102,400 B，选定云端载荷平均 3,842.53 B，通信量减少 96.25%。只有证据策略明确要求时才上传热力图，并校验大小与 SHA-256。

工业冲突测试构造 30 组 RGB/红外语义分歧，共 60 个事件。30 组都到达权威 final，全部降为安全 `review`，残余语义冲突为 0，解决率 100%。这组阳性测试与自然数据中的一致样本分开统计。

### 5. 工业复现

安装工业依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -e scenes/industrial_anomaly --no-deps
python -m pip install -r scenes/industrial_anomaly/perception/requirements.txt
```

只下载 `capsule` 的 RGB 和红外数据：

```bash
python scenes/industrial_anomaly/perception/download_mulsen_subset.py \
  --product capsule \
  --output /path/to/mulsen-capsule
```

评估 RGB PatchCore；红外只需替换 `--modality` 和 memory bank：

```bash
python scenes/industrial_anomaly/perception/patchcore_onnx.py evaluate \
  --onnx scenes/industrial_anomaly/perception/assets/vit_small_patch8_160.onnx \
  --bank scenes/industrial_anomaly/perception/assets/capsule_rgb.pcbank \
  --product-root /path/to/mulsen-capsule/MulSen_AD/capsule \
  --modality rgb \
  --review-bands scenes/industrial_anomaly/industrial_anomaly/review_bands.json \
  --output /tmp/capsule-rgb-eval.json
```

从一张真实图片生成可直接提交给边缘服务的事件：

```bash
python scenes/industrial_anomaly/perception/patchcore_onnx.py infer-event \
  --onnx scenes/industrial_anomaly/perception/assets/vit_small_patch8_160.onnx \
  --bank scenes/industrial_anomaly/perception/assets/capsule_rgb.pcbank \
  --image /path/to/capsule/RGB/test/broken_inside/0.png \
  --product capsule \
  --modality rgb \
  --sample-id broken-inside-0 \
  --heatmap /tmp/rgb-heatmap.f32 \
  --output /tmp/rgb-event.json
```

复测云端 ExtraTrees 快速路径和 9B 选择路径：

```bash
python scenes/industrial_anomaly/benchmark_industrial_cloud_chain.py \
  --cloud-url http://127.0.0.1:18100 \
  --output /tmp/industrial-cloud-chain.json
```

复测 18 对固定留出集：

```bash
python scenes/industrial_anomaly/benchmark_industrial_cloud_heldout_http.py \
  --cloud-url http://127.0.0.1:18100 \
  --rgb-predictions /path/to/rgb/predictions.csv \
  --infrared-predictions /path/to/infrared/predictions.csv \
  --output /tmp/industrial-heldout-http.json
```

## 四、部署

### 1. 目录

```text
cloud_edge_framework/       公共事件、调度、Outbox、聚合、冲突与 HTTP 服务
edge_llm_factory/           蒸馏、训练、合并、量化、发布和启动门禁
schemas/                    公共信封、语义事件、决策和运行时 Schema
scene_plugin_template/      新场景插件模板
scene_adapter_template/     新场景动作模型模板
scenes/freeway_traffic/     PEMS08 交通感知、决策、云端协调和验收
scenes/industrial_anomaly/  RGB/红外感知、工业插件、ExtraTrees 和验收
deployment/framework/       公共插件加载配置
model_bundle/               云端 9B 和边缘模型安装入口
tests/                      公共框架与发布门禁测试
MANIFEST.json               Git 文件字节数和 SHA-256 清单
```

### 2. 运行环境

- Python 3.8 及以上；
- 云端：Linux、Ollama、`qwen3.5:9b`；
- 边缘：NVIDIA Jetson Orin、CUDA 11.4、TensorRT 8.5.2、CUDA 版 llama.cpp；
- 通用 Python 依赖：`requirements.txt`；
- 模型训练依赖：`requirements-training.txt`。

安装基础环境：

```bash
python -m pip install -r requirements.txt
python -m pip install -e scenes/freeway_traffic --no-deps
python -m pip install -e scenes/industrial_anomaly --no-deps
```

### 3. 云端正式配置

云端同时加载交通和工业插件，监听 `0.0.0.0:18100`：

```bash
python scenes/freeway_traffic/deploy_node.py check \
  --role cloud \
  --device cuda

python scenes/freeway_traffic/deploy_node.py run \
  --role cloud \
  --service-config scenes/industrial_anomaly/deployment/cloud_service_traffic_industrial.json \
  --device cuda
```

云配置使用：

- 交通 ExtraTrees、拓扑和全局效用配置；
- 工业 `industrial_cloud_extratrees_capsule_v1.joblib`；
- 本机 Ollama `qwen3.5:9b`；
- 独立的幂等库、聚合库和证据目录。

### 4. Nano222 正式配置

正式 systemd 配置位于：

```text
scenes/industrial_anomaly/deployment/joint_static_raw16_formal_v2/
```

其中：

- `edge_service.json`：18101、云地址、Outbox 和回填参数；
- `scene_plugins_edge.json`：交通和工业边缘插件；
- `runtime_output_traffic.json` / `runtime_output_industrial.json`：两场景原子 runtime 输出；
- `startup_probes.json`：两场景 16→1 启动探针；
- `cloud-edge-edge-18101-joint-static-q4-raw16-v2.service`：正式 systemd unit。

正式启动命令的核心参数为：

```bash
python scenes/freeway_traffic/deploy_node.py run \
  --role edge \
  --cloud-url http://192.168.31.135:18100 \
  --service-config scenes/industrial_anomaly/deployment/joint_static_raw16_formal_v2/edge_service.json \
  --llama-registry runtime/traffic_industrial_joint_static_q4km_raw16_formal_v2/edge_llm_release_store.json \
  --llama-binary /path/to/llama-server \
  --llama-port 18190 \
  --context-tokens 128 \
  --threads 4 \
  --llama-batch-size 16 \
  --llama-ubatch-size 16 \
  --parallel 1 \
  --gpu-layers 99 \
  --device cuda \
  --llama-runtime-output scenes/industrial_anomaly/deployment/joint_static_raw16_formal_v2/runtime_output_traffic.json \
  --llama-runtime-output scenes/industrial_anomaly/deployment/joint_static_raw16_formal_v2/runtime_output_industrial.json \
  --llama-startup-probes scenes/industrial_anomaly/deployment/joint_static_raw16_formal_v2/startup_probes.json \
  --startup-timeout-seconds 180
```

启动后至少检查：

```bash
curl -fsS http://127.0.0.1:18190/health
curl -fsS http://127.0.0.1:18190/props
curl -fsS http://127.0.0.1:18190/lora-adapters
curl -fsS http://127.0.0.1:18101/health
curl -fsS http://127.0.0.1:18100/health
```

`/lora-adapters` 必须返回 `[]`，18101 和 18100 的 `scenes` 必须同时包含 `traffic` 与 `industrial_anomaly`，Outbox、reconciliation、durable pending 和云端 waiting/inflight 在测试结束后应回到 0。

## 五、评测口径与赛题对应

### 1. 内存和稳定性

Nano222 上用新鲜候选进程运行交通、工业各 250 条严格交替请求。采样器每 20 ms 读取进程 `VmRSS/VmHWM/VmSwap`、系统 `MemAvailable`、OOM、swap-in 和 D 状态。严格峰值为 998,445,056 B，低于十进制 1,500,000,000 B，余量 501,554,944 B；request 100 到 500 的 RSS+swap 只增长 299,008 B，OOM 增量为 0，推理阶段没有 D 状态样本。

该证据是高频观测门，不是独立 cgroup 的硬限额。报告中保留这一测量边界，但不使用基线扣除或只报稳态 RSS 来压低数字。

### 2. TTFT

TTFT 使用同一台 Nano 客户端、同一批 80 个短输入、同一单 token 输出协议。边缘为正式静态 Q4，云端为完整 9.7B 参数 Qwen3.5 的 Q4_K_M Ollama 部署版本。两端各预热 5 次：

| 模型 | 平均 TTFT |
| --- | ---: |
| 边缘静态 Q4 | 73.218 ms |
| 云端 Qwen3.5 9.7B | 311.942 ms |
| 降幅 | 76.528% |

这项比较衡量相同短协议下的首 token 响应差异，不把网络断开时间、长 JSON 输入或多 token 解释生成混进 TTFT。

### 3. 双场景端到端

路由感知平均值按 580 个事件计算：

```text
(400 × 119.865930 + 180 × 57.951630) / 580 = 100.651147 ms
```

交通中 `cloud_sync` 事件计到权威 final，其他路由计到安全 provisional 和 durable summary；工业这 180 个受控 score 事件均走异步安全结果。真实视觉图片另做 2 事件闭环，平均 189.471 ms，不用 2 条烟测替代 180 条压力矩阵。

### 4. 网络波动和通信效率

工业网络矩阵中 mild、severe 和 outage 的本地业务保持率均为 100%，断网后的权威结果恢复率为 100%。紧凑摘要相对原始 float32 热力图减少 96.25% 上传量。交通同样使用持久 Outbox、可重试摘要和 final 回填；两场景复用同一故障恢复实现。

### 5. 一致性

交通关联活跃决策的自然冲突率为 4.478%，低于 5%；协调后残余冲突率为 0。交通注入压力集 48/48 解决，工业跨模态分歧 30/30 解决，均高于 90%。

### 6. 仍需如实说明的边界

- 当前静态 Q4 是面向交通和工业动作的任务模型，已经完成两场景冻结测试；数学、代码、自然语言推理的 80%—90% 通用能力保持尚无该正式版本的盲测证据，不能用旧模型或开发集替代。
- 工业 180 事件弱网矩阵从冻结 score 开始，视觉模型执行为 false；真实图像能力由 58+58 张感知评测和 2 事件在线闭环分别证明。
- Nano 的 TensorRT 时延从冻结 NCHW tensor 开始；PC 的 ONNX 时延包含原图预处理。两个口径不能直接相减。
- 云端 9B 是选择性审计器，不是 ExtraTrees 的替代品，也不在线覆盖权威 final。
- 正式交通部署可在一块 Nano 上模拟四个独立逻辑边缘进程；这不等于四块物理边缘板卡。

## 六、验证与交付

公共测试：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

交通测试：

```bash
python -m unittest discover -s scenes/freeway_traffic/tests -p 'test_*.py'
```

工业测试：

```bash
python -m unittest discover -s tests -p 'test_industrial*.py'
```

检查所有 Git 文件的字节数和 SHA-256：

```bash
python scripts/build_manifest.py
git diff -- MANIFEST.json
```

发布包内包含源码、Q4 GGUF、工业 ONNX/TensorRT/PatchCore 资产、交通模型、云端 ExtraTrees、正式配置和原始测试证据。原始 MulSen_AD 图片和 PEMS08 大数组不重复打包，下载器会按固定来源、字节数和 SHA-256 获取并验证。GitHub Release 只保留一个完整压缩包，避免代码包、模型包和证据包版本错位。

## 七、核心文件

| 路径 | 内容 |
| --- | --- |
| `cloud_edge_framework/server.py` | 公共 HTTP 协议和服务入口 |
| `cloud_edge_framework/scheduling.py` | 网络、风险、时限路由 |
| `cloud_edge_framework/reliable_transport.py` | durable handoff、Outbox 和重试 |
| `cloud_edge_framework/aggregation.py` | 多成员持久聚合与 finality |
| `cloud_edge_framework/conflicts.py` | 公共冲突协调 |
| `edge_llm_factory/serve_release.py` | llama-server 启动、探针和回滚 |
| `edge_llm_factory/release_store.py` | 不可变发布记录和 active revision |
| `scenes/freeway_traffic/freeway_traffic_full/plugin_impl.py` | 交通边缘和云端插件 |
| `scenes/freeway_traffic/traffic_system/current_state_perception_runtime.py` | 当前态势感知 |
| `scenes/freeway_traffic/traffic_system/conflict_coordinator.py` | 交通图约束协调 |
| `scenes/industrial_anomaly/perception/patchcore_onnx.py` | 工业原图、PatchCore 和事件生成 |
| `scenes/industrial_anomaly/industrial_anomaly/plugin.py` | 工业边缘/云端插件 |
| `scenes/industrial_anomaly/industrial_anomaly/cloud_coordinator.py` | 工业 ExtraTrees 与 9B 选择性复核 |
| `scenes/industrial_anomaly/train_industrial_cloud_coordinator.py` | 工业云端模型训练 |
| `scenes/industrial_anomaly/evidence/` | 工业视觉、云端和边云闭环摘要 |
| `scenes/industrial_anomaly/deployment/joint_static_raw16_formal_v2/` | 当前 Nano 正式配置 |

系统的边界很明确：场景感知解决“看到了什么”，边缘模型解决“现在能做什么”，云端协调解决“多个节点一起做会不会冲突”，持久传输和发布机制保证网络波动、服务重启和模型更新时仍能追溯到同一个事件、同一个模型和同一份权威结果。
