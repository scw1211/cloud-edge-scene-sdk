# 云边协同感知与决策系统

这套系统面向交通监控和工业异常检测两类业务。边缘节点负责接收原始感知结果、完成轻量推理并在断网时维持安全动作；云端负责跨节点汇聚和全局约束检查。提交源码已将9B改为低频异步辅助审计，尚未在5070Ti与Orin正式拓扑重新部署验证。两类场景共享同一套事件协议、调度器、持久队列、结果回填和模型发布机制，区别只保留在感知模型、特征编码和业务动作上。

正式提交拓扑固定为 RTX 5070 Ti 云端与 Jetson Orin Nano 边缘。交通计分配置让冻结的 Q4_K_M 对每个节点事件给出建议，再由确定性规则做执行授权和安全回退；该配置没有切换在线生产服务。工业模型、算法、正式阈值、策略与路由已经 `DONE / FROZEN`，受控阈值候选只安装在隔离离线演示注册表。模型身份、运行时和证据 SHA 不再由 README 中的历史名称推断，统一以最终评分矩阵及其证据索引为准。

## 当前提交入口

最终成绩只从 [`results/final_submission_v1/final_scoring_matrix.json`](results/final_submission_v1/final_scoring_matrix.json) 进入报告和界面。矩阵中的状态只允许 `PASS / PARTIAL / FAIL / N/A`，每项同时绑定测量值、基线、证据路径和限制。其他日期版汇报、开发门禁和实验目录只用于追溯，不得覆盖最终矩阵。

实验正文统一使用[云边协同感知与决策实验报告](results/final_submission_v1/contest_experiment_report.md)。报告先单列大模型对照，再按评分项4.1至7.2逐项写实验目的、设计、指标、结果、结论和证据；工业与交通放在同一报告中。简表见[赛题要求与测试结果逐项对照](results/final_submission_v1/contest_requirement_test_report.md)，机器可读版为[`contest_requirement_test_summary.json`](results/final_submission_v1/contest_requirement_test_summary.json)。当前15项按赛题原文全部PASS，工程边界和瓶颈在同一报告中单列。

| 当前提交项 | 冻结状态 | 可直接展示的证据 |
| --- | --- | --- |
| 工业双模态感知与决策 | `PASS` | RGB/红外Image AUROC 90.27%/93.03%；多目标阈值开发实验硬冲突73/443→18/443、Accuracy 91.20%→94.36%、Macro-F1 86.88%→91.46%，状态`DEVELOPMENT_DEMO_ONLY` |
| 工业边缘推理 | `PASS` | Orin RGB / 红外均值 34.06 / 31.65 ms |
| 工业通信效率 | `PASS` | 同一 886 事件配对中，应用请求体减少 15.81%，捕获双向 L2 总量减少 15.46% |
| 交通通信效率 | `PASS` | 同一100窗中，summary-first加按需证据回拉相对四节点完整主分区raw窗减少21.464565%；五类业务结果均400/400一致。通信量以完整raw为基线，业务语义以冻结正式路径A（11,113,184 B）为参考 |
| 工业弱网可靠性 | `PASS` | 五档 500/500 本地业务成功，263/263 恢复补传，丢失 0、重启 0 |
| 工业一致性 | `PASS` | 硬冲突18/443（4.063205%）；工业30/30与交通48/48受控挑战共78/78解决，满足≤5%与≥90%门；自然18组逐件回放未保存 |
| 双场景含感知业务 E2E | `PASS` | 900 条加权均值 101.497 ms，831/900（92.3333%）不超过 0.2 s；交通 / 工业场景 P95 分别为 232.502 / 62.046 ms，不伪造跨分布总体 P95 |
| 交通专用云边能力保持率 | `PASS` | 正式边0.8B Q4/云端交通ExtraTrees：Accuracy/Macro-F1/Weighted-F1保持率91.52%/95.96%/95.29%，三门均≥80% |
| 框架创新一 | `PASS` | 模型绑定的紧凑语义编码与单token推理：raw16将平均输入167.296→16 tokens、服务端模型TTFT降低90.099%，240/240更快；树阈值IR跨异构实例payload降低51.715%且决策路径严格一致 |
| 框架创新二 | `PASS` | 面向关联任务冲突感知的摘要先行渐进证据调度：400次首次summary、380次因高风险、不确定性或模型/规则分歧等闭集原因按需feature、20次summary足够、普通事件主动raw为0；应用JSON降低21.464565%，五类结果400/400等价。通信量以完整raw为基线，业务语义以冻结正式路径A（11,113,184 B）为参考 |
| 框架创新三 | `PASS` | 安全门控双时标响应与可修订终态：安全响应相对阻塞权威终态均值提前71.545%，12个需云确认动作零提前授权；25/25截止后迟到组升级终态，错误final为0 |
| 正式协作路径选择 | `CollaborationScheduler` | 依据冻结风险、网络与时限规则选择边缘、同步云或异步云路径 |
| 交通场景 | 以最终矩阵为准 | 正式定位为四区域感知、风险分析与决策建议；ALINEA 仅为附加仿真证据 |

### 三项框架创新的统一口径

第一项是模型绑定的紧凑语义编码与单token推理。生成式模型使用与训练合同一致的16-token raw16任务ABI；同一0.8B Q4和240个配对事件中，非原生长语义输入平均167.296 tokens、服务端模型TTFT 359.997 ms，原生raw16为16 tokens和35.644 ms，TTFT降低90.099%，240/240更快。该结果是表示与训练合同匹配的联合效果；两臂动作只一致32/240，不声称业务等价或纯token因果。树阈值IR作为跨异构模型第二实例，相对121维active-float32把平均payload从484.000 B降至233.698 B，预测1600/1600、单树路径80,000/80,000一致。1600个网络事件来自800个唯一特征行，4,254,279 B共享工件未计入逐事件字节，主机Python开销不代表Jetson端到端时延。

第二项是面向关联任务冲突感知的摘要先行渐进证据调度方法。交通同一100窗口、400事件首次均只发送summary；20个事件summary已足够，380个因高风险、不确定性或模型/规则分歧等闭集原因按需升级feature，普通事件主动raw为0。相对四节点完整主分区raw预送，应用JSON从11,328,303 B降至8,896,732 B，降低21.464565%，五类业务结果均400/400等价。通信量以完整raw为基线，业务语义以冻结正式路径A（11,113,184 B）为参考。正式100窗自然道路集冲突为0；在真实sample上受控注入的内容、策略和动作摘要不一致只回拉exact 2个owner的raw并重算，证据获取失败不授权。工业只验证状态驱动热图上传的跨场景适配，当前没有云端反向pull，不称与交通同闭环。

第三项是安全门控的双时标响应与可修订终态。同一次100窗口、400事件正式运行的端点配对中，安全门控业务响应均值159.703 ms，相对阻塞到权威终态的561.247 ms提前71.545%；12个需云确认动作零提前授权，局部到权威终态有185/400次修订。该实验是一次运行上的反事实端点配对，不是独立live A/B，提前返回的是provisional。受控终态实验中，当前协议在100/100组给出首结果，对最终可完整的75组恢复权威终态75/75，对25/25截止后迟到组完成revision升级，错误final和不完整证据危险授权均为0。四类到达条件各25组，使用受控时钟，不表示现实故障率或计算性能。

工业阈值优化、9B教师链、确定性调度、Outbox、鉴权、缓存和发布门继续作为业务效果、稳定性或工程边界，不再作为独立创新主张。

最终自动化回归收据：根框架与工业套件 376/376，交通场景 128/128，演示检查 9/9，合计 513/513 通过、0 跳过、0 失败、0 错误，traffic smoke `PASS`。道路集、选择性补证据与summary-first专项 83/83 另作定向重跑，因用例已包含在上述完整套件中，不重复计入 513。完整回归使用 traffic Python 环境和 scikit-learn 1.7.2，收据见 `results/final_submission_v1/evidence/system/final_test_execution_summary.json`。

工业最终评分以总矩阵为准；公开主口径只保留RGB/红外AUROC与73→18多目标阈值开发实验。96.25%只表示单个逻辑摘要相对单张热图的尺寸差，不冒充链路总字节；网络实验使用隔离命名空间中的kernel netem，不冒充公网实测。

项目源码位于 `cloud-edge-scene-sdk`。当前交付压缩包包含源码、静态 Q4、场景模型资产、Nano 运行库、正式配置、门禁摘要和源证据哈希；它没有包含摘要所引用的全部逐请求 JSON、内存采样和冲突原始结果。交付以本地压缩包的文件名、字节数和 SHA-256 为准；不把尚未建立的远端标签写成既成事实。

预留发布地址为 `competition-final-v1.0.0`；在远端标签和附件实际上传、逐项核验前，不把该地址列作已交付证据。

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
  ├─ CollaborationScheduler 选择 edge_only / cloud_async / cloud_sync
  ├─ 云不可用或执行失败时进入本地自治运行状态
  └─ Durable handoff + SQLite Outbox
        │  紧凑摘要、按需证据
        ▼
云端公共入口 :18100
  ├─ 按 sample/group 持久汇聚
  ├─ 场景专用 ExtraTrees 综合判断
  ├─ 全局冲突协调与约束检查
  └─ Qwen3.5 9B 低频异步辅助审计
        │
        ▼
最终业务结果回填边缘（内部字段仍保留历史兼容名称）
```

边缘模型不是直接处理所有原始模态。交通时序模型和工业视觉模型先把大输入压缩为结构化态势；0.8B 模型只处理短编码并输出动作。这样既降低推理时延，也避免把摄像头图像、热力图或完整交通张量持续上传到云端。

云端9B不替代场景专用ExtraTrees，也不参与每一个请求。提交源码的目标语义是：ExtraTrees先给出云端综合判断，低置信或复杂组再异步记录9B审计意见。这个源码修改尚未在5070Ti与Orin正式拓扑重新部署验证。旧隔离回放pair62仍同步等待9B约23.559秒后超时，是旧测量链的反证，不能称旧生产已经异步。精确模型身份以最终证据清单为准，README不沿用早期9B TTFT或模型SHA。

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

### 3. 三类路径与执行降级

`CollaborationScheduler` 是唯一会改变最终执行路径的模块，输出三类正式路径：

- `edge_only`：本地结果已经足够，直接完成业务。
- `cloud_async`：先返回不越过安全边界的边缘即时结果（代码字段：`provisional`），摘要进入持久队列，云端融合结果随后回填。
- `cloud_sync`：动作必须等待云端确认，业务终点是最终业务结果（代码字段：`final`）。

`local_autonomy` 不是第四个学习动作，而是云不可用或云路径执行失败后由统一执行降级阶段触发的运行状态。异步返回只是缩短响应，不是放宽安全规则；边缘服务会保留网络快照和路径选择原因。

### 4. 持久传输和结果生命周期

普通事件先写入带校验和的 durable handoff，随后进入 SQLite Outbox。后台 worker 批量上传摘要；失败任务按退避策略重试，云端以请求指纹和幂等键去重。该链路提供“本地持久接受 + 至少一次重试 + 接收端幂等”，不把它描述成跨机 exactly-once。

一个决策会经历：

```text
边缘即时结果（`provisional`）
  -> queued / inflight
  -> cloud durable acceptance
  -> 阶段性结果或最终业务结果
  -> edge review record completed
```

云端成员未到齐时只能产生阶段性结果（代码字段：`partial_final`），不能冒充全局确认。结果回填保留 revision，同一聚合组后来补齐成员时会得到更高 revision 的 `final`。服务重启后，Outbox、待复核记录、幂等表和聚合状态都从 SQLite 恢复。

### 5. 聚合和冲突协调

聚合键由场景插件定义。交通按同一时间窗和道路分区汇聚，工业按 `product + sample_id` 汇聚 RGB 与红外。框架等待预期成员到齐，再把完整组交给场景协调器；到期缺失则保留缺失列表和未确认状态。

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

ASTGCN未来预测路径仍保留，用于预测类实验和消融；正式低时延链路采用当前态势感知。两者使用同一套事件协议和后续协同流程，指标不能混写。

### 2. 边缘决策

每个交通事件的边缘主链固定为：

```text
专业感知
  -> Student 初步判断
  -> 规则约束
  -> 每事件Qwen建议
  -> 规则硬授权
  -> Student拒绝或运行时回退
  -> 局部态势与复核建议
```

正式主线对每个事件调用Qwen给出建议，再由规则完成动作合法性、安全底线和执行器硬授权；建议被拒绝或运行时失败时由Student回退。模型读取16位紧凑态势编码并返回单token内部结论，对外正式输出只解释为局部态势、风险和协调/复核建议。历史A-F控制标签仍保留在开发证据中，不代表在线TraCI控制器，也不允许据此宣称自动交通控制收益。选择性Q4只作为81.257 ms路径消融，不代表正式链。

### 3. 云端协调

同一时间窗的四个区域事件在云端聚合。当前态势专用 ExtraTrees 根据四区特征形成全局风险、重点区域和协调/复核结果。Qwen3.5 9B 只对少量低置信或复杂结果做异步辅助审计；普通事件不会为了“使用大模型”而强制调用 9B。

当前Q4在线链覆盖100个聚合窗口、400个区域事件，协调前后均未出现自然冲突，因此自然冲突率为0/400，不能由此推导阳性冲突解决率。为检查协调分支，同一批当前态势输入在Q4与调度前的student候选层检测到2/26对冲突（7.692%，未达到≤5%），协调后残余为0；这只是诊断口径，不能替代部署链指标。另用当前态势边界事件构造48个压力样本，48/48均检测并解决。

### 4. 交通实测

| 测试 | 结果 |
| --- | --- |
| 正式边0.8B Q4 / 云端交通ExtraTrees保持率 | 同一1600条normal与weak-network事件；Accuracy 91.5194%，Macro-F1 95.9627%，Weighted-F1 95.2893%，三门PASS |
| Nano 严格交替中的交通子集 | 250 条，accuracy 70.4%，weighted-F1 71.19% |
| 当前态势云 ExtraTrees | 测试准确率 73.73% |
| 交通请求级含感知业务端到端 | 全Qwen primary 400个单节点事件请求；均值159.703 ms，P95 232.502 ms，331/400不超过0.2 s；赛事均值门PASS |
| 自然冲突 | 在线 0/400；未触发阳性解决分支 |
| 候选冲突诊断 | 2/26（7.692%），协调后 0；未达到 ≤5% |
| 注入冲突 | 48/48 解决 |

正式交通策略为“每事件Qwen建议＋规则硬授权＋Student回退”：400/400实际调用0.8B Q4，242条直接采纳，158条因候选动作或风险安全约束回退，业务400/400成功。赛事正式单位为一个边缘节点的一次事件请求，含感知均值159.703 ms低于200 ms，因此正式PASS。四节点请求共享单slot时的max窗口均值227.446 ms仅作并发诊断，不进入赛事门。源实验包按更严格窗口门得到的旧`NO_GO_AS_FINAL_PRIMARY`未删改，由官方事件口径卡标记为superseded。选择性Q4的事件均值81.257 ms、窗口均值151.885 ms只作低时延消融。仓库默认复现配置切换为primary；现网未现场切换。

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

工业数据来自 Hugging Face 数据集 `orgjy314159/MulSen_AD` 的第三个压缩包 `MulSen_AD_new.zip`。最终评价覆盖 10 个产品的 RGB 和 Infrared 两种模态，每个模态 443 条；训练、开发和测试身份由冻结清单隔离。原始图片不进入 Git，证据包记录来源、样本身份和 SHA-256。

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

RGB和红外共享视觉骨干，同时使用模态独立的memory bank。Orin上的正式测量包含边缘视觉流水线；不同主机或旧单产品烟测不再作为最终十产品成绩。

视觉模型实测：

| 模态 | 宏图像 AUROC | 宏像素 AUROC | Orin 单样本均值 |
| --- | ---: | ---: | ---: |
| RGB | 90.27% | 98.08% | 34.06 ms |
| Infrared | 93.03% | 95.83% | 31.65 ms |

以上只外推到已评价的 10 个已知产品，不宣称未见产品泛化。逐产品分母和时延口径见工业最终证据包。

### 2. 边缘复核

视觉模型先根据产品和模态对应的 `review_low/review_high` 形成三态初判：

- 分数低于下界：`normal`；
- 分数位于区间：`review`；
- 分数高于上界：`anomaly`。

确定的 `normal` 和 `anomaly` 走快速路径。只有落在 review band 的样本才把相对阈值距离、模态和产品编码成 16 位序列，交给同一静态 Q4 复核。工业动作 token 为 `A=normal`、`B=review`、`C=anomaly`。Q4 超时、输出不合法或与安全规则冲突时，结果保持 `review`，不会把不确定样本自动放行为 normal。

“边缘复核”解决的是阈值附近的决策稳定性，不是重新做一次视觉特征提取。专业视觉模型先完成图像感知，Q4只读取紧凑态势编码。冻结960条工业三态任务中，边0.8B Q4的Accuracy/Macro-F1/Weighted-F1及合法输出率均为100%；同0.8B BF16三项及逐条一致率100%只作量化参考。该三态阈值策略与正式云端ExtraTrees双模态工件融合不是同一业务任务，因此不构造工业9B/0.8B云边能力保持率。这也不是图像检测准确率。

工业含感知E2E使用两个冻结实测分布做组件组合：500个事件的均值54.933 ms、P95 62.046 ms，500/500不超过0.2 s；RGB与红外并行的工件级均值56.394 ms、P95 64.539 ms。感知起点是已准备的160×160 NCHW张量，不包含PNG读取、缩放和归一化；它不是同一请求逐条直测。这些测量边界保留在限制说明中，不降低双场景平均E2E门的`PASS`裁决。

一次隔离回放把此前Orin真实感知的冻结输出送入正式Q4、规则调度和云端融合：前61对共122个事件中，Q4实际选择25次并完成25/25，selected-only均值81.052 ms；未选Q4的97条边缘即时结果均值35.126 ms，选中Q4的25条为116.088 ms。调度器首选`cloud_sync 83 / cloud_async 39`；该回放启用`prefer_provisional`，实际接受并执行为`cloud_async 122/122`。这不是本轮从原始PNG开始的同请求全链，也不能把冻结感知与回放时延机械相加成正式E2E。

### 3. 云端跨模态协调

RGB和红外事件以`product + sample_id`为聚合键。云端收到完整两模态后，把两侧分数、阈值位置、局部状态、模型身份和证据完整性送入冻结resolver。工业与交通共享训练、加载、调度、审计和发布框架，场景特征和目标各自隔离。

云端主流程是：

```text
RGB + Infrared 完整组
  -> 冻结云端resolver综合判断 normal / anomaly
  -> 置信度和预期收益计算
  -> 公共冲突协调器检查
  -> 最终业务结果
  -> 同一结果回填 RGB 与 Infrared 两条边缘记录
  -> 提交源码对少量低置信组异步调用 Qwen3.5 9B
  -> 审计意见另行记录，必要时形成后续修订
```

冻结resolver负责云端综合判断。Qwen3.5 9B位于受成本门控制的异步辅助审计路径，其`challenge`只记录建议和理由，不直接覆盖业务判断。

工业公开主结果只保留视觉AUROC与多目标阈值开发实验。RGB/红外Image AUROC为90.27%/93.03%。阈值实验固定同一443条全量标签、40维阈值和冻结resolver，优化期间不重训resolver，共搜索65,167个坐标候选。硬冲突73/443→18/443，Accuracy 91.20%→94.36%，Macro-F1 86.88%→91.46%，异常Recall 95.92%→98.54%，误放行14→5、误隔离25→20、review 248→148，综合代价下降45.3685%。状态为`DEVELOPMENT_DEMO_ONLY`；候选只进入隔离开发注册表，生产端点、正式阈值和插件均未修改。

### 4. 弱网、通信和冲突

工业稳定性实验覆盖 normal、目标附加 RTT 约 50 ms/1%、约 100 ms/5%、约 200 ms/10% 和临时断网五档，共 500 条。边缘业务 500/500 成功，受损窗口未完成的 263 条在恢复后 263/263 补传成功，最终丢失 0、服务重启 0。该实验使用隔离 Docker 网络命名空间中的 kernel netem，不表述为公网实测。

同一 886 条事件每臂配对中，full 对全部事件上传完整证据，selective 只对复核或异常事件上传。应用请求体从 93,678,858 B 降至 78,872,766 B（15.81%）；捕获双向 L2 总量从 101,071,507 B 降至 85,444,120 B（15.46%）。L2 口径不含 FCS、前导码、帧间隙和物理无线空口开销。旧 96.25% 只保留为单个逻辑摘要相对单张热图的尺寸对比。

交通通信按同一PEMS08前100个test窗口单独评价。summary-first路径的400个成员事件首次上行全部为compact summary，云端按允许条件回拉feature 380次，raw回拉0次；应用JSON总量为8,896,732 B。四个owner分别上传本区完整未归一化float32 raw窗时，43/42/42/43个传感器合计170且无重复，总应用JSON为11,328,303 B。summary-first加按需证据回拉减少2,431,571 B（21.464565%），final business、risk level、route decision、cloud final result和conflict result均为400/400一致。通信量以完整raw为基线，业务语义以冻结正式路径A（11,113,184 B）为参考，裁决为`TRAFFIC_SUMMARY_FIRST_PASS`。交通字节不含HTTP头、TCP/IP、TLS、L2和物理链路；all-feature与incident-overlap all-raw仅保留为辅助消融，不作为正文主指标。

工业模型1.0.0冻结汇总记录：至少一路review为103/443，normal↔anomaly硬动作冲突为18/443（4.063205%），三态完全一致为337/443。review是可逆中间态，不计硬冲突，因此冲突发生率通过5%数值门。工业30/30与交通48/48受控冲突挑战共78/78解决，按赛题实验口径解决率为100%，一致性整项为`PASS`。Git工作簿固定了20组产品×模态阈值与边际计数；443对配对CSV和18个冲突样本ID未提交，所以不能把受控挑战写成自然18件18/18。自然18件解决率仍为`NOT_EVALUATED`。9B继续作为低频异步辅助审计，不阻塞工业最终业务结果，也不直接替代冻结resolver。

### 5. 工业复现

安装工业依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -e scenes/industrial_anomaly --no-deps
python -m pip install -r scenes/industrial_anomaly/perception/requirements.txt
```

下列命令只用于 `capsule` 开发烟测，不是最终十产品评价的复现入口：

```bash
python scenes/industrial_anomaly/perception/download_mulsen_subset.py \
  --product capsule \
  --output /path/to/mulsen-capsule
```

评估 RGB PatchCore；红外只需替换 `--modality` 和 memory bank：

```bash
python scenes/industrial_anomaly/perception/patchcore_onnx.py evaluate \
  --onnx scenes/industrial_anomaly/perception/assets/vit_small_patch8_160.onnx \
  --bank scenes/industrial_anomaly/perception/assets/banks/rgb/capsule.pcbank \
  --product-root /path/to/mulsen-capsule/MulSen_AD/capsule \
  --modality rgb \
  --review-bands scenes/industrial_anomaly/industrial_anomaly/review_bands.json \
  --output /tmp/capsule-rgb-eval.json
```

从一张真实图片生成可直接提交给边缘服务的事件：

```bash
python scenes/industrial_anomaly/perception/patchcore_onnx.py infer-event \
  --onnx scenes/industrial_anomaly/perception/assets/vit_small_patch8_160.onnx \
  --bank scenes/industrial_anomaly/perception/assets/banks/rgb/capsule.pcbank \
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

赛事分项、最终状态、测量值、基线、证据路径和限制只在 [`results/final_submission_v1/final_scoring_matrix.json`](results/final_submission_v1/final_scoring_matrix.json) 维护。报告和演示不得从旧 README 表格、日期版组会材料或单次开发结果抄数。

### 1. 当前可进入主结果的证据

- 工业是核心效果场景：十产品双模态感知、Orin 推理、真实配对通信捕获、受控弱网、恢复补传和冲突挑战均有冻结证据。
- 交通是第二场景和扩展性证据：正式主链只做四区域感知、边缘判断、规则调度、云端聚合、风险建议，以及必要时的9B低频异步辅助审计，不宣称自动控制收益。补充4→3节点配对实验中两臂窗口成功率均100%，存活事件风险、动作和调度路由100%一致；三节点吞吐提高52.94%，传感器覆盖诚实下降到75%。
- 两场景共用事件入口、`CollaborationScheduler`、持久 Outbox、幂等重试、事件版本和过期结果保护。

### 2. 最终裁决边界

- 最终TTFT与基础毫秒级业务语义门固定比较Qwen3.5-9B Q4和最终joint Qwen3.5-0.8B Q4。TTFT降低86.767%，通过75%门；Math、Code、NLR语义保持率分别为96.55%、100%、90%，三类均通过80%门。能力运行使用同一冻结llama.cpp并请求最大GPU offload（`--gpu-layers 999`），精确tensor placement未归档；该基础门不证明广义通用泛化。
- 正式专用云边能力仅评价交通，云端交通ExtraTrees为分母、边0.8B Q4为分子。同一1600条normal与weak-network事件上，Accuracy/Macro-F1/Weighted-F1保持率91.52%/95.96%/95.29%，三门PASS。ExtraTrees读取226维特征，Q4读取部署原生raw16；两臂完成同一A-F任务，输入载体不同。离线800条不进入分母，9B只负责异步复核和教师标签。
- 正式交通策略为全Qwen primary：每事件由Qwen给出建议，规则只做动作合法性、安全底线和执行器硬授权，失败时Student回退。400/400调用、242采纳、158规则回退；按赛题单节点事件请求口径，含感知均值159.703 ms通过。四节点max窗口均值227.446 ms仅作非计分并发诊断；选择性Q4的81.257 ms事件均值作为低时延消融。仓库默认复现配置使用primary，现网未现场切换。
- 工业当前主结论是：RGB/红外Image AUROC为90.27%/93.03%；多目标阈值开发实验将硬冲突73/443降至18/443，同时提高Accuracy、Macro-F1和异常Recall。443条标签全部参与选参，状态为`DEVELOPMENT_DEMO_ONLY`，不转换成生产收益声明。
- 正式系统只使用`CollaborationScheduler`的冻结规则选择路径；自动控制不进入最终提交主链。

### 3. 共同边界

- 应用层字节、捕获 L2 字节和物理空口总字节必须分别标注。
- 故障注入、隔离 kernel netem 和真实公网条件必须分别标注。
- 冲突挑战集的解决率不能替代自然业务流量的冲突发生率。
- 提交源码已将9B改为必要时的低频异步辅助审计，不替代ExtraTrees；该修改尚未在5070Ti与Orin正式拓扑重新部署验证，旧pair62仍是同步等待导致超时的反证。
- 当前正式硬件只写 RTX 5070 Ti + Jetson Orin Nano；RTX 3090 与 `192.168.31.160` 仅属于历史部署痕迹。

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

最终提交包以`results/final_submission_v1/`中的评分矩阵、证据索引、报告一致性审计和SHA256SUMS为入口。原始大数据集不重复打包；每项成绩必须能从矩阵跳转到仓库内证据或明确标记的外部冻结证据。历史实验目录可以随源码保留，不得被界面或主报告自动汇总为当前成绩。

## 七、核心文件

| 路径 | 内容 |
| --- | --- |
| `cloud_edge_framework/server.py` | 公共 HTTP 协议和服务入口 |
| `cloud_edge_framework/scheduling.py` | 网络、风险、时限路由 |
| `cloud_edge_framework/reliable_transport.py` | durable handoff、Outbox 和重试 |
| `cloud_edge_framework/aggregation.py` | 多成员持久聚合与结果阶段管理（内部兼容字段 `finality`） |
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

系统的边界很明确：场景感知回答“看到了什么”，边缘模型回答“现在能做什么”，云端协调检查多节点行动是否冲突。持久传输和发布机制让网络波动、服务重启或模型更新后的结果仍能追溯到原事件和对应的模型版本。
