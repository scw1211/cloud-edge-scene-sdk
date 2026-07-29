# 云边协同框架学习手册

本文对应独立SDK `cloud-edge-scene-sdk` 0.12.0和公共框架0.4.0。目标不是教你逐行背代码，而是让你能回答三件事：

1. 一个场景事件从边缘进入后，经过哪些对象和模块；
2. 云断开、请求重复、模型更新或多节点冲突时，系统如何处理；
3. 新场景应该改哪些文件，哪些公共文件不应修改。

## 先记住一句话

场景专用模型先完成图像、点云、波形或时序数据的识别，再由场景插件把识别结果变成统一语义事件；公共边缘运行时完成初判、证据裁剪和动态调度，必要时请求云端复核或协调；Edge Qwen 的训练、量化、发布和在线加载由另一套模型工厂管理。

框架不是一个模型，而是三部分的组合：

| 部分 | 解决的问题 | 主要目录 |
| --- | --- | --- |
| 在线协同运行时 | 事件接入、边缘决策、动态调度、云端复核、冲突协调、断网自治 | `cloud_edge_framework/` |
| Edge LLM 生命周期 | 基座锁定、通用蒸馏、场景 SFT、评估、LoRA 合并、GGUF 量化、发布回滚 | `edge_llm_factory/`、`edge_llm/` |
| 场景接入层 | 定义本场景数据格式、语义映射、动作、证据和模型配置 | `scene_plugin_template/`、`scene_adapter_template/` |

## 一、系统边界

### 框架负责

- 接收统一外部事件信封；
- 调用场景插件校验和标准化数据；
- 调用边缘场景决策模型；
- 根据风险、置信度、网络、deadline 和历史实测选择计算路径；
- 选择上传摘要、压缩特征或局部原始证据；
- 在弱网和断网时本地执行并可靠排队；
- 云端恢复后重放、复核并形成纠错样本；
- 检测真正共享资源上的动作冲突并协调；
- 管理 Edge Qwen 的训练、量化、发布和回滚。

### 场景团队负责

- 原始感知模型，例如交通预测、工业异常检测或电网故障识别；
- 场景原生输出的 JSON Schema；
- 从原生结果到风险、置信度、证据、关联范围和候选动作的映射；
- `edge_decide()` 和 `cloud_decide()` 中的真实模型；
- 本场景 LoRA 数据、动作映射和未见测试集；
- 本场景冲突定义和控制收益验证。

### 框架不负责

- 自动理解任意业务 JSON；
- 用一个通用 LLM 直接处理所有原始模态；
- 自动证明场景风险标签正确；
- 把“模型结果不同”直接判为冲突；
- 用模板规则代替真实模型效果。

## 二、五个核心对象

### 1. `SceneEventEnvelope`

外部接入协议，定义在 `event_envelope.py`，对应 `schemas/scene_event_envelope.schema.json`。

它只统一事件身份和路由字段：`id`、`source`、`type`、`scene`、`edgeid`、`time`、`dataschema`、`data`。其中 `data` 由各场景自己的 Schema 决定，因此工业可以传异常分数和热力图引用，交通可以传区域风险和预测摘要，电网也可以传自己的状态结果。

### 2. `SemanticEvent`

框架内部统一语义，定义在 `contracts.py`，对应 `schemas/semantic_event.schema.json`。

它包含：

- `scope`：实体、子系统、区域、时间窗、共享资源和关联键；
- `prediction`：场景模型预测及置信度；
- `risk`：公共调度可理解的风险等级和分数；
- `uncertainty`：校准状态、预测集合和方法；
- `timing`：deadline、预处理和边缘推理耗时；
- `evidence`：summary、feature、raw 三层证据；
- `candidate_actions`：场景允许执行的动作；
- `scene_payload`：必要时保留的场景原始结构；
- `metadata`：内部调度提示和版本信息。

外部模型不需要直接生成这个对象。场景插件的 `normalize()` 负责转换。

### 3. `DecisionEnvelope`

统一决策结果，定义在 `contracts.py`，对应 `schemas/decision_envelope.schema.json`。

它记录决策、动作、置信度、来源、策略版本、执行路径和状态。边缘初判、云端复核和冲突协调都使用同一结构，便于比较与生成纠错数据。

### 4. `NetworkSnapshot`

边缘对当前云路径的观测，定义在 `scheduling.py`。正式服务由 `CloudNetworkMonitor` 主动探测生成，不接受调用方伪造“网络良好”。内容包括 RTT、抖动、丢包、上下行带宽和云端排队/计算估计。

### 5. `ScheduleDecision`

调度器输出，定义在 `scheduling.py`。它不仅给出路由，还给出预测闭环时延、deadline、证据级别、上传字节数、是否等待云端和选择原因。

四条路由是：

| 路由 | 含义 |
| --- | --- |
| `edge_only` | 低风险且稳定，边缘结果直接完成 |
| `cloud_sync` | 当前网络和 deadline 允许，等待云端复核后返回 |
| `cloud_async` | 先返回临时边缘动作，事件进入 Outbox，稍后云端复核 |
| `local_autonomy` | 云不可用时的故障降级；允许的本地动作立即执行，高风险事件排队补传且全局状态为待复核 |

## 三、一条请求如何跑完

正式双服务链路如下：

```text
场景感知模型
  -> 完整 SceneEventEnvelope JSON
  -> POST 边缘 /api/v1/collaboration/decide
  -> EdgeApiService.decide()
  -> 请求幂等校验
  -> EdgeRuntime.process()
  -> ScenePlugin.validate_envelope()
  -> ScenePlugin.normalize()
  -> ScenePlugin.edge_decide()
  -> EvidencePlanner.plan()
  -> ScenePlugin.prepare_cloud_event()
  -> CollaborationScheduler.schedule()
       -> edge_only
       -> cloud_sync -> 云端 cloud_decide()
       -> cloud_async -> SQLite Outbox
       -> local_autonomy -> 本地动作 + 必要时 Outbox
  -> DecisionEnvelope + 调度、时延、传输和证据统计
```

### 1. HTTP 接入

`edge_service.py` 的 `EdgeApiService.decide()` 接收 `{ "event": {...} }`。它提取事件 ID 和 trace ID，生成幂等键，并读取边缘主动探测得到的网络快照。

同一个幂等键和同一请求内容只执行一次；重复请求返回第一次结果。相同幂等键对应不同请求内容会报冲突，防止误把另一事件当作重试。

### 2. 校验和标准化

`EdgeRuntime.process()` 先把字典解析为 `SceneEventEnvelope`，再由 `SceneRegistry` 根据 `scene`、`type` 和 `dataschema` 找到插件。

插件基类会同时检查：

- scene 是否由该插件处理；
- event type 是否在白名单；
- dataschema URI 是否完全匹配；
- `data` 是否满足插件自己的 JSON Schema。

全部通过后，插件的 `normalize()` 才构造 `SemanticEvent`。框架不会猜字段，也不会静默回退到旧格式。

### 3. 边缘初判

`plugin.edge_decide(event)` 应调用真实边缘决策模型。模板中只是根据候选动作演示接口，正式实验必须替换，且 `health().template_mode` 应为 `false`。

如果使用 Edge Qwen，推荐让模型只输出一个动作 token，再由 `ActionDecoder` 映射为经过授权的场景动作。这样可以限制输出长度、避免无关文本，并在非法 token 时使用明确的安全回退。

### 4. 证据选择

`EvidencePlanner` 根据风险、不确定性、冲突和场景声明的最低证据级别选择：

- `summary`：风险、置信度和少量统计量；
- `feature`：热力图、嵌入、压缩时序等任务特征；
- `raw`：争议场景需要的局部原始证据或对象存储 URI。

随后 `plugin.prepare_cloud_event()` 删除边缘私有数据，并决定云端实际收到哪些内容。调度器使用真实序列化字节数估算传输时延。

### 5. 动态调度

`CollaborationScheduler.schedule()` 综合：

- 风险是否达到 high/severe；
- 预测和风险置信度是否低于阈值；
- prediction set 是否含多个可能等级；
- 边缘模型是否主动请求云复核；
- 是否怀疑多节点冲突；
- 网络是否可用、丢包是否过高；
- 上传和返回数据的预计传输时间；
- 当前 deadline；
- 同类请求历史实测的云路径 EWMA。

因此它不是“高风险固定上云”的硬编码开关，而是受时限约束的计算路径选择器。

### 6. 云端复核

同步路径通过 `ReliableHttpCloudClient` 调用云端 `/api/v1/collaboration/cloud-decision`。云端 `CloudApiService` 只接收已经标准化的 `SemanticEvent`，再由 `CloudRuntime.decide()` 调用该场景的 `cloud_decide()`。

云端成功后，边缘保存“本地初判 vs 云端修正”的反馈记录；云端失败则自动退回本地自治并把事件放入 Outbox。

### 7. 输出和计时

边缘结果同时返回：

- `local_decision` 与 `final_decision`；
- `schedule` 和选择原因；
- `evidence_plan`；
- 旧全量上传与当前选择上传的字节数；
- 分阶段耗时；
- 计入预处理、场景推理和框架闭环的统一口径；
- 当前待复核和反馈样本数量。

## 四、两层可靠性不要混淆

### 第一层：旧模型文件到边缘服务

由 `file_bridge.py` 负责。旧工程只需要把完整事件 JSON 原子写入 `inbox`，桥接器完成：

```text
文件稳定性检查
  -> 公共信封 Schema 校验
  -> 场景 data Schema 校验
  -> 本地 file URI 证据检查
  -> SQLite DurableEnvelopeOutbox
  -> HTTP 发送到边缘服务
  -> 回执 / 重试 / 隔离非法文件
```

这层解决的是“场景程序和框架 Python 环境不同、文件刚写一半、边缘服务临时不可用”等问题。

### 第二层：边缘服务到云服务

由 `reliability.py`、`reliable_transport.py` 和 `replay.py` 负责：

- `SQLiteOutbox` 持久保存待云复核事件；
- claim/lease 防止多个重放线程重复处理同一事件；
- 指数退避避免故障期间不停打云端；
- `SQLiteIdempotencyStore` 保证重试不会重复执行云动作；
- `OutboxReplayWorker` 在网络恢复后批量协调并确认完成；
- 失败事件释放回队列，不会因为一次异常消失。

这层解决的是“云端弱网、断网、请求超时、进程重启和重复重试”等问题。

### 第三层：临时判断到最终结果的生命周期

可靠传输只保证事件不丢，`review_tracking.py`进一步记录决策是否发生变化：

```text
边缘临时判断
  -> 待复核
  -> 复核处理中
  -> 云端最终结果
  -> 是否修正、同步闭环或异步最终时延
```

正常网络下由边缘直接完成的事件可以成为本地最终结果；断网自治事件只要已经进入补传队列，全局状态就是“待复核”，不能因为当前已经返回一个安全动作就冒充全局最终结果。需要云确认的不可逆动作只进入延后列表，不会提前授权。

## 五、多节点协调如何工作

`CloudRuntime.coordinate()` 先按场景调用 `fuse_cloud_context()`，场景插件只能补充上下文，不能改变事件数量和身份。之后云端分别生成候选决策，再交给 `ConflictCoordinator`。

在线到达的多边缘事件还可以通过`aggregation.py`按场景插件提供的关联键持久汇聚。全部预期成员到齐时立即协调；超时但满足最小成员数时执行部分汇聚；重复事件按事件编号去重。关联键、成员名称、预期成员和超时时间均由场景插件定义，公共框架不会假设所有场景都使用`sample_id`。

框架只有在以下条件成立时才进一步检查动作冲突：

1. 时间窗相关；
2. 实体、区域、关联键或任务语义相关；
3. 动作影响共同的执行资源；
4. 场景插件判定两个动作不兼容。

默认冲突规则会比较动作类型和参数，并选择风险更高、优先级更高的一侧。真实场景应覆盖 `action_conflict()` 和 `resolve_action_conflict()`，实现本领域的资源约束。协调结果会保存冲突记录、解决原因和最终动作，因此“协调后 0 冲突”不等于回避冲突，而是要同时报告协调前冲突数和解决成功数。

## 六、Edge LLM 从基座到上线

在线运行时和模型工厂是两条链。模型工厂先产出经过验证的发布物，在线运行时只加载活动版本。

### 1. 锁定共享基座

`edge_llm/base_manifest.json` 锁定 Qwen 上游 revision、文本架构、允许的 LoRA 层、动作 token 和资源门槛。`text_snapshot_manifest.json` 锁定从上游多模态快照派生纯文本快照的过程、参数数量和逐文件哈希。

`verify-base` 与 `verify-text-base` 会检查模型身份、文件哈希、动作 token、参数结构和多模态参数是否已移除。场景团队不能把另一个同名模型悄悄替换进来。

### 2. 场景无关通用处理

通用校准链：

```text
通用数学/代码/自然语言样本
  -> build-calibration
  -> build-imatrix
  -> Q4/Q5 GGUF
  -> benchmark-runtime
```

通用蒸馏链：

```text
锁定训练/验证来源
  -> build-general-kd-source
  -> 9B Teacher 生成并自动验收答案
  -> build-general-kd
  -> train-general-kd
  -> gate-general-kd 与原 0.8B 比较
```

通用候选未通过宏平均和分项门禁时，不会自动替换共享基座。

### 3. 场景 LoRA

每个场景在共享文本基座上独立训练 LoRA：

```text
train/val JSONL
  -> train-sft
  -> test JSONL 上 evaluate
  -> merge-lora
  -> export-gguf
  -> build-adapter
  -> validate-adapter
  -> publish_release
```

`scene_adapter_template/pipeline.json` 把这些步骤串成可恢复流水线。每个阶段先检查输入，执行后保存输出摘要和哈希；失败后修复原因，再使用 `--resume` 继续，不能跳过门禁。

### 4. 在线发布和回滚

`ReleaseStore` 原子记录活动 release、历史版本、基座指纹、适配器目录摘要和 GGUF 摘要。激活或回滚前都会重新验文件完整性。

`serve_release.py` 监听活动 release，管理常驻 `llama-server` 进程；`release_runtime.py` 把活动发布、动作映射和 provider 绑定为可调用对象；框架侧 `release_watcher.py` 发现 release 改变后原子重载插件。新版本加载失败时，旧运行快照继续服务。

## 七、新场景真正需要改什么

### 必改：场景插件

复制 `scene_plugin_template/` 后修改：

- `plugin.py`：scene、event types、`normalize()`、边缘/云决策、证据和冲突方法；
- `data_schema.json`：本场景原生模型输出；
- `sample_event.json`：一个合法可运行样例；
- `deployment/framework/scene_plugins.json`：加载新插件类。

### 使用 Edge Qwen 时必改：场景适配器

复制 `scene_adapter_template/` 后修改：

- `action_mapping.json`：动作槽到业务动作的映射；
- `package_spec.json`：Teacher、数据集、测试证据、门禁和部署物；
- `pipeline.json`：实际数据、基座、工具和输出路径；
- `runtime.edge.json`、`runtime.cloud.json`：真实模型服务地址和模型名；
- `datasets/<scene>/train.jsonl`、`val.jsonl`、`test.jsonl`：严格拆分的数据。

### 通常不改：公共层

- `cloud_edge_framework/contracts.py`；
- `runtime.py`、`scheduling.py`、`reliability.py`；
- `edge_llm_factory/`；
- 公共 schemas。

只有多个场景都证明公共协议或机制有缺陷时，才修改这些文件。

## 八、逐文件说明

阅读级别：A 为今天必须精读；B 为理解职责和接口；C 为按需查阅或生成文件。

### 根目录

| 级别 | 文件 | 作用 |
| --- | --- | --- |
| C | `.gitignore` | 排除运行状态、权重、缓存、训练输出等不应提交的文件。 |
| A | `README.md` | SDK 安装、接入、启动、模型流水线和验收标准的主入口。 |
| A | `FRAMEWORK_STUDY_GUIDE.md` | 本手册，解释整体逻辑、调用链和全部文件。 |
| B | `FILE_BRIDGE.md` | 旧工程通过 JSON 文件接入时的目录、原子写入、校验、Outbox、回执和测试口径。 |
| B | `HANDOFF_CHECKLIST.md` | 发给其他场景团队的交付检查表，明确插件、数据、模型和证据要求。 |
| C | `VERSION` | SDK 发布版本，必须与 `pyproject.toml` 一致。 |
| C | `MANIFEST.json` | 构建时生成的文件清单、字节数和 SHA-256，用于核对分发包完整性。 |
| B | `pyproject.toml` | Python 包名、版本、最低 Python、依赖、要安装的包和随包 JSON 资源。 |
| B | `requirements.txt` | 在线运行时最小依赖，目前主要是 `jsonschema`。 |
| B | `requirements-training.txt` | Transformers、PEFT、训练和 safetensors 等离线训练依赖；边缘在线服务不必安装全部训练栈。 |

### `cloud_edge_framework/`：在线协同运行时

| 级别 | 文件 | 作用 |
| --- | --- | --- |
| A | `__init__.py` | 导出外部代码应使用的核心对象和运行时，是公共 Python API 入口。 |
| C | `version.py` | 保存公共框架版本，健康接口会返回它。 |
| A | `event_envelope.py` | 解析和严格校验外部 CloudEvents 风格信封，处理时间、URI、扩展字段和 `data/data_base64` 互斥。 |
| A | `contracts.py` | 定义内部 `EventScope`、`Prediction`、`Risk`、`Uncertainty`、`Evidence`、`Action`、`Timing`、`SemanticEvent`、`DecisionEnvelope`，以及稳定 ID。 |
| C | `plugins/__init__.py` | 导出插件基类。 |
| A | `plugins/base.py` | 定义场景插件必须实现的 schema、normalize、edge/cloud decide 接口，以及可覆盖的证据、融合和冲突方法。 |
| B | `registry.py` | 从 `模块:类` 或 Python entry point 动态加载插件，建立 scene、alias、event type 索引并拒绝重复注册。 |
| B | `plugin_manager.py` | 构建不可变运行快照并提供 lease；热重载成功后原子切换，旧请求结束后再关闭旧插件。 |
| A | `runtime.py` | 核心编排器；`EdgeRuntime.process()` 串联边缘完整链路，`CloudRuntime` 完成云决策和多事件协调。 |
| A | `scheduling.py` | 计算闭环预算和四条路由，综合风险、不确定性、网络、冲突、上传字节和历史时延。 |
| B | `evidence.py` | 选择 summary/feature/raw 证据，统计内联编码、引用源和未压缩源字节。 |
| A | `artifacts.py` | 接收真实大证据文件，校验SHA-256、内容长度和类型，去重保存并统计实际通信量。 |
| A | `aggregation.py` | 按场景关联键持久汇聚多个边缘事件，处理重复、缺失成员、超时部分汇聚和进程恢复。 |
| A | `conflicts.py` | 建立事件关联组，检测共享资源动作冲突，调用场景插件解决并生成可审计记录。 |
| B | `networking.py` | 边缘主动探测云端健康、RTT、抖动和丢包，维护滚动网络快照；`StaticNetworkMonitor` 供测试使用。 |
| B | `performance.py` | 按 scene、证据级别和网络档保存云路径成功率、时延和字节 EWMA，供调度预测使用。 |
| B | `feedback.py` | 异步持久保存本地初判和云端修正差异，形成后续纠错蒸馏或策略更新数据。 |
| B | `review_queue.py` | 简单 JSONL/内存待复核队列，主要供单进程工具和兼容路径使用；正式边缘服务使用 SQLite Outbox。 |
| A | `review_tracking.py` | 持久记录边缘临时判断、待复核、处理中、云端最终结果、修正率和分阶段时延。 |
| A | `reliability.py` | 实现正式 SQLite Outbox 的 enqueue/claim/lease/ack/release，以及云端和边缘请求幂等缓存。 |
| B | `replay.py` | 后台从 Outbox 领取事件，在网络恢复时调用云端多事件协调并确认或退避重试。 |
| B | `transport.py` | 基础 HTTP 云客户端，调用云决策、协调和反馈接口，并附加传输耗时与字节统计。 |
| B | `reliable_transport.py` | 在基础 HTTP 客户端上加入有限重试、幂等键和 trace header。 |
| B | `service_config.py` | 加载和语义校验边缘/云端配置，把相对路径解析到项目根目录。 |
| A | `monitoring.py` | 计算公共校准误差、风险集合覆盖率和数据漂移；监测失效时强制复核。 |
| B | `utility_routing.py` | 加载轻量效用模型，支持只记录不接管的影子模式和通过门禁后的主动路由。 |
| B | `cloud_llm.py` | 对场景专业云模型结果进行可选结构化大模型复核；失败时保留专业模型基线。 |
| B | `http_api.py` | 通用标准库 HTTP 外壳，负责 JSON body 限制、路由转发、状态码和错误响应。 |
| A | `edge_service.py` | 正式独立边缘进程；装配插件、EdgeRuntime、网络探测、Outbox、重放、指标和 release watcher。 |
| A | `cloud_service.py` | 正式独立云端进程；提供单事件复核、多事件协调、反馈、插件热重载和幂等处理。 |
| C | `server.py` | 早期单进程兼容服务，把边和云能力放在一个进程；只用于开发，不作为正式双服务部署。 |
| C | `edge_client.py` | 单次真实边到云调试 CLI，可选择预定义网络档；它绕过正式常驻边缘服务，不作为生产入口。 |
| B | `ingest_client.py` | 读取一个完整事件 JSON，并提交到正式边缘 `/decide` 接口。 |
| A | `file_bridge.py` | 旧模型文件接入桥；完成双 Schema 校验、证据文件检查、SQLite Outbox、inotify 监听、重试、隔离和回执。 |
| C | `benchmark_file_bridge.py` | 测量文件桥的缓存校验和完整持久接入时延，不包含场景模型推理。 |
| B | `release_watcher.py` | 监听活动 Edge LLM release，校验新发布物后触发插件运行快照原子重载。 |
| B | `metrics.py` | 聚合边缘、云端、重放、冲突、路由、时延和失败计数，输出统一 JSON 指标。 |
| C | `benchmark_edge_service.py` | 重复请求正式边缘 HTTP 服务，测 Edge Qwen、云复核和完整链路。 |
| C | `benchmark_services.py` | 为每次请求生成唯一 ID，测独立边云服务真实闭环，避免幂等缓存把结果测低。 |
| C | `benchmark_monitoring.py` | 测量公共监测器的逐事件开销和分位时延。 |
| C | `benchmark_review_fault_matrix.py` | 验证发送前断网、提交前失败、响应丢失、进程重启、重复补传和云端修正等故障点。 |

### `deployment/framework/`：服务启动配置

| 级别 | 文件 | 作用 |
| --- | --- | --- |
| A | `scene_plugins.json` | 插件清单；指定 `模块:类`、启用状态和构造参数，也可开启 entry-point 发现。 |
| A | `edge_service.json` | 边缘监听端口、Outbox/反馈/性能文件、调度阈值、云地址、重试、网络探测和重放参数。 |
| A | `cloud_service.json` | 云端监听端口、插件清单、反馈库和幂等库配置。 |

### `edge_llm/`：共享模型身份和研究证据

| 级别 | 文件 | 作用 |
| --- | --- | --- |
| A | `README.md` | 解释纯文本 Qwen 快照如何从上游多模态模型派生和验证。 |
| A | `base_manifest.json` | 锁定共享基座身份、revision、结构、LoRA 契约、动作槽和资源门槛。 |
| A | `text_snapshot_manifest.json` | 锁定纯文本快照的派生来源、移除参数、最终结构和逐文件摘要。 |
| B | `calibration/calibration.txt` | 场景无关的数学、代码和自然语言量化校准文本。 |
| B | `calibration/manifest.json` | 校准数据来源、排除的冻结评测集、token 数和校准文本摘要。 |
| B | `general_compression.md` | 通用 Q4/Q5、imatrix、资源和能力实测，以及哪些结论不能外推到场景模型。 |
| B | `general_distillation.md` | 9B 到 0.8B 通用行为蒸馏的方法、门禁、实测和当前负结果。 |

### `edge_llm_factory/`：离线模型工厂与在线模型适配

| 级别 | 文件 | 作用 |
| --- | --- | --- |
| B | `__init__.py` | 标识模型工厂包和内部版本。 |
| A | `__main__.py` | `python -m edge_llm_factory <command>` 的统一命令分发入口。 |
| A | `contracts.py` | 校验基座、动作映射、输入契约、适配器 manifest、指标门禁、路径安全和文件摘要。 |
| B | `base_snapshot.py` | 核对锁定基座快照、分片、tokenizer 和动作 token。 |
| A | `text_base.py` | 从上游快照导出/验证纯文本模型，扫描并拒绝视觉塔、MTP 等多模态或辅助参数残留。 |
| B | `general_calibration.py` | 从通用样本构建无测试泄漏、可审计的量化校准文本和 manifest。 |
| B | `build_imatrix.py` | 生成绑定当前模型权重的 llama.cpp importance matrix 命令并执行。 |
| B | `benchmark_runtime.py` | 对常驻 Ollama/llama.cpp 模型测 TTFT、总时延、token 速率和 RSS 峰值。 |
| A | `providers.py` | 统一 llama.cpp、Ollama、OpenAI-compatible 文本生成接口和运行配置校验。 |
| A | `runtime.py` | 强制动作模型短输出，解码动作 token，执行风险/离线权限检查并返回安全动作。 |
| B | `general_kd_source.py` | 从锁定 revision 的独立训练/验证分片构建通用 KD 源数据，并阻止冻结评测 prompt 泄漏。 |
| B | `general_kd_eval.py` | 自动验收 Teacher 的数学答案、选择题和受限代码执行结果，生成规范目标。 |
| B | `general_kd_data.py` | 请求 Teacher 生成通用蒸馏目标，复用已验收结果并记录请求指纹和类别统计。 |
| B | `train_general_kd.py` | 在纯文本 0.8B 上训练共享通用 LoRA，支持采样平衡和只监督 assistant token。 |
| B | `general_kd_gate.py` | 比较候选与基线的完整样本、宏平均和各类别结果，未达门槛则禁止晋级。 |
| A | `train_sft.py` | 使用场景 train/val JSONL 训练 LoRA/QLoRA，验证目标动作 token 和文本模型结构。 |
| A | `evaluate_action_tokens.py` | 在未见 test JSONL 上计算动作准确率、宏 F1、合法输出率和推理耗时。 |
| B | `merge_lora.py` | 校验后把 LoRA 合并到锁定纯文本基座，并清理不应导出的 tokenizer 多模态字段。 |
| B | `export_gguf.py` | 调用 llama.cpp 转换器生成 F16 GGUF，再按指定 Q4/Q5 方法量化并记录摘要。 |
| A | `adapter_package.py` | 构建和验证场景发布包，检查 safetensors、PEFT 配置、评测证据、GGUF 和文件哈希。 |
| A | `pipeline.py` | 执行可恢复流水线，负责输入检查、阶段日志、输出摘要、候选选择、发布和状态持久化。 |
| A | `release_store.py` | 原子发布、查询和回滚 release；记录历史并在切换前重新验证所有产物完整性。 |
| B | `release_runtime.py` | 读取活动 release，检查 runtime 配置是否指向对应产物，并构造在线 `ValidatedEdgeLLM`。 |
| B | `serve_release.py` | 守护常驻 llama-server；活动 GGUF 改变时重启服务并原子更新运行配置。 |

### `model_bundle/`：默认模型分发

| 级别 | 文件 | 作用 |
| --- | --- | --- |
| A | `README.md` | 解释云端 9B Teacher 和边缘通用 0.8B Student 的身份、来源与安装命令。 |
| A | `catalog.json` | 锁定两个模型的角色、下载地址、大小、SHA-256 和 Ollama 标识。 |
| B | `install_models.py` | 下载、安装或只验证模型；摘要不一致时立即失败。 |
| C | `__init__.py` | 标识可通过 `python -m model_bundle.install_models` 调用的 Python 包。 |
| C | `APACHE-2.0.txt` | 随包保存模型许可文本。 |

### `scene_plugin_template/`：在线场景插件模板

| 级别 | 文件 | 作用 |
| --- | --- | --- |
| B | `__init__.py` | 导出示例插件类。 |
| A | `plugin.py` | 完整示例：原生异常检测输出校验、语义映射、证据构造、候选动作和边/云占位决策。 |
| A | `data_schema.json` | 示例工业异常模型 `data` 的 JSON Schema；新场景应复制后重写。 |
| A | `sample_event.json` | 与 Schema 和插件匹配的完整外部事件，用于 smoke test 和手工请求。 |
| B | `smoke_test.py` | 验证正常网络、断网自治和两个共享资源事件的冲突协调，不证明真实模型准确率。 |
| B | `action_mapping.json` | 为 smoke test 提供动作槽到示例工业动作的映射。正式场景以自己的 adapter 目录版本为准。 |
| B | `base_manifest.json` | 为 smoke test 自包含复制的共享基座 manifest；内容来源于 `edge_llm/base_manifest.json`。 |

### `scene_adapter_template/`：场景 LoRA 与运行配置模板

| 级别 | 文件 | 作用 |
| --- | --- | --- |
| A | `README.md` | 场景团队使用 LoRA 模板的最短说明。 |
| A | `DATA_FORMAT.md` | 定义 SFT JSONL 格式：输入紧凑态势编码，assistant 目标为一个动作 token。 |
| A | `action_mapping.json` | 定义 A-H 动作槽、候选动作类型、适用风险、是否请求云端和是否允许离线执行。 |
| A | `package_spec.json` | 锁定场景输入契约、Teacher、训练/测试数据 ID、指标来源、门禁和部署 GGUF。 |
| A | `pipeline.json` | 从基座验证到 release 发布的八阶段可执行模板。 |
| A | `runtime.edge.json` | 边缘模型 provider、endpoint、模型、上下文、单 token、no-thinking 和 keep-alive 配置。 |
| A | `runtime.cloud.json` | 云端 Teacher provider、endpoint、模型和生成参数配置。 |

### `schemas/`：公共协议的机器可校验定义

| 级别 | 文件 | 作用 |
| --- | --- | --- |
| A | `scene_event_envelope.schema.json` | 外部场景事件信封，允许 `data` 或 `data_base64`，业务内容保持开放。 |
| A | `semantic_event.schema.json` | 内部统一事件的 scope、prediction、risk、uncertainty、timing、evidence 和 actions。 |
| A | `decision_envelope.schema.json` | 边缘、云端和协调结果的统一决策结构。 |
| B | `framework_service_config.schema.json` | 边缘/云服务配置，包括存储、云连接、探测、重放、发布监听和幂等设置。 |
| B | `edge_llm_base_manifest.schema.json` | 共享基座身份、架构、LoRA 契约、动作协议和资源门槛。 |
| B | `edge_llm_text_snapshot.schema.json` | 纯文本派生快照的来源、转换、验证和文件记录。 |
| B | `edge_llm_action_mapping.schema.json` | 场景动作槽映射和离线/上云权限。 |
| B | `edge_llm_adapter_manifest.schema.json` | 已构建 LoRA 适配器包的机器清单。 |
| B | `edge_llm_package_spec.schema.json` | 构建适配器前的人类配置，包含数据、评测、部署和门禁要求。 |
| B | `edge_llm_pipeline.schema.json` | 可恢复流水线及阶段结构。 |
| B | `edge_llm_runtime.schema.json` | llama.cpp、Ollama、OpenAI-compatible provider 的统一在线配置。 |
| B | `edge_llm_release_store.schema.json` | 活动 release、历史版本和切换审计记录。 |

## 九、容易混淆的文件

| 容易混淆 | 正确区别 |
| --- | --- |
| `event_envelope.py` 与 `contracts.py` | 前者是外部开放信封，后者是插件转换后的内部固定语义。 |
| `file_bridge.py` 与 `reliability.py` | 前者保证模型文件送到边缘服务，后者保证边缘事件送到云服务。 |
| `scene_plugin_template/` 与 `scene_adapter_template/` | 前者接在线事件和业务语义，后者训练/打包 Edge Qwen LoRA。 |
| `edge_llm/` 与 `edge_llm_factory/` | 前者保存基座身份和实验文档，后者是执行训练、量化和发布的代码。 |
| `model_bundle/` 与 `edge_llm/` | 前者下载已发布模型，后者锁定共享基座和纯文本快照的技术契约。 |
| `review_queue.py` 与 `SQLiteOutbox` | 前者是轻量兼容队列，正式边缘服务使用后者支持 lease、退避和进程恢复。 |
| `server.py` 与 `edge_service.py/cloud_service.py` | 前者是单进程兼容入口，后两者才是正式边云独立部署。 |
| `runtime.py` 两份 | `cloud_edge_framework/runtime.py` 编排业务事件；`edge_llm_factory/runtime.py` 只负责动作模型调用和安全解码。 |
| “模型准确率”与“决策收益” | 前者比较标签/Teacher 一致性，后者必须通过 SUMO 或业务仿真验证控制效果，不能互相替代。 |

## 十、今天下午的阅读顺序

### 第一小时：看懂在线主链

1. `README.md` 的“场景接入步骤”和“启动独立边云服务”；
2. `scene_plugin_template/sample_event.json`；
3. `event_envelope.py`；
4. `scene_plugin_template/plugin.py` 的 `normalize()`；
5. `contracts.py` 中 `SemanticEvent` 和 `DecisionEnvelope`；
6. `runtime.py` 中 `EdgeRuntime.process()`。

目标：能从一个输入 JSON 说到最终决策每一步经过什么对象。

### 第二小时：看懂协同和可靠性

1. `scheduling.py` 的 `schedule()`；
2. `edge_service.py` 的初始化和 `decide()`；
3. `cloud_service.py` 的 `cloud_decision()` 和 `coordinate()`；
4. `reliability.py` 的 Outbox 和幂等存储；
5. `replay.py`；
6. `conflicts.py` 与 `plugins/base.py` 的冲突接口；
7. `file_bridge.py` 只看 `LocalEventValidator`、`DurableEnvelopeOutbox`、`FileEventBridge` 三个类。

目标：能解释断网为什么不影响本地业务、恢复后为什么不会丢或重复执行，以及什么才叫冲突。

### 第三小时：看懂模型生命周期

1. `edge_llm/base_manifest.json`；
2. `scene_adapter_template/action_mapping.json`；
3. `scene_adapter_template/package_spec.json`；
4. `scene_adapter_template/pipeline.json`；
5. `edge_llm_factory/__main__.py`；
6. `train_sft.py`、`evaluate_action_tokens.py`、`merge_lora.py`、`export_gguf.py`；
7. `release_store.py`、`serve_release.py`、`release_watcher.py`。

目标：能从 Teacher 数据说到边缘 GGUF 上线和回滚，并知道每个门禁防什么问题。

### 最后半小时：亲手跑一遍

```bash
python -m pip install -r requirements.txt
python -m scene_plugin_template.smoke_test
```

再开两个终端分别启动云端和边缘：

```bash
python -m cloud_edge_framework.cloud_service \
  --project_root . \
  --config deployment/framework/cloud_service.json
```

```bash
python -m cloud_edge_framework.edge_service \
  --project_root . \
  --config deployment/framework/edge_service.json
```

第三个终端提交事件：

```bash
python -m cloud_edge_framework.ingest_client \
  --event scene_plugin_template/sample_event.json \
  --edge_base_url http://127.0.0.1:18101
```

重点观察返回中的：`schedule.route`、`schedule.reason`、`evidence_plan`、`local_decision`、`final_decision`、`data_plane` 和 `closed_loop_accounting`。

## 十一、学完后应该能回答的问题

1. 为什么外部 `data` 不做成一个固定大 JSON？
2. `normalize()` 为什么是每个场景最关键的接口？
3. `edge_only` 和 `local_autonomy` 有什么区别？
4. 为什么高风险事件有时是 `cloud_async` 而不是 `cloud_sync`？
5. 为什么文件桥和边云 Outbox 必须同时存在？
6. 为什么重复 HTTP 请求不会重复执行云动作？
7. 为什么两个模型预测不同不一定是决策冲突？
8. 场景插件与场景 LoRA 分别解决什么问题？
9. 通用 0.8B 基座、场景 LoRA 和云端 9B Teacher 各自做什么？
10. 新版本 GGUF 加载失败时，为什么旧服务还能继续运行？

能完整回答这十个问题，就已经掌握了这个框架的主体，而不是只会照着命令启动。
