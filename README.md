# 云边协同场景接入 SDK

这个SDK用于把任意具备结构化感知结果的业务场景接入同一套云边协同运行时，并为边缘Qwen提供统一的基座、场景适配器、动作槽、训练流水线和发布校验。

> 通用框架只固定外层事件信封和内部统一语义事件，不固定`data`中的业务字段。`scene_plugin_template/`默认采用异常检测，仅用于演示如何定义场景Schema和转换逻辑；复制模板后应改成自己的场景名称、字段和动作，不能把示例中的8个字段理解成SDK统一输入。

当前SDK版本为0.13.1，对应公共框架0.4.1。本版新增：

- 真实证据文件上传、SHA-256校验、去重和实际通信量统计；
- 在线普通聚合事件先写边缘 Outbox，再由后台逐样本上报；明确要求完整云确认且预算允许的业务可同步等待，未完成时自动转入 Outbox；支持多边缘持久汇聚、超时部分结果和迟到成员重算；
- 边缘临时判断、待复核、云端最终结果和修正率的完整生命周期；
- 公共校准误差、风险集合覆盖率和数据漂移监测；
- 学习式效用路由的影子模式与主动模式；
- 可选云端大模型结构化复核，失败时保留场景专业模型基线。
- `scenes/freeway_traffic/`交付真实 ASTGCN、交通 Student、defer gate、特征编码器、云端 ExtraTrees 和 Edge-Qwen 适配器；大文件由安装器下载并校验 SHA-256；
- 一键真实全链路验收会强制检查四分区全部汇聚、provisional→final 全部回填和残余冲突为零；
- 两台 Jetson 可按 `0,1` 与 `2,3` 分区在同一时刻运行真实感知并向云端汇聚。

固定分工如下：

- 场景专用感知模型负责原始图像、点云、波形或时序数据；
- 公共文本基座负责接收紧凑态势编码；
- 需要边缘大模型的场景维护独立 LoRA 和动作映射，由边缘 Qwen 输出一个动作 token；规则、小模型或其他本地决策器也可直接实现统一插件接口，公共框架不强制每个场景复制交通 Student；
- 云端 Teacher 生成和纠错标签，云端协调器处理跨节点冲突。

源码仓库直接交付体积较小的交通 ASTGCN、Student、defer gate、特征编码器、
ExtraTrees 和 Edge-Qwen 适配器包。PEMS08 推理数组、0.8B GGUF 和 Qwen 9B
不进入 Git，由资产清单锁定下载地址、字节数和 SHA-256。模板决策只验证接口；
交通正式效果必须运行真实全链路验收。

## 目录

```text
cloud_edge_scene_sdk/
├── cloud_edge_framework/          公共云边运行时
├── FILE_BRIDGE.md                  本地 JSON 校验、Outbox 和上传说明
├── edge_llm_factory/              蒸馏、评估、合并、量化和适配器校验
├── model_bundle/                  云端 Teacher 与边缘通用 Student 安装目录
├── edge_llm/base_manifest.json    锁定的共享 Qwen 上游身份与 LoRA 契约
├── edge_llm/text_snapshot_manifest.json  锁定的纯文本派生权重清单
├── edge_llm/calibration/            不含场景样本的通用校准文本与清单
├── edge_llm/general_compression.md  通用压缩方法、能力与资源实测
├── edge_llm/general_distillation.md 通用 Teacher 行为蒸馏方法与实测
├── scene_plugin_template/         可复制改名的场景插件
├── scene_adapter_template/        场景 LoRA、动作映射和流水线模板
├── scenes/freeway_traffic/         可独立安装的交通双边缘参考场景
├── schemas/                       统一事件和决策协议
├── deployment/framework/          插件加载配置
├── HANDOFF_CHECKLIST.md           场景团队交付清单
├── MANIFEST.json                  文件哈希和排除项
├── requirements.txt               公共运行时依赖
└── requirements-training.txt      LoRA 训练依赖
```

## 从零部署完整交通系统

全新云服务器和两台 Jetson 的安装、CUDA `llama-server` 编译、真实模型下载、
节点启动及四分区验收见
[`docs/从零部署真实交通系统.md`](docs/从零部署真实交通系统.md)。
完整部署不依赖原 ASTGCN 工程，也不使用无权重冒烟结果。

## 先验证 SDK

要求 Python 3.8 及以上。运行时使用 `jsonschema` 校验插件自有数据结构。

```bash
cd cloud_edge_scene_sdk
python -m pip install -r requirements.txt
python -m scene_plugin_template.smoke_test
```

看到 `"status": "smoke_test_passed"` 表示以下公共链路正常：

- 正常网络同步云复核；
- 断网时本地自治；
- 两个边缘事件发生共享资源冲突；
- 云端完成冲突消解。

这只能证明框架接通，不代表场景模型已经满足准确率要求。

如需验证双 Jetson 汇聚、冲突协调和断网补传，再执行：

```bash
python -m pip install -e ./scenes/freeway_traffic --no-deps
python -m freeway_traffic_scene.smoke_test
```

学校三机部署与逐文件说明见
`scenes/freeway_traffic/README.md`。其中“便携冒烟”只验证接口；正式验收必须运行
`run_full_acceptance.py`，输出会记录每个真实模型是否执行、四边汇聚状态和最终回填。

## 安装默认模型

模型包同时定义未经本项目微调的云端 `qwen3.5:9b` Teacher，以及边缘 0.8B 共享基座。通用蒸馏候选尚未通过发布门禁，交通场景使用的是独立训练和验收的场景模型，不能把它写成通用能力已达标：

```bash
python -m model_bundle.install_models --all
```

云端模型从官方 Ollama Registry 拉取；边缘模型从本项目 GitHub Release 下载。安装器会核对模型字节数、SHA-256 和 Ollama 清单摘要。只安装或验证其中一个模型的命令见 `model_bundle/README.md`。

## 场景接入步骤

### 1. 复制并改名插件

将 `scene_plugin_template/` 改为本场景名称，例如：

```text
industrial_plugin/
power_grid_plugin/
```

同步修改：

- 插件类名；
- `scene` 和 `aliases`；
- `event_types`；
- `data_schema.json` 的 `$id` 和业务字段；
- `policy_version`；
- `deployment/framework/scene_plugins.json` 中的 `spec`。

### 2. 实现模型输出适配

场景感知模型不输出公共`SemanticEvent`，只输出自己的结果。下面的异常检测数据只是示例，没有伪造缺陷类型、检测框或统一风险字段；其他场景可完全替换`data`结构。

外部请求统一使用事件信封：

```json
{
  "specversion": "1.0",
  "id": "industrial_event_0001",
  "source": "urn:edge:industrial_edge_01:anomaly-detector",
  "type": "com.example.industrial.anomaly-map.v1",
  "scene": "industrial_anomaly",
  "edgeid": "industrial_edge_01",
  "time": "2026-07-17T00:00:00Z",
  "dataschema": "https://example.local/schema-v1.json",
  "datacontenttype": "application/json",
  "data": {"anomaly_score": 0.94, "heatmap": {"uri": "...", "shape": [64, 64]}}
}
```

公共信封只约束身份、来源、类型、时间和 schema URI，`data` 完全由场景团队的 `data_schema.json` 定义。插件必须实现：

1. `payload_schema()`：返回带 `$id` 的场景 JSON Schema；
2. `normalize()`：把已校验的 `SceneEventEnvelope.data` 映射为内部 `SemanticEvent`；
3. `event_types`：声明插件接受的事件类型。

内部 `SemanticEvent` 才需要 `scope / prediction / risk / uncertainty / timing / evidence / candidate_actions`，因为公共调度和冲突协调依赖这些语义。它由插件生成，不是对场景模型输出格式的要求。

事件类型、`dataschema` 或数据结构不匹配时会直接报错，不会猜字段或回退到旧格式。模板的完整例子见 `scene_plugin_template/sample_event.json`、`data_schema.json` 和 `plugin.py`。

#### 旧项目通过文件桥接

旧模型工程不需要安装或导入本框架。让模型把完整事件 JSON 写入专用 `inbox`，再在独立 Python 环境启动常驻桥接器：

```bash
python -m cloud_edge_framework.file_bridge watch \
  --input-dir runtime/file_bridge/inbox \
  --state-dir runtime/file_bridge/state \
  --schema-dir scene_plugin_template \
  --edge-base-url http://127.0.0.1:18101
```

桥接器会先校验公共信封和场景 `data_schema.json`，合法事件进入 SQLite Outbox 后发送；非法文件进入隔离目录，断网事件在恢复后补传。详细口径见 `FILE_BRIDGE.md`。

### 3. 接入边缘和云端模型

必须替换模板中的两个占位方法：

- `edge_decide()`：调用边缘轻量模型，弱网和断网时仍可运行；
- `cloud_decide()`：调用云端专家模型或场景协调器。只有定义了目标函数、约束并经过相应基准验证的实现，才能称为全局优化器。

`health()` 中的 `template_mode` 在正式实验前应改为 `false`，并报告真实模型名称与版本。

公共模型调用统一读取 `runtime.edge.json` 或 `runtime.cloud.json`，场景插件不再分别实现 llama.cpp、Ollama 和 OpenAI-compatible HTTP 细节。先校验配置，再按需执行真实探测：

```bash
python -m edge_llm_factory verify-runtime \
  --config scene_adapter_template/runtime.edge.json
python -m edge_llm_factory probe-runtime \
  --config scene_adapter_template/runtime.cloud.json \
  --prompt 'Return one short decision.'
```

provider 会统一返回文本、总时延、输入/输出 token 和后端分项时延。动作模型还必须经过 `ConfiguredActionClient`，强制 `thinking=false`、`max_output_tokens=1` 和合法动作 token：

```python
from edge_llm_factory.runtime import ConfiguredActionClient

client = ConfiguredActionClient.from_path("scene_adapter_template/runtime.edge.json")
result = client.predict(prompt, {"A": "A", "G": "G", "H": "H"})
```

OpenAI-compatible 密钥只能通过 `authentication.api_key_env` 指定的环境变量读取，不能写入 JSON。模板中的 endpoint 和模型名是占位值，正式部署时按设备修改。

#### 边缘 Qwen 适配器

执行基座验证或 LoRA 流水线前，先安装训练依赖：

```bash
python -m pip install -r requirements-training.txt
```

SDK 不分发 1.5 GB 权重。框架维护者先把已验证的共享纯文本快照放到 `models/base/qwen35_0_8b_text_2fc06364/`，场景团队不得直接使用官方多模态快照，也不得自行更换同名模型。接入前执行：

```bash
python -m edge_llm_factory verify-text-base \
  --base edge_llm/base_manifest.json \
  --snapshot-manifest edge_llm/text_snapshot_manifest.json \
  --snapshot models/base/qwen35_0_8b_text_2fc06364
```

上游官方快照含视觉塔和 MTP 辅助头，SDK 中的文本快照清单记录了完整派生过程和逐文件哈希；训练流水线会再次验证权重中不存在多模态参数。详细说明见 `edge_llm/README.md`。

SDK 同时分发 293 条、50,135 token 的通用校准文本，只包含数学、代码和自然语言推理，交通等场景样本为 0。它用于建立无场景 Q4/Q5 基线，也可以作为各场景量化校准的通用核心：

```bash
python -m edge_llm_factory build-imatrix --help
python -m edge_llm_factory export-gguf --help
python -m edge_llm_factory benchmark-runtime --help
```

importance matrix 与生成它的模型权重严格绑定。LoRA 合并后必须针对合并模型重新计算，不能直接套用无场景基座矩阵。最终场景 GGUF 仍需在未见任务测试集和目标边缘设备上验收；通用基线不能替代场景准确率证据。完整结果见 `edge_llm/general_compression.md`。

SDK 还提供场景无关的 9B Teacher -> 0.8B 多 token 行为蒸馏入口。构建器会排除冻结评测 prompt，并对数学答案、选择题答案和代码执行结果进行自动验收：

```bash
python -m edge_llm_factory build-general-kd-source --help
python -m edge_llm_factory build-general-kd --help
python -m edge_llm_factory train-general-kd --help
python -m edge_llm_factory gate-general-kd --help
```

`build-general-kd-source` 只读取锁定 revision 的独立训练/验证分片，并按标准化 prompt 拦截冻结测试泄漏。候选 F16 必须先通过 `gate-general-kd` 的完整样本、宏平均和分项回归门槛，才能继续量化。通用蒸馏只更新共享候选基座，不会自动替代当前共享基座或任一场景 LoRA。当前实测、负结果和未达标项见 `edge_llm/general_distillation.md`。

场景团队不修改公共基座权重，只提交以下内容：

`package_spec.json` 的 `input_contract` 必须绑定外部 `event_type`、场景 `data_schema` 和版本化 `context_encoder`。感知模型可以处理任意模态，但当前共享 Qwen 是纯文本基座，`direct_media_to_llm` 必须为 `false`。

1. Teacher 校验后的 train/val/test JSONL；
2. PEFT LoRA，权重必须是 `adapter_model.safetensors`；
3. `action_mapping.json`，把 A-H 动作槽映射到场景候选动作；
4. `package_spec.json`，锁定数据 ID、指标证据、GGUF 哈希和验收门槛。

公共流水线执行：

```text
锁定基座 -> LoRA/QLoRA SFT -> 未见测试集评估 -> 合并 -> GGUF 量化 -> 发布校验
```

先检查配置，不启动训练：

```bash
python -m edge_llm_factory run-pipeline \
  --config scene_adapter_template/pipeline.json \
  --project_root . \
  --dry_run
```

基座 revision、tokenizer、LoRA 目标层或动作 token ID 任一不一致都会直接失败。第二阶段纠错蒸馏和 DPO 是候选实验，不会自动替换第一阶段；只有未见测试集更好时才发布。

模板流水线最后的 `publish_release` 会重新验证门禁，并把基座指纹、LoRA 包目录哈希、GGUF 哈希及门禁证据原子写入版本仓库。查询或回滚时会再次验完整性：

```bash
python -m edge_llm_factory release status \
  --registry runtime/edge_llm_releases.json
python -m edge_llm_factory release rollback \
  --registry runtime/edge_llm_releases.json
```

回滚只切换活动版本指针，不删除训练产物；目标文件被改动时会拒绝切换。要回滚到指定版本，可附加 `--release-id <id>`。

### 4. 定义数据面

在 `prepare_cloud_event()` 中决定上传什么：

- `summary`：风险、置信度、少量统计量；
- `feature`：任务相关压缩特征；
- `raw`：有争议时上传的局部原始证据或对象存储 URI。

RGB、红外、点云和波形不需要转成文本。场景插件可以上传特征张量、二进制编码或原始证据 URI，并在 `Evidence.codec` 中记录编码方式和版本。

`normalize()` 还可以在内部 `SemanticEvent.metadata` 中声明两个调度提示：

- `cloud_review_requested=true`：即使聚合风险较低，也必须保留局部异常、模型策略或业务规则提出的云复核意图；
- `minimum_evidence_level=summary|feature|raw`：指定这次复核最低需要哪一级证据。

它们不是外部场景 `data` 的固定字段，而是插件完成语义解释后交给公共调度器的内部信号。网络不可用时，框架先执行本地策略并持久排队；网络恢复后再补传指定证据。

### 5. 定义多节点关联和冲突

- `fuse_cloud_context()`：只融合业务上相关的设备、区域或任务；
- `action_conflict()`：判断两个动作是否争用同一资源或互不兼容；
- `resolve_action_conflict()`：给出可执行的统一动作。

不要把“两个模型输出不同”直接等同于冲突。只有时间窗口相关、任务语义相关，并且共同影响实体或执行资源时才算决策冲突。

## 启动独立边云服务

先启动云端：

```bash
python -m cloud_edge_framework.cloud_service \
  --project_root . \
  --config deployment/framework/cloud_service.json
```

再启动边缘端。将 `edge_service.json` 中的 `cloud.base_url` 改为真实云端地址：

```bash
python -m cloud_edge_framework.edge_service \
  --project_root . \
  --config deployment/framework/edge_service.json
```

边缘 `/ready` 不以云端在线为前提；断网时仍可接收事件并执行本地策略。边缘主动探测云端网络，调用方不能手工声明网络良好。具有多边缘汇聚规格的每个在线样本都会先持久化到 SQLite Outbox，由后台发送器上报；其他需要复核的事件也使用同一队列。边缘接口先返回临时结果，不等待摘要到云确认。普通 Outbox 在部分汇聚或本地超时后按时结束；同库中另有带截止时间的低频 reconciliation 子队列追踪迟到成员，完整新版本会回填原生命周期为权威 final，无新版本则到期清除。云端恢复后自动补传，异步协调结果会同时写入边缘和云端反馈样本库，保留边缘初判与云端修正，供后续蒸馏或策略更新使用。

提交样例事件：

```bash
python -m cloud_edge_framework.ingest_client \
  --event scene_plugin_template/sample_event.json \
  --edge_base_url http://127.0.0.1:18101
```

健康与指标：

```bash
curl http://127.0.0.1:18101/health
curl http://127.0.0.1:18101/api/v1/framework/metrics
curl http://127.0.0.1:18101/api/v1/framework/outbox
curl http://127.0.0.1:18100/health
```

### 角色接口

| 角色 | 方法 | 路径 | 用途 |
| --- | --- | --- | --- |
| 边缘 | POST | `/api/v1/collaboration/decide` | 场景信封接入、边缘决策和调度 |
| 边缘 | POST | `/api/v1/collaboration/flush-pending` | 运维人员立即触发一次 Outbox 重放 |
| 边缘 | GET | `/api/v1/framework/outbox` | 查看 pending/inflight/completed 状态 |
| 边缘 | GET | `/api/v1/collaboration/reviews` | 查询边缘临时判断到云端最终结果的复核生命周期 |
| 边缘 | GET/POST | `/api/v1/collaboration/monitoring` | 查询监测状态、写入结果或参考分布 |
| 边缘 | GET | `/api/v1/collaboration/routing-dataset` | 导出学习式路由训练数据 |
| 云端 | POST | `/api/v1/collaboration/cloud-decision` | 单事件云端复核 |
| 云端 | POST | `/api/v1/collaboration/coordinate` | 多事件融合和冲突协调 |
| 云端 | POST | `/api/v1/collaboration/aggregate` | 按场景关联键提交一个多边缘语义事件 |
| 云端 | POST | `/api/v1/collaboration/aggregate/flush` | 触发超时部分汇聚；部分结果不具备完整云确认，迟到成员可触发新版本 |
| 云端 | PUT/GET | `/api/v1/evidence/{sha256}` | 上传或查询经过哈希校验的大证据文件 |
| 云端 | GET/POST | `/api/v1/collaboration/feedback` | 查询或写入纠错反馈 |
| 两端 | GET | `/health`、`/ready` | 健康和就绪状态 |
| 两端 | GET | `/api/v1/framework/metrics` | 统一 JSON 指标 |
| 两端 | POST | `/api/v1/collaboration/plugins/reload` | 原子热重载插件 |

边缘不会暴露云端复核接口，云端也不会暴露外部场景接入接口。重复请求通过 SQLite 幂等缓存返回第一次完成的结果。旧的 `cloud_edge_framework.server` 仅保留为单进程开发兼容入口，不作为正式部署。

## 闭环基准

```bash
python -m cloud_edge_framework.benchmark_services \
  --event scene_plugin_template/sample_event.json \
  --edge_base_url http://127.0.0.1:18101
```

脚本为每次请求生成唯一事件 ID，避免幂等缓存把时延测低。

## 接入完成标准

场景团队至少应证明：

1. 正常网络、弱网、断网三条路径都能运行；
2. 断网时基本业务不依赖云端；
3. 上传字节数和闭环时延有真实测量；
4. 边缘和云端模型效果使用同一测试集比较；
5. 至少准备一组真实业务冲突，并展示协调前后的动作；
6. 模型、特征编码和策略都有明确版本；
7. 报告中的准确率、模型一致率和控制收益分别统计，不能混写。
