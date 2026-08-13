# 真实交通云边系统

这个目录是交通场景的完整可运行包，不是只有规则的接口演示。正式链路包含：

```text
PEMS08 最近观测窗口
  → 默认纯 NumPy 当前态势风险（可选 ASTGCN 未来预测）
  → 四个 METIS 区域摘要
  → 当前态势 Student + 验证集增益路由 + Edge-Qwen 0.8B 单动作 token
  → 公共在线调度 + Outbox 后台批量传输
  → 云端按样本四边汇聚 + ExtraTrees 一次批量协调
  → 可选 Qwen3.5 9B 结构化复核
  → provisional→final 回填与冲突审计
```

常态实时链路使用不加载 PyTorch/ASTGCN 的当前态势路径：直接用最近 12 个
观测时间步计算节点和区域风险，然后进入同一套边缘决策、Outbox、云端汇聚和
final 回填链路。该路径做当前状态识别，不能与 ASTGCN 的未来预测准确率混为
一谈。`send_real_partitions.py` 默认使用该路径；传入
`--perception-mode astgcn` 可恢复原预测链路。

当前态势 Student 先处理普通事件。Edge-Qwen 只接管验证集上证明相对 Student
有净纠错收益的可观察子群；路由不再由“风险高”单独触发。在正常、弱网、断网
三种网络状态组成的独立测试集中，选择率为 14.71%，净纠错 267、改错 0；其中
主要收益来自断网状态下的安全动作适配，不能写成普通联网交通上的普遍增益。
固定的正常网络 `test[100:200]` 连续窗口中有 30/400 个事件自然满足验证集增益
门槛；短片段仍可能全部由 Student 处理，不能用单个样本的选择数代表整体路由。
current-state v2.0.2 允许 Qwen 在当前仍为 low、但 Student 已授权非执行型
`traffic_advisory` 时发布提前预警；它不会由此获得 VSL、匝道、绕行或跨区控制权。
旧 v9 模型与新上下文编码不兼容，框架会在新发布包未激活时自动禁用这份增益
路由，避免误调用。

`freeway_traffic_scene/`仍保留一个无权重便携冒烟测试，只用于检查 SDK
协议和服务能否启动。比赛结果必须来自本目录的真实全链路脚本，不能引用便携
冒烟的规则结果。

## 文件是做什么的

| 文件或目录 | 作用 |
| --- | --- |
| `run_full_acceptance.py` | 单机启动真实 ASTGCN、Edge-Qwen、边缘服务和云服务；强制检查四分区到齐、全部 final 回填和残余冲突为零 |
| `install_full_assets.py` | 下载大文件，逐字节核对大小与 SHA-256，并把 Edge-Qwen 发布到本地活动版本仓库 |
| `asset_catalog.json` | 所有真实资产的版本、位置、字节数、SHA-256 和下载地址 |
| `send_real_partitions.py` | 发送指定分区；默认运行当前态势直通模式，两台机器分别使用 `0,1` 和 `2,3`，加 `--perception-mode astgcn` 可运行原预测链路 |
| `benchmark_real_current_state_e2e.py` | 常驻加载 PEMS08，按连续窗口顺序加速回放；统一测量本地可执行、业务完成、后台摘要字节和四分区权威 final |
| `prepare_metis_partition_data.py` | 部署前按照冻结的 METIS4 映射生成四份区域 NPZ 和 SHA-256 清单；运行时禁止重新划分 |
| `run_partitioned_current_state_edges.py` | 在单台 Nano 上启动四个独立常驻进程；每个进程只加载自己的区域 NPZ、生成一个事件并并发调用公共边缘服务 |
| `traffic_system/current_state_perception_runtime.py` | 纯 NumPy 当前态势感知：根据最近 12 步流量、占有率和速度生成风险事件，不加载 PyTorch 或 ASTGCN |
| `verify_real_two_edge.py` | 从两台 Jetson 和云服务器读取 review、aggregation，检查四边完整闭环 |
| `freeway_traffic_full/plugin_impl.py` | 真实交通插件：按事件合同选择当前态势或 ASTGCN 的 Student、特征编码器和 ExtraTrees，再做拓扑融合与冲突处理 |
| `traffic_system/train_current_state_cloud_coordinator.py` | 按训练/验证/测试时间切分训练当前态势专用云端 ExtraTrees；测试准确率 73.73%，分组 bootstrap 95% CI 为 72.88%–74.66% |
| `freeway_traffic_full/edge_llm.py` | Edge-Qwen 的选择、动作 token 校验、安全回退和运行时隔离 |
| `traffic_system/traffic_perception_runtime.py` | 常驻加载 ASTGCN，一次推理生成四个区域事件 |
| `traffic_system/accept_traffic_framework.py` | 完整验收实现和指标采集，不是合成数据脚本 |
| `model/`、`lib/` | ASTGCN 网络结构、图卷积工具和邻接矩阵读取 |
| `assets/models/` | 已随 Git 提交的小型真实权重：ASTGCN、Student、defer gate、编码器和 ExtraTrees |
| `assets/edge_llm/adapter_package_current_state_v2/` | 当前态势 Edge-Qwen v2 候选发布包、动作映射和本机实测证据；Jetson 门禁通过后才能作为正式性能结果 |
| `assets/downloads/` | 安装器下载的 PEMS08 推理数组与 0.8B Q6 GGUF；不进入 Git |
| `deployment/full/scene_plugins_edge.json` | 真实边缘模型和插件路径 |
| `deployment/full/scene_plugins_cloud.json` | 两套真实云端 ExtraTrees、各自绑定的特征编码器和拓扑路径；云端按事件合同自动选择，禁止混用 |
| `deployment/full/edge_service.json` | 实验室边缘服务配置模板 |
| `deployment/full/edge_service_qwen9b.json` | 9B 异常增强实验专用边缘配置，放宽云端等待时间 |
| `deployment/full/cloud_service.json` | 常态云服务配置，9B 关闭 |
| `deployment/full/cloud_service_qwen9b.json` | 可疑事件增强配置，打开 Qwen3.5 9B |
| `configurations/PEMS08_astgcn.conf` | SDK 内部可迁移的 ASTGCN 配置 |
| `evidence/本机完整系统实测.md` | 已完成的真实权重验证、诚实边界和实验室待测项 |

## PEMS08 时间流与样本上传边界

运行资产不是现场读取原始 CSV 后再切窗，而是已经按时间顺序切好的归一化
`split_x`。一个 sample 是 `[170, 3, 12]`：170 个检测点、流量/占有率/速度
三个通道、12 个 5 分钟步，也就是最近 60 分钟。float32 名义输入为 24,480
字节。sample `i+1` 是时间上紧接 sample `i` 的下一个滑动窗口。

旧兼容基准由`current_state_perception_runtime.py`常驻加载完整 NPZ，每个 sample
索引一次完整窗口，再按冻结的 METIS4 映射产生四个分区事件。该模式便于逐字段
回归，但只能称为“单进程四逻辑区域”，不能称为四个边缘节点。

正式单板分布式输入先运行`prepare_metis_partition_data.py`，离线生成四份区域
NPZ。四个常驻进程分别只加载一份区域数据；每个窗口各自产生一个事件并并发提交。
METIS 不在在线路径重新计算，`node_id → partition_id → edge_id`由带 SHA-256 的
清单锁定。脚本支持两种可对照模式：兼容模式由四个感知进程共用一个边缘服务；
推荐的仿真模式为每个分区启动独立边缘服务端口、独立 Outbox、Review、幂等库和
监测库，只共享一块 Nano、一个可选 Edge-Qwen 服务和同一云端。这是“一台物理
Nano、四个逻辑边缘节点”，不能写成四台物理板卡。

准备数据：

```bash
python scenes/freeway_traffic/prepare_metis_partition_data.py
```

运行四个独立逻辑边缘节点并验收10个连续窗口。`--cloud-url`必须填写从 Nano
实际可访问的云端地址；19101—19104 仅用于隔离测试，不占用正式18101：

```bash
python scenes/freeway_traffic/run_partitioned_current_state_edges.py \
  --project-root . \
  --manifest scenes/freeway_traffic/runtime/pems08_metis4_partitions/manifest.json \
  --launch-isolated-edge-services \
  --edge-port-base 19101 \
  --cloud-url http://192.168.31.160:18100 \
  --sample-start 100 \
  --sample-stop 110 \
  --output /tmp/pems08-four-edge-processes.json
```

如需复现旧的共享服务模式，去掉上述三个选项并传入
`--edge-url http://127.0.0.1:18101`。共享模式只用于消融，不能作为四个完整逻辑
边缘节点的部署证据。

输出必须同时满足：4个不同 PID、170个节点恰好覆盖一次、每进程每窗口一个事件、
4/4 权威 final；独立模式还必须满足4个服务 PID、4个端口和4套持久化状态互不
共享。任何一项缺失都会失败关闭，不能把2/4部分汇聚写成完整结果。测试脚本逐窗口
测量：四个边缘并发提交，观察本窗口权威 final 后再进入下一窗口，避免把后续样本的
运行时间错误计入前一窗口。这是连续窗口回放，不包含传感器接入、原始 CSV 解析和
线上12步缓冲时间，冷加载也必须单独报告。

数据在链路上分三层，不能都叫“上传原始样本”：

1. 区域 NPZ 到感知进程：只读取本区域约42—43个节点的 `[N,3,12]` 窗口；完整
   `[170,3,12]` 仅存在于离线预切分工具中。
2. 感知进程到本机 edge `/decide`：四个语义事件，包含区域摘要、控制能力、
   top-10 风险节点和它们的 12 步速度历史，不包含完整 170 节点三通道 tensor。
3. 138 到 160：再裁剪为区域摘要或编码特征；默认不带 `raw_evidence`，Outbox
   后台批量发送，云端持久接收后立即 ACK，再做四分区汇聚和 final 回填。

普通本地动作在持久 handoff/Outbox 接受边界后即可返回，摘要发送失败会重试。
交通连续监测与四边缘性能脚本对 `/decide` 使用
`Prefer: return=minimal, respond-async`：即使动作后果风险高、模型高不确定、
跨区冲突或策略强制审核命中，也先返回无云确认的 provisional，后台继续取得
权威 final。这里异步的是响应和复核执行，不是放宽动作安全；需要云确认的动作在
权威 final 前仍不会被授权。不带 `respond-async` 的兼容调用保留同步等待能力。

连续时间流中，四个分区通常会紧邻到达。常态完整配置对尚未形成结果的完整聚合
每 25 ms 复查一次，对已收到 `partial_final` 的结果每 50 ms 复查一次，使迟到
分区补齐后能尽快取得新的完整版本；这只缩短查询间隔，不改变 10 s 的完整聚合
最大等待、5 s 的 reconciliation 复查间隔或 60 s 的 reconciliation 截止边界。
更短的轮询会提高聚合等待期间的云端结果查询频率，部署时应结合并发样本数和云端
查询负载监控；非连续流或大规模部署可适当调大 `waiting_poll_seconds` 和
`partial_poll_seconds`，但不能借此放宽云确认和动作授权边界。

当前态势链路区分“建议复核”和“必须同步”：Student 置信度低于 0.75 或与当前
规则不同会设置宽泛的 `requires_review`，用于特征证据和异步全局复核；只有
Student 最高类概率低于 `current_state_sync_confidence_threshold`（默认 0.50）
且与规则不同、预测集多值或 defer 命中，才设置
`requires_synchronous_review` 要求业务动作等待云确认。使用 provisional-first
接口时，它不再阻塞本地临时结果的返回。若本地 Qwen 的决策及动作语义与 Student
一致，可消解第一种 Student 不确定性；它不能消解预测集歧义、高动作风险、
跨区冲突、策略强制审核或动作的云确认要求。

正式基准的共同 T0 位于常驻、预热完成后，紧挨一个已到齐的 12 步窗口处理前：

- `event_local_actionable_ms` / `event_business_completion_ms`：单个 METIS 区域
  的本地返回与路径声明的安全业务终点时延，覆盖连续段内全部风险层；
- `local_actionable_ms`：T0 到四个 compact `/decide` 响应全部返回；
- `business_completion_ms`：`edge_only`、`local_autonomy`、`cloud_async` 止于带
  action authorization 且摘要已持久入队的安全 provisional；其中高风险控制仍为
  deferred，不能执行。`cloud_sync` 才止于完整成员上的权威 final；
- `global_authoritative_final_ms`：T0 到四个 review 都完成权威回填；
  `partial_final` 和 `local_only_timeout` 不算权威完成。

测试进程为四个分区各保留一条 HTTP/1.1 连接和一个常驻 worker；预热、连续窗口
共用同一连接池，避免把每个 5 分钟窗口都错误建模为四次冷 TCP 连接。报告同时给出
单区域 event 口径和“四区域全部完成”的 sample 最大值口径；`<= 200 ms` 的完整性、
平均值门禁和达标率分别列出，不能用低风险子集代替全段指标。

异步上传字节从框架 `/metrics` 的 `distributions.async_http_*` 测量前后增量计算，
不能使用 `/decide` 返回瞬间尚未发生的后台传输字段。报告同时分开 scheduled
sync 与最终实际执行的 `cloud_sync`、`cloud_async`、`local_autonomy`，避免把同步
调度失败后的本地降级误写成云审成功时延。

使用固定连续段复测：

```bash
python scenes/freeway_traffic/benchmark_real_current_state_e2e.py \
  --project-root . \
  --edge-url http://127.0.0.1:19101 \
  --cloud-url http://云服务器地址:19100 \
  --sample-start 100 \
  --sample-stop 200 \
  --warmup-samples 86,125,0 \
  --require-qwen-selected \
  --require-qwen-accepted \
  --require-congestion-level-coverage \
  --require-complete-final \
  --output /tmp/pems08-current-state-e2e.json
```

## 设计依据和术语边界

这套调度采用“风险硬门 + 受验证增益和时延预算约束的选择性升级”，不是只凭一个
主观置信度阈值。Selective Classification 的 risk-coverage 口径提供了让不可靠
样本退出本地接受集合的理论背景；Learning to Defer 进一步要求考虑下游专家在
该类输入上的能力和咨询成本。NeurIPS 2023 的级联研究还表明，下游模型具有专长
差异、标签噪声或分布漂移时，单一 confidence 阈值可能明显次优。因此，本实现
把动作后果风险、预测集歧义、Student/规则分歧、验证集 Qwen 增益、云端可用性
和剩余 deadline 分开记录和判定；当前是有审计字段的规则/模型混合路由，不能写成
已经训练出的最优 Learning-to-Defer 策略。

基准参数 `--require-congestion-level-coverage` 只检查固定连续段是否出现 low、
medium、high、severe 四类拥堵层，不计算 selective-classification 的选择性风险、
本地接受覆盖率或 AURC。旧参数名 `--require-risk-coverage` 仅为命令兼容别名，
不能作为论文 risk-coverage 指标；正式能力评估需要另做带真值的选择性风险曲线。

- [Selective Classification for Deep Neural Networks, NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)
- [Consistent Estimators for Learning to Defer to an Expert, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html)
- [When Does Confidence-Based Cascade Deferral Suffice?, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1f09e1ee5035a4c3fe38a5681cae5815-Abstract-Conference.html)
- [Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge, ASPLOS 2017](https://doi.org/10.1145/3037697.3037698)

其中 Neurosurgeon 只作为“调度应测量设备、云端和传输成本”的边云背景；当前
链路不是 DNN 层级切分，不能声称复现了它的算法。

当前态势感知在四个分区内分别为所属节点计算与旧实现相同的轻量风险和稳定排序，
再只为每分区 Top-10（全样本最多 40 个）构造概率字典、观测字段和 12 步历史。
这是 late materialization 的工程类比：延后昂贵的宽对象构造，同时保持公开六位
分数、并列顺序和除时延字段外的事件语义字段兼容；它仍评估全部节点，也没有阈值
提前终止，因此不是 Fagin Threshold Algorithm，不能作为新的 Top-K 算法创新。
`test[100:200]` 的 400 个真实事件另做逐字段 golden 对比，只有
`inference_latency_ms` / `perception_ms` 被排除。

- [Materialization Strategies in a Column-Oriented DBMS, ICDE 2007](https://www.cs.umd.edu/~abadi/papers/abadiicde2007.pdf)
- [Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting, AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/3881)

ASTGCN 论文在这里用于说明 PEMS 的时空序列背景；默认纯 NumPy 当前态势路径不做
ASTGCN 未来预测，也不引用该论文证明 late materialization 的性能。

Outbox 链路提供的是本地持久接受、至少一次重试和接收端幂等组合，不宣称跨机
exactly-once。幂等键、请求指纹、原响应重放和 lease/fencing 属于故障恢复边界，
性能优化不能绕过这些字段。普通事件的云端摘要是异步全局汇聚；只有业务动作确实
等待权威云端结果的分支才称为同步审核。

自然连续流的冲突发生率与冲突阳性回归的解决率分开报告：固定段没有自然冲突时，
解决率为 `null` 而不是 100%；聚合样本缺失时，两种比率都为 `null`。竞赛要求的
“冲突解决成功率”必须来自另行构造的冲突阳性集，不能由自然段的 0/0 推导。

- [Life beyond Distributed Transactions, CIDR 2007](https://www.cidrdb.org/cidr2007/papers/cidr07p15.pdf)
- [Making retries safe with idempotent APIs, Amazon Builders' Library](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)
- [Debezium Outbox Event Router documentation](https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html)

## 一、安装

全新机器建议直接按照
[`../../docs/从零部署真实交通系统.md`](../../docs/从零部署真实交通系统.md)
执行。脚本会创建独立环境、安装依赖、编译 Jetson CUDA `llama-server`、
下载真实资产并做 SHA-256 核验。

三台机器都拉取同一个 Git commit：

```bash
git clone https://github.com/scw1211/cloud-edge-scene-sdk.git
cd cloud-edge-scene-sdk
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e ./scenes/freeway_traffic --no-deps
```

Jetson 不使用 WSL 的 `traffic` Conda 环境。Jetson 上先按 JetPack 版本安装
NVIDIA 提供的 PyTorch wheel，再安装：

```bash
python -m pip install -r scenes/freeway_traffic/requirements-runtime.txt
```

云服务器使用与本机 CUDA 匹配的 PyTorch。云端只做 ExtraTrees 协调时不需要
ASTGCN 和 PEMS08；如果在服务器上运行一键单机验收，则需要全部资产。

## 二、安装真实资产

每台 Jetson：

```bash
python scenes/freeway_traffic/install_full_assets.py --edge
```

这会校验随仓库交付的小型模型，下载并校验：

- `qwen35_0_8b_current_state_future_v2_q6_k.gguf`
- `PEMS08_r1_d0_w0_astcgn_multitask.npz`

云服务器如需 Qwen3.5 9B：

```bash
python scenes/freeway_traffic/install_full_assets.py --cloud
```

只复核现有文件、不重新下载：

```bash
python scenes/freeway_traffic/install_full_assets.py --edge --verify-only
python scenes/freeway_traffic/install_full_assets.py --cloud --verify-only
```

## 三、先在有 GPU 的机器跑完整系统

`llama-server`需要支持 Qwen3.5。填写本机实际二进制路径：

```bash
python scenes/freeway_traffic/run_full_acceptance.py \
  --llama-binary /实际路径/llama-server \
  --device cuda \
  --samples 0 \
  --edge-llm-mode primary
```

成功不是看进程是否退出，而是输出同时满足：

- `success_rate = 1.0`
- `edge_decision_paths.edge_qwen = 4`
- `edge_llm_acceptance_rate_when_selected = 1.0`
- `complete_aggregation_rate = 1.0`
- `final_completion_rate = 1.0`
- 云端聚合 `received_members` 包含四个边、`missing_members` 为空
- `residual_conflict_count = 0`

验证异常增强路径：

```bash
python scenes/freeway_traffic/run_full_acceptance.py \
  --llama-binary /实际路径/llama-server \
  --device cuda \
  --samples 0 \
  --edge-llm-mode primary \
  --with-cloud-qwen9b
```

9B 路径不属于 0.2 s 常态路径。它必须异步使用，报告中单独列出
`cloud_llm_review` 时延。

## 四、学校三机部署

机器角色：

| 机器 | 角色 | 分区 |
| --- | --- | --- |
| 云服务器 | 云端汇聚、ExtraTrees、可选 Qwen 9B | 无 |
| Jetson A | 真实感知、Edge-Qwen、边缘服务 | `0,1` |
| Jetson B | 真实感知、Edge-Qwen、边缘服务 | `2,3` |

先把两台 Jetson 的
`scenes/freeway_traffic/deployment/full/edge_service.json` 中
`cloud.base_url` 改成云服务器局域网地址。

云服务器：

```bash
python -m cloud_edge_framework.cloud_service \
  --project_root . \
  --config scenes/freeway_traffic/deployment/full/cloud_service.json
```

若本次专门测试 9B 异步增强，云端改用 `cloud_service_qwen9b.json`，两台边缘
同时改用 `edge_service_qwen9b.json`。常态时延测试不要打开 9B。

两台 Jetson 分别启动自己的 Edge-Qwen：

```bash
python -m edge_llm_factory serve-release \
  --registry scenes/freeway_traffic/runtime/edge_llm_release_store.json \
  --runtime-config scenes/freeway_traffic/deployment/full/edge_llm_runtime.json \
  --binary /实际路径/llama-server \
  --host 127.0.0.1 \
  --port 18190 \
  --context-tokens 128 \
  --threads 4 \
  --parallel 1 \
  --gpu-layers 99
```

再启动边缘服务：

```bash
python -m cloud_edge_framework.edge_service \
  --project_root . \
  --config scenes/freeway_traffic/deployment/full/edge_service.json
```

三机 NTP 同步后，选择一个共同实验编号和未来 5 秒的 Unix 毫秒时间。
Jetson A：

```bash
python scenes/freeway_traffic/send_real_partitions.py \
  --edge-url http://127.0.0.1:18101 \
  --partitions 0,1 \
  --sample-id 0 \
  --experiment-id lab01 \
  --start-at-ms 共同发送时刻 \
  --device cuda
```

Jetson B：

```bash
python scenes/freeway_traffic/send_real_partitions.py \
  --edge-url http://127.0.0.1:18101 \
  --partitions 2,3 \
  --sample-id 0 \
  --experiment-id lab01 \
  --start-at-ms 共同发送时刻 \
  --device cuda
```

任意能访问三台机器的电脑核验：

```bash
python scenes/freeway_traffic/verify_real_two_edge.py \
  --edge-a http://JetsonA地址:18101 \
  --edge-b http://JetsonB地址:18101 \
  --cloud http://云服务器地址:18100 \
  --experiment-id lab01 \
  --sample-id 0
```

只有脚本输出 `status=passed` 才表示实际三机闭环完成。

## 五、时延口径

一键单机验收为了检查四边功能，让四个逻辑边共享一个
`parallel=1` Edge-Qwen 服务。因此并发压力结果不能替代两台 Jetson 的硬件
结果。正式报告至少分开写：

- 无竞争单边请求：ASTGCN + Edge-Qwen + 边缘服务；
- 两台 Jetson 四分区同时到达：provisional 时延与全部摘要到达偏差；
- ExtraTrees 最终汇聚：provisional→final；
- Qwen 9B：异步复核，不计入常态 0.2 s；
- 断网：本地 provisional、持久队列、恢复后 final 与修正率。

当前状态直通路径在本机对 100 个真实测试样本做纯计算剖析：当前态势感知、
四事件规范化与编码、四次当前态势 Student 初判、使用当前态势专用 ExtraTrees
完成云端四事件批量融合，合计平均 16.40 ms、P95 17.16 ms。该数字不含 HTTP、
Outbox、多机汇聚等待和 final 回填，
只能证明算法计算段具备 0.2 s 预算，不能替代两台 Jetson 加云服务器的正式
端到端结果。完整 0.2 s 是否达标必须以三机同提交、真实网络下的输入到业务可
执行结果为准。

Student、defer gate 和交通 Edge-Qwen 都属于交通场景，不是公共框架要求每个
场景都实现一份。工业场景可以拥有自己的专业模型和局部决策器，公共框架只负责
事件信封、调度、可靠传输、汇聚、复核、冲突和模型发布。

## 六、交通联合目标与七项证据门禁

当前仓库已实现优化器、证据提取器和统一门禁，但正式云配置仍默认为 `shadow`，
且未随仓库提供当前提交在隔离 `active` 环境中的正式输出。因此第六项当前是
“可取证、未判定”，不能写成已实测通过。

云端在同一 `sample_id` 的区域决策完成后，可调用交通插件的
`optimize_global_plan()`。当前目标定义在
`assets/models/traffic_global_utility_v1.json`，对模型已经提出的有限动作组合计算：

```text
联合效用 = 延误代理改善收益
         + 排队代理改善收益
         + 吞吐代理改善收益
         + 风险缓解收益
         - 控制切换代价
         - 动作冲突惩罚
         - 设备能力越界惩罚
         - 安全约束越界惩罚
```

这些交通项使用每个区域摘要中的当前流量、占有率和速度计算无量纲代理量；它们
用于比较同一次完整汇聚内的有限候选动作，不把当前态势规则包装成未来交通预测。

四区域候选不超过配置上限时使用完整枚举，并记录目标文件 SHA、候选数量、基线
效用、选中效用和约束结果。正式云配置默认使用 `shadow`：计算和留证，但不改变
现有动作。只有经过留出集验证后才可改为 `active`；部分汇聚即使配置为 `active`
也不会应用联合动作。即使正式门禁通过，该结果也只能表述为“测量前锁定的有限
候选集合上的代理目标结果”，不能表述成物理道路网络、未来真实交通延误或无限
动作空间中的全局最优。

正式成员集合来自目标定义文件预登记的 `expected_members`。CloudService 使用云端
持久 aggregation lease 的成员上下文覆盖客户端 metadata，证据提取器和门禁再核对
expected/observed 成员与定义完全一致；客户端不能通过自报较小成员集合把部分汇聚
包装成“全局”。

正式取证不要直接把生产配置从 `shadow` 改成 `active`。使用隔离端口、独立数据库和
`deployment/full/scene_plugins_cloud_global_objective_active.json` 启动旁路云服务，
在预先固定的留出样本上完成真实 4/4 汇聚后，再从端到端原始结果提取第六项证据：

```bash
python scenes/freeway_traffic/extract_global_objective_evidence.py \
  --benchmark-json /绝对路径/真实端到端结果.json \
  --objective-definition scenes/freeway_traffic/assets/models/traffic_global_utility_v1.json \
  --sample-plan /绝对路径/测量前锁定的全局目标样本计划.json \
  --dataset-id /预注册留出集ID/ \
  --model-id /云端协调模型ID/ \
  --hardware-id /云服务器及运行时ID/ \
  --run-id /唯一实验ID/ \
  --output /绝对路径/交通全局目标证据.json
```

样本计划必须先按`schemas/traffic_global_objective_sample_plan.schema.json`锁定
样本清单、预期数量和慢样本阈值。提取器将成功、慢、部分汇聚和失败样本全部写入
`attempts`；只有完整权威汇聚进入`records`。样本缺失、重复或偏离计划会立即停止；
证据完整但含失败/部分汇聚时，统一门禁会如实判为未通过，不能只提交成功子集。
`shadow`结果可以被原样导出用于诊断，但统一门禁会拒绝把它认作第六项达标证据。

七项非工业指标统一使用 `scripts/evaluate_competition_targets.py` 判定，证据格式、
SHA 和口径见 `docs/七项统一证据门禁.md`。缺失当前提交、当前模型或当前硬件的
证据时，工具会输出“未测量”，不会自动沿用历史实验。
