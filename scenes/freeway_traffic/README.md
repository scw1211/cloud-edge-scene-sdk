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

`current_state_perception_runtime.py` 常驻加载 NPZ。每个 sample 只索引一次窗口、
反归一化并计算当前态势，然后按冻结的 METIS4 映射产生四个分区事件。测试脚本
按 sample 顺序提交；前一窗口取得四个本地响应后立即推进下一窗口，不等待它的
异步云端 final。这是连续窗口的加速回放，不包含传感器接入、原始 CSV 解析和
线上 12 步缓冲时间，冷加载也必须单独报告。

数据在链路上分三层，不能都叫“上传原始样本”：

1. NPZ 到感知进程：完整 `[170,3,12]` 窗口，只在 Jetson 内存中使用。
2. 感知进程到本机 edge `/decide`：四个语义事件，包含区域摘要、控制能力、
   top-10 风险节点和它们的 12 步速度历史，不包含完整 170 节点三通道 tensor。
3. 138 到 160：再裁剪为区域摘要或编码特征；默认不带 `raw_evidence`，Outbox
   后台批量发送，云端持久接收后立即 ACK，再做四分区汇聚和 final 回填。

普通本地动作在持久 handoff/Outbox 接受边界后即可返回，摘要发送失败会重试。
动作后果风险高、模型高不确定、跨区冲突或策略强制审核命中时，网络和 deadline
允许才同步等待云端；需要云确认的动作在权威 final 前不会被授权。

正式基准的共同 T0 位于常驻、预热完成后，紧挨一个已到齐的 12 步窗口处理前：

- `local_actionable_ms`：T0 到四个 compact `/decide` 响应全部返回；
- `business_completion_ms`：本地已授权动作止于本地响应，需要云确认的动作止于
  权威 final；
- `global_authoritative_final_ms`：T0 到四个 review 都完成权威回填；
  `partial_final` 和 `local_only_timeout` 不算权威完成。

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
  --require-risk-coverage \
  --require-complete-final \
  --output /tmp/pems08-current-state-e2e.json
```

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
