# 真实交通云边系统

这个目录是交通场景的完整可运行包，不是只有规则的接口演示。正式链路包含：

```text
PEMS08 时空窗口
  → 联合 ASTGCN 预测与节点/区域风险
  → 四个 METIS 区域摘要
  → 交通 Student + defer gate + Edge-Qwen 0.8B 单动作 token
  → 公共在线调度 + Outbox 后台批量传输
  → 云端按样本四边汇聚 + ExtraTrees 一次批量协调
  → 可选 Qwen3.5 9B 结构化复核
  → provisional→final 回填与冲突审计
```

`freeway_traffic_scene/`仍保留一个无权重便携冒烟测试，只用于检查 SDK
协议和服务能否启动。比赛结果必须来自本目录的真实全链路脚本，不能引用便携
冒烟的规则结果。

## 文件是做什么的

| 文件或目录 | 作用 |
| --- | --- |
| `run_full_acceptance.py` | 单机启动真实 ASTGCN、Edge-Qwen、边缘服务和云服务；强制检查四分区到齐、全部 final 回填和残余冲突为零 |
| `install_full_assets.py` | 下载大文件，逐字节核对大小与 SHA-256，并把 Edge-Qwen 发布到本地活动版本仓库 |
| `asset_catalog.json` | 所有真实资产的版本、位置、字节数、SHA-256 和下载地址 |
| `send_real_partitions.py` | 在一台 Jetson 上运行真实 ASTGCN，并发送指定分区；两台机器分别使用 `0,1` 和 `2,3` |
| `verify_real_two_edge.py` | 从两台 Jetson 和云服务器读取 review、aggregation，检查四边完整闭环 |
| `freeway_traffic_full/plugin_impl.py` | 真实交通插件：校验输入、Student 决策、特征压缩、拓扑融合、ExtraTrees 协调和冲突处理 |
| `freeway_traffic_full/edge_llm.py` | Edge-Qwen 的选择、动作 token 校验、安全回退和运行时隔离 |
| `traffic_system/traffic_perception_runtime.py` | 常驻加载 ASTGCN，一次推理生成四个区域事件 |
| `traffic_system/accept_traffic_framework.py` | 完整验收实现和指标采集，不是合成数据脚本 |
| `model/`、`lib/` | ASTGCN 网络结构、图卷积工具和邻接矩阵读取 |
| `assets/models/` | 已随 Git 提交的小型真实权重：ASTGCN、Student、defer gate、编码器和 ExtraTrees |
| `assets/edge_llm/adapter_package/` | 已通过门禁的交通 LoRA 包、动作映射和实测清单 |
| `assets/downloads/` | 安装器下载的 PEMS08 推理数组与 0.8B Q6 GGUF；不进入 Git |
| `deployment/full/scene_plugins_edge.json` | 真实边缘模型和插件路径 |
| `deployment/full/scene_plugins_cloud.json` | 真实云端 ExtraTrees、特征编码器和拓扑路径 |
| `deployment/full/edge_service.json` | 实验室边缘服务配置模板 |
| `deployment/full/edge_service_qwen9b.json` | 9B 异常增强实验专用边缘配置，放宽云端等待时间 |
| `deployment/full/cloud_service.json` | 常态云服务配置，9B 关闭 |
| `deployment/full/cloud_service_qwen9b.json` | 可疑事件增强配置，打开 Qwen3.5 9B |
| `configurations/PEMS08_astgcn.conf` | SDK 内部可迁移的 ASTGCN 配置 |
| `evidence/本机完整系统实测.md` | 已完成的真实权重验证、诚实边界和实验室待测项 |

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

- `qwen35_0_8b_freeway_action_token_v9_text_only_q6_k.gguf`
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

Student、defer gate 和交通 Edge-Qwen 都属于交通场景，不是公共框架要求每个
场景都实现一份。工业场景可以拥有自己的专业模型和局部决策器，公共框架只负责
事件信封、调度、可靠传输、汇聚、复核、冲突和模型发布。
