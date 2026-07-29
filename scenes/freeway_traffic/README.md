# 交通双边缘参考场景

这个目录让交通场景随 SDK 一起交付，但不把交通代码塞进公共框架。
它可以独立验证两台边缘节点上报摘要、云端按同一样本汇聚、冲突消解、
断网持久排队和恢复后补传。

默认演示不需要 ASTGCN、Student、ExtraTrees 或 Qwen 权重。样例中的风险摘要
相当于专业模型已经输出的结果，插件用确定性规则产生动作。因此它证明的是
框架闭环可运行，不是模型准确率已经达标。

## 每个文件做什么

| 文件或目录 | 作用 |
|---|---|
| `freeway_traffic_scene/plugin.py` | 校验交通模型输出，转换成公共语义事件，定义边缘动作、云端动作和跨边上下文融合 |
| `freeway_traffic_scene/data_schema.json` | 定义交通场景自己的 `data` 字段；不是公共框架强制格式 |
| `freeway_traffic_scene/smoke_test.py` | 不启动 HTTP 服务，快速验证双边汇聚、冲突消解和第一边最终回填 |
| `freeway_traffic_scene/school_demo.py` | 三台机器启动后，同时向两个边缘服务发送事件并等待两个 final |
| `samples/edge_a_event.json` | 第一台边缘的严重风险摘要 |
| `samples/edge_b_event.json` | 第二台边缘的高风险摘要；与第一边共享道路资源，故意产生动作冲突 |
| `deployment/cloud_service.json` | 云端服务配置，默认端口 `18100` |
| `deployment/edge_a_service.json` | 第一台边缘配置，默认端口 `18101` |
| `deployment/edge_b_service.json` | 第二台边缘配置，默认端口 `18102` |
| `deployment/scene_plugins.json` | 告诉运行时加载哪个交通插件 |
| `model_assets.example.json` | 记录真实模型放在哪台机器、路径和哈希；不保存权重 |
| `pyproject.toml` | 让交通插件作为可选 Python 包安装 |

## 单机先验证

在 SDK 根目录执行：

```bash
python -m pip install -e .
python -m pip install -e ./scenes/freeway_traffic --no-deps
python -m freeway_traffic_scene.smoke_test
```

出现 `traffic_smoke_test_passed` 表示插件和公共运行时闭环正常。

## 学校三机部署

三台机器都拉取同一个 SDK commit。Jetson 不需要 WSL 的 `traffic` Conda
环境；只要 Python 3.9 及以上，并安装 SDK 与交通可选包即可。

云端服务器执行：

```bash
python -m cloud_edge_framework.cloud_service \
  --project_root . \
  --config scenes/freeway_traffic/deployment/cloud_service.json
```

把两个边缘配置中的 `cloud.base_url` 从 `127.0.0.1` 改成云服务器局域网地址。
第一台 Jetson 执行：

```bash
python -m cloud_edge_framework.edge_service \
  --project_root . \
  --config scenes/freeway_traffic/deployment/edge_a_service.json
```

第二台 Jetson 执行：

```bash
python -m cloud_edge_framework.edge_service \
  --project_root . \
  --config scenes/freeway_traffic/deployment/edge_b_service.json
```

在能访问两台 Jetson 的机器执行：

```bash
python -m freeway_traffic_scene.school_demo \
  --edge-a http://第一台Jetson地址:18101 \
  --edge-b http://第二台Jetson地址:18102 \
  --cloud http://云服务器地址:18100
```

脚本每次生成新的事件编号，避免服务端幂等缓存影响重复实验。成功标准是两边
复核生命周期都变成 `completed`，云端聚合状态为 `completed`，且最终冲突数为零。

## 接入真实交通模型

真实 ASTGCN 仍在边缘产生预测。只需把结果组织成
`data_schema.json` 定义的摘要并请求边缘 `/api/v1/collaboration/decide`，
或使用 SDK 的文件桥接器。公共框架不会导入 ASTGCN 环境，也不会要求 Jetson
安装 WSL 的 Conda 环境。

Student、交通 Edge-Qwen、ExtraTrees 都是交通场景内部可选实现，不是每个场景
必须拥有的公共组件。接入时替换 `plugin.py` 的 `edge_decide()` 或
`cloud_decide()`，并在 `model_assets.example.json` 记录路径和 SHA-256。

真实实验必须另外报告感知精度、模型时延和资源占用；本目录的规则演示结果不能
作为这些模型指标。
