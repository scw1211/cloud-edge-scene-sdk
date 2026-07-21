# 场景 JSON 文件桥接器

## 作用

`cloud_edge_framework.file_bridge` 将场景模型与云边框架隔离开。场景模型只负责把一次识别结果写成 JSON，桥接器负责：

1. 校验统一事件信封。
2. 按 `dataschema` 校验场景数据。
3. 将合法事件写入 SQLite Outbox。
4. 调用边缘框架 `/api/v1/collaboration/decide`。
5. 对断网和服务暂不可用事件进行退避重试。
6. 归档合法文件、隔离非法文件并保存边缘决策回执。

场景模型不需要导入框架代码，也不需要与框架使用同一个 Conda、CUDA 或 Python 环境。

```text
场景专用模型 -> inbox/*.json -> 文件桥接器 -> 边缘框架 -> 云端
                           |          |
                           |          +-> receipts/*.json
                           +-> SQLite Outbox
```

## 输入边界

输入文件必须是完整的 CloudEvents 风格事件，而不是任意模型字典。公共字段由 `schemas/scene_event_envelope.schema.json` 定义，`data` 内部字段由每个场景自己的 JSON Schema 定义。

必须保持以下关系：

- `dataschema` 必须与本地场景 Schema 的 `$id` 完全相同。
- `id` 在系统中必须唯一。同一个 `id` 再次出现时，只允许内容完全相同。
- `time` 必须带时区。
- `data` 与 `data_base64` 只能出现一个。
- 模型没有缺陷类型或定位结果时，不应虚构字段；可以只提交异常分数、阈值和热力图证据。
- 网络状态不属于场景模型输出，由边缘框架自行探测。

工业异常图示例位于：

- `examples/cloud_edge_framework/file_bridge/industrial_event.json`
- `examples/cloud_edge_framework/file_bridge/industrial_anomaly_map_v1.schema.json`

示例只是工业场景合同，不是所有场景必须遵循的数据结构。交通、电网和其他场景可定义完全不同的 `data`，只保留公共信封。

## 启动

桥接器应在独立 Python 环境中常驻，不应安装进旧模型工程的环境。

```bash
python -m cloud_edge_framework.file_bridge watch \
  --input-dir runtime/file_bridge/inbox \
  --state-dir runtime/file_bridge/state \
  --schema-dir /path/to/scene/schemas \
  --edge-base-url http://127.0.0.1:18101 \
  --metrics-output runtime/file_bridge/metrics.json
```

首次联调可以只处理当前文件并退出：

```bash
python -m cloud_edge_framework.file_bridge once \
  --input-dir runtime/file_bridge/inbox \
  --state-dir runtime/file_bridge/state \
  --schema-dir examples/cloud_edge_framework/file_bridge \
  --edge-base-url http://127.0.0.1:18101
```

若桥接器与证据文件位于同一台机器，可增加 `--verify-local-evidence`，校验所有 `file://` URI 是否存在。跨机器部署时应使用边缘节点可访问的 HTTP、对象存储或共享目录 URI。

## 文件状态

```text
state/
  accepted/       已通过校验并写入 Outbox 的原文件
  rejected/       JSON、信封或场景字段不合法的文件
  receipts/       边缘框架成功响应或永久拒绝回执
  outbox.sqlite3  待发送、发送中和已完成事件
```

处理规则：

- 校验失败：移入 `rejected`，旁边生成 `*.error.json`，不发送。
- HTTP 400 等合同错误：标记永久失败并生成回执，不重复发送。
- 超时、断网、HTTP 429 或 5xx：留在 Outbox，指数退避后重试。
- 发送成功：写入 `receipts/<event_id>.json`。
- 进程异常退出：租约到期后，未确认事件重新进入待发送状态。

## 时延口径

每个成功回执分别记录：

- `read_parse_ms`
- `envelope_validation_ms`
- `payload_validation_ms`
- `durable_enqueue_ms`
- `ingestion_total_ms`
- `http_round_trip_ms`
- `file_queue_to_ack_ms`

当前 WSL x86_64 上对 733 B 工业事件执行 1000 次实测：双层校验平均 `0.183 ms`，包含 JSON 读取、双层校验、SQLite `synchronous=FULL` 持久化入队和归档的总开销平均 `5.114 ms`、P95 `6.072 ms`。结果保存在 `results/framework/file_bridge_local_overhead.json`，目标设备应使用同一命令重新测量：

```bash
python -m cloud_edge_framework.benchmark_file_bridge \
  --event examples/cloud_edge_framework/file_bridge/industrial_event.json \
  --schema-dir examples/cloud_edge_framework/file_bridge \
  --iterations 1000
```

JSON 解析、缓存后的 Schema 校验和 SQLite 入队属于实时路径。图片、点云和热力图正文不应在实时路径中做 Base64 编码；实时事件只携带摘要与 URI，原始证据由场景插件按风险和网络状态决定是否上传。

## 场景侧写入要求

场景模型应写入专用 `inbox`。推荐先写临时文件，再原子重命名为 `.json`：

```python
temporary = inbox / ".event-0001.tmp"
target = inbox / "event-0001.json"
temporary.write_text(json.dumps(event), encoding="utf-8")
temporary.replace(target)
```

常驻模式使用 Linux `inotify` 监听文件关闭和原子重命名，不依赖高频目录轮询；周期性扫描只用于恢复遗漏事件。

## 联调前提

桥接器校验通过只表示输入合同有效。边缘服务还必须加载与 `scene`、`type`、`dataschema` 对应的场景插件，才能完成归一化、边缘决策和云端协同。模型预测准确率仍由各场景自己的测试集负责，不能用 Schema 校验结果代替。
