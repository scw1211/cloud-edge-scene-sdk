# 单 token 蒸馏数据格式

训练、验证和测试文件都使用 JSONL，每行包含 `event_id` 和 `messages`。最后一条消息只能是一个基座动作槽，例如 `A`、`B`、`G` 或 `H`。

数据进入 Qwen 前有两次显式转换：

```text
场景模型原生 data --(场景插件 normalize)--> SemanticEvent
SemanticEvent --(context_encoder)--> 短文本或紧凑文本码
```

`package_spec.json` 必须同时记录原生事件 `event_type`、插件 `data_schema` 和 `context_encoder` 版本。三者任一变化都要重新生成数据并做未见测试集评估。

RGB、点云、波形和热力图不直接写进 `messages`。文本 Qwen 只接收场景编码器生成的上下文；原始证据保留 URI、张量描述或二进制引用，由插件和云端模型按需读取。

```json
{"event_id":"sample-001","messages":[{"role":"user","content":"场景适配器生成的紧凑态势编码"},{"role":"assistant","content":"B"}]}
```

约束：

- `train.jsonl`、`val.jsonl`、`test.jsonl` 必须按实体和时间去重拆分；
- `test.jsonl` 不得参与 SFT、纠错蒸馏、DPO 或阈值选择；
- 输入编码由场景团队定义，但必须固定版本并能从原始感知结果复现；
- 动作 token 的业务语义只写在 `action_mapping.json`，不要散落在 prompt 和执行代码里；
- Teacher 生成标签后仍要通过场景安全约束校验，不能默认 Teacher 全部正确。
