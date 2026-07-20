# 场景团队交付清单

场景团队完成接入后，请交回以下材料。

## 代码与模型

- [ ] 场景插件目录，已经移除 `ExampleScenePlugin` 等模板命名
- [ ] 边缘模型加载和推理代码
- [ ] 云端模型或优化器加载和推理代码
- [ ] 模型权重、版本号、哈希及依赖说明
- [ ] 插件配置 `scene_plugins.json`
- [ ] 可重复运行的启动和测试命令

## 边缘 Qwen 适配器

- [ ] 使用 SDK 提供的同一份 `base_manifest.json`
- [ ] `text_snapshot_manifest.json` 与共享纯文本快照校验通过
- [ ] 没有直接使用含视觉塔和 MTP 头的官方多模态快照训练 LoRA
- [ ] LoRA 采用 PEFT 配置和 `adapter_model.safetensors`
- [ ] `action_mapping.json` 中定义了断网可执行动作和 G/H 保留槽
- [ ] train/val/test 按实体和时间拆分，测试集没有用于纠错或 DPO
- [ ] `package_spec.json` 的每个指标都能回溯到带 SHA256 的报告
- [ ] `input_contract` 已绑定 `event_type`、`data_schema` 和 `context_encoder` 版本
- [ ] 文本 Qwen 的 `direct_media_to_llm` 为 `false`
- [ ] 上下文编码可从原生模型输出确定性复现
- [ ] 量化 GGUF、平均 TTFT、整机内存和一 token 有效率已经实测
- [ ] `python -m edge_llm_factory validate-adapter` 校验通过
- [ ] `runtime.edge.json` 和 `runtime.cloud.json` 已执行 `verify-runtime`，配置中没有明文密钥
- [ ] 流水线 `publish_release` 已生成版本仓库，活动版本的基座、LoRA 和 GGUF 哈希校验通过
- [ ] 已实际演练一次指定版本回滚，产物篡改时回滚会被拒绝
- [ ] 基座升级时重新训练适配器，没有跨 revision 混用 LoRA
- [ ] 通用基座候选未使用冻结评测集生成 Teacher 标签或训练
- [ ] 通用基座候选通过 `gate-general-kd` 后才进入量化和发布


## 数据协议

- [ ] 场景原生模型输出样例，不要求伪造统一 label、bbox 或风险字段
- [ ] 带 `$id` 的场景 `data_schema.json`
- [ ] 完整 `SceneEventEnvelope` 样例，`type`、`dataschema` 与插件声明一致

- [ ] 一条低风险事件样例
- [ ] 一条高风险或严重风险事件样例
- [ ] 至少一组来自多个边缘节点的关联事件
- [ ] summary、feature、raw 证据分别包含什么
- [ ] 实体 ID、区域 ID、共享资源 ID 的命名规则
- [ ] 事件时间窗和业务 deadline 的定义

## 决策语义

- [ ] 所有候选动作及参数范围
- [ ] 哪些动作可以同时执行
- [ ] 哪些动作构成真实资源冲突
- [ ] 冲突消解后的动作约束
- [ ] 本地自治时允许执行的安全动作

## 框架联调

- [ ] 云端使用 `cloud_service.json` 独立启动并通过 `/ready`
- [ ] 边缘使用 `edge_service.json` 独立启动，`cloud.base_url` 指向真实云端
- [ ] 事件只提交给边缘 `/decide`，没有绕过边缘调度直接调用云端
- [ ] 停止云端后本地决策仍可返回，Outbox 出现 pending 事件
- [ ] 云端恢复后 Outbox 自动归零，同一事件重试未重复执行
- [ ] `/metrics` 保存了闭环时延、通信字节、路由和自治统计

## 实验结果

- [ ] 场景模型准确率、召回率或 F1
- [ ] 边缘模型相对云端模型的能力保持率
- [ ] 正常网络完整闭环时延
- [ ] 弱网和断网业务成功率
- [ ] 单次请求上传量和压缩率
- [ ] 自然冲突率、残余冲突率和消解成功率
- [ ] 边缘设备整机内存和推理资源占用

模板冒烟测试结果不能替代以上场景实验。
