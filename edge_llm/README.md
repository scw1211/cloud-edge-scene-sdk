# Qwen3.5-0.8B 共享纯文本基座

这里保存的是框架级基座契约和派生快照清单，不包含交通 LoRA，也不是可直接提交的场景决策模型。

## 两份清单

- `base_manifest.json`：锁定官方 `Qwen/Qwen3.5-0.8B` 的 commit、Tokenizer、语言模型结构、LoRA 目标层和单 token 动作协议。
- `text_snapshot_manifest.json`：锁定从上游快照派生出的纯文本 BF16 权重、逐文件哈希、参数裁剪统计和可移植目录。

官方上游是 `Qwen3_5ForConditionalGeneration`，包含视觉塔和 MTP 辅助预测头，不能直接把它称为纯文本基座。当前派生过程通过 `AutoModelForCausalLM` 只加载语言主干，并重新导出为 `Qwen3_5ForCausalLM`：

| 项目 | 参数量 |
| --- | ---: |
| 上游完整权重 | 873,438,784 |
| 移除视觉参数 | 100,592,896 |
| 移除 MTP 辅助参数 | 20,452,864 |
| 共享文本基座 | 752,393,024 |
| 文本快照中的多模态参数 | 0 |

本机权重目录：

```text
models/base/qwen35_0_8b_text_2fc06364/
```

该目录约 1.5 GB，是用于 LoRA/QLoRA 训练和合并的 BF16 基座。它不是最终 Jetson 部署格式，不能用它的 BF16 进程内存判断 `<=1.5 GB` 部署指标。每个场景完成 LoRA 后，必须合并并量化为 GGUF，再在目标板上测量时延和内存。

## 通用压缩资产

框架已经增加不依赖场景的校准量化链路。校准集只含数学、代码和自然语言推理，场景样本和冻结测试 prompt 均为 0；量化报告绑定校准文本、importance matrix 和 F16 模型哈希。

```text
共享纯文本 BF16
    + 场景 LoRA -> 合并权重 -> 复用通用校准方法 -> 场景 GGUF

共享纯文本 BF16
    -> 通用校准 Q4/Q5 -> 无场景资源与能力基线
```

LoRA 在 BF16/QLoRA 链路上训练，不能把下面的 GGUF 直接当成 Hugging Face LoRA 训练基座。场景权重合并后应复用通用校准集重新计算 importance matrix，不能直接套用基座矩阵。当前无场景候选：

- `models/gguf/qwen35_0_8b_text_general_calibrated_q5_k_m.gguf`：577,991,040 B。

Q5 相对 F16 文件减少 61.9%，本机 CPU 常驻 RSS 增量约 1162 MB。在 80 题通用测试上达到 9B Teacher 的 64.17%，仍不能写成已经满足 80% 通用能力指标。Q4 和 F16 中间文件的评测记录保留在 `results/edge_llm/`，大型权重已清理。完整方法、数据隔离和原始结果见 `docs/edge_llm_general_compression.md`。

## 重新导出

```bash
python -m edge_llm_factory export-text-base \
  --base deployment/edge_llm/base/qwen35_0_8b_text/base_manifest.json \
  --source-snapshot /path/to/huggingface/snapshot/2fc06364715b967f1860aea9cf38778875588b17 \
  --output models/base/qwen35_0_8b_text_2fc06364 \
  --relative-path models/base/qwen35_0_8b_text_2fc06364 \
  --manifest-output deployment/edge_llm/base/qwen35_0_8b_text/text_snapshot_manifest.json \
  --dtype bfloat16
```

命令拒绝覆盖已有目录。需要重建时，应先人工确认并移走旧快照，不能在原目录上覆盖写入。

## 验证

```bash
python -m edge_llm_factory verify-text-base \
  --base deployment/edge_llm/base/qwen35_0_8b_text/base_manifest.json \
  --snapshot-manifest deployment/edge_llm/base/qwen35_0_8b_text/text_snapshot_manifest.json \
  --snapshot models/base/qwen35_0_8b_text_2fc06364 \
  --load-smoke \
  --report results/edge_llm/qwen35_0_8b_text_base_verification.json
```

验证内容包括逐文件大小和 SHA-256、Safetensors 参数键扫描、参数量闭合、配置中无视觉字段、纯文本 no-thinking 模板、A-H token ID，以及一次真实模型加载和前向。

## 场景使用方式

```text
共享纯文本 BF16 基座
    + 交通 LoRA -> 交通合并模型 -> 通用校准量化 -> 交通 GGUF
    + 工业 LoRA -> 工业合并模型 -> 通用校准量化 -> 工业 GGUF
    + 电网 LoRA -> 电网合并模型 -> 通用校准量化 -> 电网 GGUF
```

各场景只能提交自己的数据、LoRA、输入编码器和动作映射，不能修改共享基座文件。框架 SDK 只分发两份清单和验证工具，不把大体积权重复制进 SDK。
