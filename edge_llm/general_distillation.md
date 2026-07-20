# Edge-Qwen 通用行为蒸馏实验

更新时间：2026-07-19。

## 1. 实验目的

这一步不使用交通、工业或电网样本，只验证能否把 Qwen3.5 9B 的通用行为迁移到共享纯文本 Qwen3.5 0.8B。它位于场景 LoRA 之前，不替代各场景后续的任务适配。

流程：

```text
Qwen3.5 9B Teacher（no-thinking）
        -> 生成数学、代码、中文推理答案
        -> 按最终答案、代码测试和选项答案自动验收
        -> 只保留 Teacher 正确样本
        -> 纯文本 Qwen3.5 0.8B + LoRA SFT
        -> 合并 LoRA
        -> 重新计算该模型自己的 importance matrix
        -> Q4/Q5 GGUF
```

## 2. 数据与防泄漏

源数据来自项目已有的通用训练集。构建器只接受 `code`、`math` 和 `natural_language_reasoning`，过滤全部交通样本，并按标准化 prompt 指纹去重。

| 项目 | 数值 |
| --- | ---: |
| Teacher 请求 | 980 |
| 训练集通过验收 | 649 |
| 验证集通过验收 | 66 |
| 训练集 code / math / 中文推理 | 80 / 404 / 165 |
| 验证集 code / math / 中文推理 | 5 / 44 / 17 |
| 被拒绝的 Teacher 输出 | 265 |
| 场景样本 | 0 |
| 冻结评测 prompt 重叠 | 0 |

数学按最终数值验收，中文单选按 A-D 验收，代码必须在受限子进程中通过样本自带测试。冻结的 80 题评测集没有参与 Teacher 数据生成、训练或量化校准。

数据清单：`datasets/edge_llm_general_kd_v1/manifest.json`。

## 3. 训练配置

| 参数 | 数值 |
| --- | ---: |
| 基座参数量 | 752,393,024 |
| LoRA rank / alpha | 32 / 64 |
| 可训练参数 | 12,779,520（1.67%） |
| epoch | 3 |
| 有效 batch | 16 |
| 学习率 | 1e-4 |
| 精度 | BF16 |
| 最佳验证损失 | 0.383322 |
| 训练耗时 | 458.05 s |

第 1 个 epoch 的验证损失最低，后两轮开始上升。Trainer 已恢复第 1 轮最佳检查点后再保存 LoRA，未使用最后一轮权重。

## 4. 能力结果

冻结评测为 GSM8K 30 题、MBPP 20 题、C-Eval 30 题。所有模型使用相同提示词、no-thinking、temperature=0、seed=42 和评分程序。

| 模型 | Code | Math | 中文推理 | Macro | 相对 9B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 9B Teacher | 60.00% | 70.00% | 70.00% | 66.67% | 100.00% |
| 原纯文本 F16 0.8B | 20.00% | 46.67% | 46.67% | 37.78% | 56.67% |
| 蒸馏 F16 0.8B | 25.00% | 56.67% | 56.67% | 46.11% | 69.17% |
| 原校准 Q4 | 15.00% | 43.33% | 53.33% | 37.22% | 55.83% |
| 蒸馏校准 Q4 | 20.00% | 43.33% | 63.33% | 42.22% | 63.33% |
| 蒸馏校准 Q5 | 20.00% | 36.67% | 53.33% | 36.67% | 55.00% |

F16 结果说明蒸馏权重本身有效：宏平均提高 8.33 个百分点，数学和中文推理分别保留 Teacher 的 80.95%。代码只保留 41.67%，是当前未达到“三项均保留 80%”的主要原因。

Q4 是本轮部署候选。它比原校准 Q4 的宏平均提高 5 个百分点，中文推理保留率达到 90.48%。Q5 在这组 80 题上反而下降，因此不按位宽高低直接选型，也不把单次非单调结果解释为模型本征能力变化。

原始结果：`results/edge_llm/qwen35_0_8b_general_kd_capability.json`。

## 5. Q4 常驻资源

本机 WSL、Ollama 常驻、`num_ctx=512`、最多输出 32 token，每类 4 条，共 12 条请求。

| 模型 | 模式 | 平均 TTFT | 平均完整生成 | RSS 增量 |
| --- | --- | ---: | ---: | ---: |
| 原校准 Q4 | CPU | 386.60 ms | 628.65 ms | 1117.16 MB |
| 蒸馏校准 Q4 | CPU | 396.18 ms | 663.78 ms | 1116.72 MB |
| 原校准 Q4 | GPU | 194.39 ms | 249.66 ms | 1173.12 MB |
| 蒸馏校准 Q4 | GPU | 199.97 ms | 261.77 ms | 1173.56 MB |

蒸馏没有增加 GGUF 大小，Q4 仍为 529,289,600 B，增量常驻内存低于 1.5 GB。GPU 平均 TTFT 接近 0.2 s，但完整 32-token 生成平均 261.77 ms，不能据此声称云边完整闭环小于 0.2 s。

资源结果：

- `results/edge_llm/qwen35_0_8b_general_kd_runtime_cpu.json`
- `results/edge_llm/qwen35_0_8b_general_kd_runtime_gpu.json`

## 6. 产物

两轮通用蒸馏都没有通过能力提升门禁，因此没有进入共享基座发布链。LoRA、checkpoint、合并模型和 GGUF 大型候选已经清理，避免与当前 Q5 基座和交通 Q6 active release 混淆。

保留的可追溯证据：

- 训练数据与清单：`datasets/edge_llm_general_kd_v2/`、`datasets/edge_llm_general_kd_source_v2/`；
- 第一轮能力与资源汇总：`results/edge_llm/qwen35_0_8b_general_kd_summary.json`；

## 7. 第二轮独立代码数据实验

第一轮代码训练样本只有 80 条。第二轮没有使用 MBPP test，而是固定使用官方 MBPP full 的 train（task 601-974）和 validation（task 511-600），数据 revision 锁定为 `4bb6404fdc6cacfda99d4ac4205087b89d32030c`。构建时发现 full validation 的 task 569 与冻结的 sanitized MBPP 测试题存在同 prompt，已在进入 Teacher 前剔除；最终训练、验证和冻结评测的 prompt 重叠均为 0。

| 项目 | 训练 | 验证 |
| --- | ---: | ---: |
| MBPP 官方源样本 | 374 | 90 |
| 通过 9B Teacher 执行验收的 code | 229 | 51 |
| math | 404 | 44 |
| 中文推理 | 165 | 17 |
| 合计 | 798 | 112 |

训练采用确定性的类别平衡采样，每个 epoch 的有效样本为 code / math / 中文推理各 404 条；原始 JSONL 不复制样本，manifest 同时记录 798 条唯一数据和 1212 条有效采样索引。LoRA 仍为 rank 32 / alpha 64，BF16、有效 batch 16、学习率 1e-4。训练 2 个 epoch，耗时 521.75 s，验证损失在第 1 个 epoch 最低，保存的是该检查点。

数据清单：

- `datasets/edge_llm_general_kd_source_v2/manifest.json`
- `datasets/edge_llm_general_kd_v2/manifest.json`
- `experiments/edge_llm_general_kd/qwen35_0_8b_general_kd_lora_v2_balanced/train_metrics.json`

## 8. 第二轮能力门禁

第二轮合并后的 F16 使用同一组冻结 80 题复测，没有先量化。

| F16 候选 | Code | Math | 中文推理 | Macro | 相对 9B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第一轮 incumbent | 25.00% | 56.67% | 56.67% | 46.11% | 69.17% |
| 第二轮 balanced candidate | 25.00% | 43.33% | 56.67% | 41.67% | 62.50% |
| 变化 | 0.00 | -13.34 | 0.00 | -4.44 | -6.67 |

第二轮增加了独立代码样本，但 20 道冻结代码题仍为 5 道通过；数学少 4 道，总体少 3 道。训练样本没有被长度过滤，失败不能归因于截断。当前证据说明，简单的类别等量过采样和总体 token loss 选点不能稳定提高代码执行正确率，还会造成通用能力回退。

框架新增 `gate-general-kd` 发布门禁。候选必须完整跑完冻结评测、宏平均不下降、所有分项不超过允许回退量；针对本轮代码专项实验还要求 Code 至少提升 1 个百分点。第二轮门禁未通过，`quantization_allowed=false`，因此没有继续生成 Q4，也没有覆盖第一轮正式候选。

结果：

- `results/edge_llm/qwen35_0_8b_general_kd_v2_capability.json`
- `results/edge_llm/qwen35_0_8b_general_kd_v2_gate.json`

## 9. 当前结论

第一轮仍是当前最好且可部署的通用蒸馏版本。现阶段可以证明场景无关 Teacher 行为蒸馏带来提升，但不能写成三项通用能力均达到 80%。第二轮是有效的负结果：它验证了独立数据、防泄漏、平衡采样和自动发布门禁，也证明“样本更多”不等于“代码能力更高”。下一轮若继续，应改为按可执行结果选择 checkpoint，或采用执行反馈和 on-policy 纠错样本，而不是继续提高静态过采样倍数。
