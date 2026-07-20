# Edge-Qwen 通用基座压缩实验

更新时间：2026-07-19。

## 1. 这层压缩解决什么

通用压缩方法不依赖交通、工业或电网场景。完整流程分成训练和部署两条支路：

```text
Qwen3.5-0.8B 官方多模态快照
        -> 只保留语言主干和 tokenizer
        -> 共享纯文本 BF16 基座
              |                     |
              |                     +-> 通用 Q4/Q5 基线，用于资源与能力对照
              |
              +-> 训练场景 LoRA -> 合并权重
                                      -> 复用通用校准方法量化
                                      -> 场景边缘 GGUF
```

通用层提供两项可复用资产：纯文本 BF16 训练基座，以及不含场景样本的校准集和量化校验流程。现有 importance matrix 只对应未适配的基座；场景 LoRA 合并后需要复用同一校准集重新计算，哈希校验会拒绝跨模型套用。各场景 LoRA 必须在 BF16/QLoRA 训练链路上完成；当前框架不会拿 GGUF 文件进行 Hugging Face LoRA 训练。

## 2. 为什么先做量化，不直接剪层

当前文本基座只有 0.8B。对这种小模型继续结构剪枝，容易直接破坏已经有限的推理能力。Findings of EMNLP 2025 对 0.5B 到 3.8B 的六个小模型进行对照，结论是量化整体比剪枝更能保留 fidelity、多语言困惑度和推理准确率；论文也提醒不能只看一个压缩指标。ThinkSLM 对 72 个小模型、17 个推理基准的结果同样表明，量化通常更能保留推理能力，而剪枝影响明显。

因此当前顺序是：

1. 先移除与边缘文本决策无关的视觉塔和 MTP 辅助头；
2. 用严格隔离的通用文本生成 importance matrix；
3. 比较 Q4_K_M、Q5_K_M 与同模板 F16；
4. 场景 LoRA 合并后复用同一校准集，重新计算 importance matrix 并量化；
5. 若后续有 3090 或更大服务器，再实验 9B -> 2B/0.8B 的 white-box KD 或结构化剪枝，不在 0.8B 上盲目删层。

论文依据：

- [Revisiting Pruning vs Quantization for Small Language Models, Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.645/)
- [ThinkSLM: Towards Reasoning in Small Language Models, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1659/)
- [AWQ, MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html)
- [QLoRA, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)

## 3. 数据隔离

校准清单为 `datasets/edge_llm_general_calibration_v1/manifest.json`，对应文本为 `calibration.txt`。

| 项目 | 数值 |
| --- | ---: |
| 校准样本 | 293 |
| 校准 token | 50,135 |
| code / math / natural-language | 98 / 98 / 97 |
| 场景样本 | 0 |
| 冻结测试 prompt 重叠 | 0 |
| 被过滤的交通样本 | 900 |

构建器按原始 prompt 指纹去重，排除冻结测试集 prompt，并在三个通用类别间确定性轮询取样。量化报告会校验校准清单、文本、importance matrix 和 F16 模型的 SHA-256，不能把另一份模型或场景语料悄悄替换进去。

## 4. 产物

| 产物 | 文件大小 | 相对 F16 |
| --- | ---: | ---: |
| 纯文本 F16 | 1,516,736,512 B | 基准 |
| 通用校准 Q4_K_M | 529,289,600 B | 减少 65.1% |
| 通用校准 Q5_K_M | 577,991,040 B | 减少 61.9% |

当前仓库保留：

- `models/gguf/qwen35_0_8b_text_general_calibrated_q5_k_m.gguf`
- `experiments/edge_llm_base_compression/qwen35_0_8b_text_general_imatrix.gguf`

F16、未校准 Q4 和校准 Q4 的大型权重已清理；对应方法、哈希和指标仍保留在 `results/edge_llm/`，需要时可由文本基座与校准矩阵重建。

保留的 Q5 是共享基座的无场景部署基线，不是已经完成任务适配的交通、工业或电网决策模型。

## 5. 通用能力结果

冻结测试共 80 题：GSM8K 30、MBPP 20、C-Eval 30。所有候选使用相同提示模板、no-thinking、解码参数和评分代码；该测试集没有进入校准文本。

| 模型 | Code | Math | 自然语言推理 | Macro | 相对 9B Teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 9B Teacher | 60.0% | 70.0% | 70.0% | 66.67% | 100.0% |
| 纯文本 F16 0.8B | 20.0% | 46.67% | 46.67% | 37.78% | 56.67% |
| 未校准 Q4_K_M | 10.0% | 30.0% | 50.0% | 30.00% | 45.00% |
| 通用校准 Q4_K_M | 15.0% | 43.33% | 53.33% | 37.22% | 55.83% |
| 通用校准 Q5_K_M | 35.0% | 43.33% | 50.0% | 42.78% | 64.17% |

关键结论：

- 校准 Q4 保留了同模板 F16 宏平均的 98.53%，比未校准 Q4 的宏平均高 24.07%；importance matrix 有实际作用。
- Q5 是本轮观测到的能力最好候选，但测试只有 80 题，不能据此声称量化提高了模型本征能力。
- 当前任何 0.8B 候选都没有达到 9B Teacher 的 80%。通用量化解决的是体积和部署成本，不会凭空补足 0.8B 与 9B 的容量差距；要继续提高该指标，需要独立的通用知识蒸馏实验。

原始结果：`results/edge_llm/qwen35_0_8b_general_quantization_capability.json`。

后续通用蒸馏已完成，方法和对照结果见 `docs/edge_llm_general_distillation.md`。

## 6. 常驻运行结果

在本机 WSL、Ollama 常驻、`num_ctx=512`、最多输出 32 token 下，每类抽 4 条，共 12 条请求。内存为 Ollama 服务及 runner 相对空载的 RSS 增量，不等同于 Jetson 整机 RAM，正式指标仍需在目标板复测。

| 模型 | 模式 | 平均 TTFT | 平均完整生成 | RSS 增量 |
| --- | --- | ---: | ---: | ---: |
| F16 | CPU | 563 ms | 1052 ms | 2057 MB |
| 校准 Q4 | CPU | 387 ms | 629 ms | 1117 MB |
| 校准 Q5 | CPU | 609 ms | 813 ms | 1162 MB |
| 校准 Q4 | GPU | 194 ms | 250 ms | 1173 MB |
| 校准 Q5 | GPU | 190 ms | 250 ms | 1175 MB |

Q4 在 CPU 实测中已将常驻增量内存压到 1.5 GB 内，完整生成相对 F16 降低约 40.3%。GPU 完整生成仍约 250 ms，所以当前结果不能写成“完整闭环已稳定小于 0.2 秒”。场景动作采用单 token 输出后还需在 Jetson 上单独测试，且通信回环要另计。

原始结果：

- `results/edge_llm/qwen35_0_8b_general_runtime_cpu.json`
- `results/edge_llm/qwen35_0_8b_general_runtime_gpu.json`

## 7. 复现命令

```bash
python -m edge_llm_factory build-calibration --help
python -m edge_llm_factory build-imatrix --help
python -m edge_llm_factory export-gguf --help
python -m edge_llm_factory benchmark-runtime --help
```

推荐把 Q4_K_M 作为默认无场景资源基线，因为它在本轮 CPU 测试中更快且保留了同模板 F16 的绝大部分宏平均；Q5 作为能力优先基线保留。各场景最终模型必须在 LoRA 合并后重新量化，并由独立任务测试和 Jetson 资源实测共同定版。
