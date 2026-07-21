# 云端与边缘模型包

这个目录定义框架默认使用的两类模型，但不把大权重提交进 Git 源码历史。

- 云端 Teacher：官方 `qwen3.5:9b`，项目不做微调、蒸馏或剪枝。Ollama 官方分发本身采用 Q4_K_M，因此这里的“原始”表示模型行为和参数未经本项目修改，不表示 BF16 权重。
- 边缘通用 Student：由 Qwen3.5 9B 的场景无关数据蒸馏到纯文本 Qwen3.5 0.8B，再量化为 Q4_K_M。场景团队仍需在其上训练自己的 LoRA。

模型名称、大小、下载地址和 SHA-256 全部记录在 `catalog.json`。安装器遇到摘要不一致会直接失败，不会静默使用其他模型。

## 安装

安装全部模型：

```bash
python -m model_bundle.install_models --all
```

只安装云端 Teacher：

```bash
python -m model_bundle.install_models --cloud
```

只安装边缘模型：

```bash
python -m model_bundle.install_models --edge
```

使用已经下载的边缘 GGUF：

```bash
python -m model_bundle.install_models \
  --edge \
  --edge-file /path/to/qwen35_0_8b_text_general_kd_q4_k_m.gguf
```

只核对当前 Ollama 模型，不下载或重建：

```bash
python -m model_bundle.install_models --all --verify-only
```

云端权重来自官方 Ollama Registry，不在本项目重复上传。边缘自训练权重由本项目 GitHub Release 分发。两者均按 Apache-2.0 模型许可使用，详见 `APACHE-2.0.txt`。
