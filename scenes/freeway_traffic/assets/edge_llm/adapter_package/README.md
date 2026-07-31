# freeway-traffic-qwen35-v9

场景：`freeway_traffic_management`
版本：`1.0.0`
绑定基座：`qwen35-0.8b-text@2fc06364`

此目录是自动构建的场景 LoRA 发布包。加载前必须执行：

```bash
python -m edge_llm_factory validate-adapter --base /path/to/base_manifest.json --package .
```

包内只包含 PEFT 配置、safetensors LoRA、动作映射和哈希锁定的评测证据；
部署 GGUF 单独分发，并通过 `scene_adapter_manifest.json` 中的 SHA256 校验。
