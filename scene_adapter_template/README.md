# 场景 LoRA 模板

每个场景维护一套独立适配器，不修改公共 Qwen 基座。

需要替换：

1. `action_mapping.json`：动作槽、候选执行动作、风险范围、断网权限；
2. `package_spec.json`：Teacher、数据集 ID、评测证据、GGUF 和验收门槛；
3. `pipeline.json`：本地基座路径、数据路径和 llama.cpp 工具路径；
4. `datasets/<scene>/`：严格拆分的 train/val/test JSONL。

先只检查流水线结构：

```bash
python -m edge_llm_factory run-pipeline \
  --config scene_adapter_template/pipeline.json \
  --project_root . \
  --dry_run
```

正式运行时不要手工跳过失败阶段。修复数据、模型或工具路径后，使用 `--resume` 从已有状态继续。
