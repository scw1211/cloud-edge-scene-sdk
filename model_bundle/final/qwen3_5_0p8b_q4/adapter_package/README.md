# joint-traffic-industrial-static-q4km-raw16

Package type: `joint-static-raw16-model-release`
Runtime mode: `joint_static_fusion`
Runtime adapters: `[]`

The Q4_K_M GGUF identified by `deployment.artifact_sha256` is the only runtime
model. `adapter_model.safetensors` and `adapter_config.json` are retained only
as training and static-merge lineage; they must never be passed to llama-server
as a runtime LoRA.

Validate with:

```bash
python -m edge_llm_factory.q4_static_joint_release validate \
  --base /path/to/base_manifest.json \
  --package .
```
