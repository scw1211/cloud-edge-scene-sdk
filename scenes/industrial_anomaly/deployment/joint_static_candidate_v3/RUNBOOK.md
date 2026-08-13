# Joint static Q5 candidate v3

This namespace uses the canonical `scene-prefixed-decimal17@v1` package and a
250 ms industrial provider timeout. The measured business-latency gate remains
mean `< 200 ms`; the larger provider timeout only prevents a 186.79 ms valid
first request from being misclassified as a runtime failure. It is isolated
from the formal release store and earlier sidecar evidence.

All promotion commands below run on Nano222. The package, model, base manifest,
registry, and configuration paths are therefore Nano-local absolute paths.

```bash
ROOT=/home/jetson02/cloud-edge-deploy-industrial-multilora-v1
PACKAGE=$ROOT/deployment/releases/traffic-industrial-joint-static-v2-q5km/adapter_package
REGISTRY=$ROOT/runtime/traffic_industrial_joint_static_q5km_candidate_v3/edge_llm_release_store.json
MODEL=$ROOT/deployment/joint_static_fusion_v1/joint_static_fusion.Q5_K_M.gguf
BASE=$ROOT/edge_llm/base_manifest.json

cd "$ROOT"
python3 -m edge_llm_factory.q5_static_joint_release validate \
  --base "$BASE" --package "$PACKAGE"

python3 -m edge_llm_factory release promote \
  --registry "$REGISTRY" \
  --release-id traffic-industrial-joint-static-v2-q5km \
  --base "$BASE" \
  --package "$PACKAGE" \
  --deployment-artifact "$MODEL"
```

The promoted release must have `runtime_adapters: []`. Before startup, print
the supervised command and verify that it contains neither `--lora` nor
`--lora-scaled`:

```bash
python3 -m edge_llm_factory serve-release \
  --registry "$REGISTRY" \
  --binary /home/jetson02/llama.cpp-src/build-cuda-edge-qk/bin/llama-server \
  --port 19393 \
  --runtime-output "$ROOT/scenes/industrial_anomaly/deployment/joint_static_candidate_v3/runtime_output_traffic.json" \
  --runtime-output "$ROOT/scenes/industrial_anomaly/deployment/joint_static_candidate_v3/runtime_output_industrial.json" \
  --startup-probe-config "$ROOT/scenes/industrial_anomaly/deployment/joint_static_candidate_v3/startup_probes.json" \
  --print-command
```

The sidecar endpoint is `127.0.0.1:19391`. While
`192.168.31.135:18100` is unavailable, traffic and industrial smoke results
are provisional only; no authoritative cloud-final claim is allowed.

Rollback is registry-scoped and must name the captured old Q6 release id:

```bash
python3 -m edge_llm_factory release rollback \
  --registry "$REGISTRY" \
  --release-id freeway-traffic-current-state-v2.0.2-q6k
```

The rollback dry output must be traffic `base` plus industrial `disabled`.
