# Joint static Q5 candidate sidecar

This candidate is isolated from the formal release store.  The model is one
statically fused Q5_K_M GGUF and the release must contain
`runtime_adapters: []`.  Do not pass `--llama-lora-adapter`.

Build and validate the fail-closed package after the packager descriptor is
complete:

```bash
python -m edge_llm_factory.q5_static_joint_release build \
  --descriptor /home/scw/unified-distill-work-20260813/joint_traffic_industrial_v1/deployment/q5_static_joint_release_v1/release_descriptor.json \
  --output /home/scw/unified-distill-work-20260813/joint_traffic_industrial_v1/deployment/q5_static_joint_release_v1/adapter_package

python -m edge_llm_factory.q5_static_joint_release validate \
  --base /home/scw/cloud-edge-scene-sdk/edge_llm/base_manifest.json \
  --package /home/scw/unified-distill-work-20260813/joint_traffic_industrial_v1/deployment/q5_static_joint_release_v1/adapter_package
```

Promote only into the isolated candidate registry (the empty adapter list is
intentional):

```bash
python -m edge_llm_factory release promote \
  --registry runtime/traffic_industrial_joint_static_q5km_candidate/edge_llm_release_store.json \
  --release-id traffic-industrial-joint-static-v1-q5km \
  --base /home/scw/cloud-edge-scene-sdk/edge_llm/base_manifest.json \
  --package /home/scw/unified-distill-work-20260813/joint_traffic_industrial_v1/deployment/q5_static_joint_release_v1/adapter_package \
  --deployment-artifact /home/jetson02/cloud-edge-deploy-industrial-multilora-v1/deployment/joint_static_fusion_v1/joint_static_fusion.Q5_K_M.gguf
```

Print the supervised llama-server command before startup and verify it contains
neither `--lora` nor `--lora-scaled`:

```bash
python -m edge_llm_factory serve-release \
  --registry runtime/traffic_industrial_joint_static_q5km_candidate/edge_llm_release_store.json \
  --binary /path/to/llama-server \
  --port 19393 \
  --runtime-output scenes/industrial_anomaly/deployment/joint_static_candidate/runtime_output_traffic.json \
  --runtime-output scenes/industrial_anomaly/deployment/joint_static_candidate/runtime_output_industrial.json \
  --startup-probe-config scenes/industrial_anomaly/deployment/joint_static_candidate/startup_probes.json \
  --print-command
```

Start the local sidecar only after the printed-command and package checks pass:

```bash
python scenes/freeway_traffic/deploy_node.py run \
  --role edge \
  --cloud-url http://192.168.31.135:18100 \
  --service-config scenes/industrial_anomaly/deployment/joint_static_candidate/edge_service.json \
  --llama-registry runtime/traffic_industrial_joint_static_q5km_candidate/edge_llm_release_store.json \
  --llama-binary /path/to/llama-server \
  --llama-port 19393 \
  --llama-runtime-output scenes/industrial_anomaly/deployment/joint_static_candidate/runtime_output_traffic.json \
  --llama-runtime-output scenes/industrial_anomaly/deployment/joint_static_candidate/runtime_output_industrial.json \
  --llama-startup-probes scenes/industrial_anomaly/deployment/joint_static_candidate/startup_probes.json
```

The candidate edge endpoint is `127.0.0.1:19391`.  A local sidecar result is
provisional while `192.168.31.135:18100` is unreachable from the edge node;
full traffic/industrial E2E must be rerun after that cloud endpoint recovers.

Rollback is registry-scoped:

```bash
python -m edge_llm_factory release rollback \
  --registry runtime/traffic_industrial_joint_static_q5km_candidate/edge_llm_release_store.json \
  --release-id OLD_RELEASE_ID
```

When the old release deployment SHA differs from the static Q5 SHA, the
traffic output is generated in base mode and the industrial output is written
as an explicit disabled-runtime sentinel.
