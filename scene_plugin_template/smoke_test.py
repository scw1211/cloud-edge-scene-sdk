"""用途：验证模板插件的正常网络、断网自治和多边缘冲突协调链路。"""

import copy
import json
from pathlib import Path

from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.registry import SceneRegistry
from cloud_edge_framework.runtime import CloudRuntime, EdgeRuntime
from cloud_edge_framework.scheduling import NetworkSnapshot
from edge_llm_factory.contracts import read_json_object
from edge_llm_factory.runtime import ActionDecoder
from scene_plugin_template.plugin import ExampleScenePlugin


def load_event(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sample event must be an object")
    return value


def main() -> None:
    package_root = Path(__file__).resolve().parent
    sample_path = package_root / "sample_event.json"
    registry = SceneRegistry()
    registry.register(
        ExampleScenePlugin(),
        "scene_plugin_template.plugin:ExampleScenePlugin",
    )
    registry.warmup()
    try:
        sample = load_event(sample_path)
        envelope = SceneEventEnvelope.from_dict(sample)
        plugin = registry.for_envelope(envelope)
        normalized_event = plugin.normalize(envelope)
        decision_input = normalized_event.to_dict()
        base = read_json_object(package_root / "base_manifest.json")
        action_mapping = read_json_object(package_root / "action_mapping.json")
        decoder = ActionDecoder(base, action_mapping)
        decoded_action = decoder.decode("B", decision_input, network_available=False)
        blocked_no_action = decoder.decode("A", decision_input, network_available=False)
        normal = EdgeRuntime(registry).process(
            sample,
            NetworkSnapshot(
                available=True,
                rtt_ms=15.0,
                jitter_ms=3.0,
                loss_rate=0.0,
                cloud_queue_ms=1.0,
                cloud_compute_ms=12.0,
            ),
        )
        outage = EdgeRuntime(registry).process(
            sample,
            NetworkSnapshot(
                available=False,
                rtt_ms=0.0,
                jitter_ms=0.0,
                loss_rate=1.0,
                cloud_queue_ms=0.0,
                cloud_compute_ms=0.0,
            ),
        )

        peer = copy.deepcopy(sample)
        peer["id"] = "industrial_event_0002"
        peer["edgeid"] = "industrial_edge_02"
        peer["source"] = "urn:edge:industrial_edge_02:anomaly-detector"
        peer["data"]["proposed_limit_percent"] = 55
        peer_envelope = SceneEventEnvelope.from_dict(peer)
        peer_event = registry.for_envelope(peer_envelope).normalize(peer_envelope)
        coordination = CloudRuntime(registry).coordinate([normalized_event, peer_event])

        assert normal["schedule"]["route"] == "cloud_sync"
        assert outage["final_decision"]["route"] == "local_autonomy"
        assert coordination["initial_conflict_count"] == 1
        assert coordination["residual_conflict_count"] == 0
        assert decoded_action["decision"] == "set_operating_limit"
        assert decoded_action["safety_fallback"] is False
        assert blocked_no_action["slot"] == "G"

        print(
            json.dumps(
                {
                    "plugin": registry.descriptors()[0],
                    "normal_route": normal["schedule"]["route"],
                    "outage_route": outage["final_decision"]["route"],
                    "initial_conflicts": coordination["initial_conflict_count"],
                    "residual_conflicts": coordination["residual_conflict_count"],
                    "resolution_success_rate": coordination["resolution_success_rate"],
                    "edge_llm_action_contract": decoded_action["decision"],
                    "unsafe_slot_fallback": blocked_no_action["slot"],
                    "status": "smoke_test_passed"
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        registry.close()


if __name__ == "__main__":
    main()
