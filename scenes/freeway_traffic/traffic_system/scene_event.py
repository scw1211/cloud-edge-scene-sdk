"""用途：把交通感知模型的原生 JSON 输出显式封装为公共场景事件信封。"""

from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from cloud_edge_framework.contracts import ContractError
from cloud_edge_framework.event_envelope import SceneEventEnvelope


TRAFFIC_EVENT_TYPE = "com.cloudedge.traffic.edge-event.v1"
TRAFFIC_DATA_SCHEMA_ID = (
    "https://cloud-edge.local/schemas/scenes/traffic-edge-event-v1.json"
)
SYNTHETIC_DATASET_EPOCH_MS = 1704067200000
_IDENTITY_FIELDS = {"scene", "event_id", "edge_id", "occurred_at_ms"}


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ContractError("{} must be an integer".format(field_name))
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("{} must be an integer".format(field_name)) from exc
    if result < 0:
        raise ContractError("{} must be non-negative".format(field_name))
    return result


def _event_time(payload: Dict[str, Any]) -> Tuple[int, str]:
    if payload.get("occurred_at_ms") is not None:
        return _integer(payload["occurred_at_ms"], "occurred_at_ms"), "sensor_timestamp"
    sample_id = _integer(payload.get("sample_id"), "sample_id")
    step_minutes = _integer(payload.get("time_step_minutes", 5), "time_step_minutes")
    occurred_at_ms = SYNTHETIC_DATASET_EPOCH_MS + sample_id * step_minutes * 60 * 1000
    return occurred_at_ms, "synthetic_sample_index"


def traffic_event_from_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a CloudEvents-style dictionary from one traffic model output."""
    if not isinstance(payload, dict):
        raise ContractError("traffic model output must be an object")
    event_id = str(payload.get("event_id", "")).strip()
    edge_id = str(payload.get("edge_id", "")).strip()
    region_id = str(payload.get("region_id", "")).strip()
    if not event_id or not edge_id or not region_id:
        raise ContractError("traffic output must include event_id, edge_id and region_id")
    occurred_at_ms, clock_source = _event_time(payload)
    timestamp = datetime.fromtimestamp(
        occurred_at_ms / 1000.0, tz=timezone.utc
    ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    data = {key: value for key, value in payload.items() if key not in _IDENTITY_FIELDS}
    return {
        "specversion": "1.0",
        "id": event_id,
        "source": "urn:edge:{}:astgcn".format(edge_id),
        "type": TRAFFIC_EVENT_TYPE,
        "scene": str(payload.get("scene", "freeway_traffic_management")),
        "edgeid": edge_id,
        "subject": region_id,
        "time": timestamp,
        "datacontenttype": "application/json",
        "dataschema": TRAFFIC_DATA_SCHEMA_ID,
        "clocksource": clock_source,
        "data": data,
    }


def traffic_envelope_from_output(payload: Dict[str, Any]) -> SceneEventEnvelope:
    """Build and validate the common envelope around a traffic-native output."""
    return SceneEventEnvelope.from_dict(traffic_event_from_output(payload))
