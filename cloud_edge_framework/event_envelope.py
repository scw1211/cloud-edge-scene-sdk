"""用途：提供 CloudEvents 风格的场景无关入口信封，业务数据由插件 schema 定义。"""

import base64
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from cloud_edge_framework.contracts import ContractError


CLOUD_EVENTS_SPEC_VERSION = "1.0"
_EXTENSION_NAME = re.compile(r"^[a-z0-9]{1,20}$")
_CORE_FIELDS = {
    "specversion",
    "id",
    "source",
    "type",
    "subject",
    "time",
    "datacontenttype",
    "dataschema",
    "data",
    "data_base64",
    "scene",
    "edgeid",
}


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError("{} must be a non-empty string".format(field_name))
    return value.strip()
def _absolute_uri(value: Any, field_name: str) -> str:
    text = _text(value, field_name)
    if not urlsplit(text).scheme:
        raise ContractError("{} must be an absolute URI".format(field_name))
    return text




def _timestamp(value: Any) -> str:
    text = _text(value, "time")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ContractError("time must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ContractError("time must include a timezone")
    return text


def _json_serializable(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError("{} must be valid JSON data".format(field_name)) from exc


def _extension_value(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ContractError("{} must be finite".format(field_name))
        return value
    raise ContractError("{} must use a CloudEvents scalar type".format(field_name))


@dataclass(frozen=True)
class SceneEventEnvelope:
    """Stable control envelope around a plugin-owned JSON or binary payload."""

    event_id: str
    source: str
    event_type: str
    scene: str
    edge_id: str
    time: str
    dataschema: str
    datacontenttype: str
    subject: str = ""
    data: Any = None
    data_base64: Optional[str] = None
    extensions: Dict[str, Any] = field(default_factory=dict)
    specversion: str = CLOUD_EVENTS_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.specversion != CLOUD_EVENTS_SPEC_VERSION:
            raise ContractError("unsupported CloudEvents specversion")
        _text(self.event_id, "id")
        _text(self.source, "source")
        _text(self.event_type, "type")
        _text(self.scene, "scene")
        _text(self.edge_id, "edgeid")
        _timestamp(self.time)
        _absolute_uri(self.dataschema, "dataschema")
        _text(self.datacontenttype, "datacontenttype")
        if self.subject:
            _text(self.subject, "subject")
        has_data = self.data is not None
        has_binary = self.data_base64 is not None
        if has_data == has_binary:
            raise ContractError("event must contain exactly one of data or data_base64")
        if has_data:
            _json_serializable(self.data, "data")
        else:
            encoded = _text(self.data_base64, "data_base64")
            try:
                base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as exc:
                raise ContractError("data_base64 must contain valid base64") from exc
        if not isinstance(self.extensions, dict):
            raise ContractError("extensions must be an object")
        for name, value in self.extensions.items():
            if not _EXTENSION_NAME.fullmatch(name) or name in _CORE_FIELDS:
                raise ContractError("invalid CloudEvents extension name: {}".format(name))
            _extension_value(value, "extension.{}".format(name))

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SceneEventEnvelope":
        if not isinstance(value, dict):
            raise ContractError("event envelope must be an object")
        has_data = "data" in value and value.get("data") is not None
        has_binary = "data_base64" in value and value.get("data_base64") is not None
        if has_data == has_binary:
            raise ContractError("event must contain exactly one of data or data_base64")
        extensions = {
            str(name): item
            for name, item in value.items()
            if name not in _CORE_FIELDS
        }
        return cls(
            event_id=_text(value.get("id"), "id"),
            source=_text(value.get("source"), "source"),
            event_type=_text(value.get("type"), "type"),
            scene=_text(value.get("scene"), "scene"),
            edge_id=_text(value.get("edgeid"), "edgeid"),
            time=_timestamp(value.get("time")),
            dataschema=_text(value.get("dataschema"), "dataschema"),
            datacontenttype=_text(
                value.get("datacontenttype"),
                "datacontenttype",
            ),
            subject=str(value.get("subject", "")).strip(),
            data=value.get("data") if has_data else None,
            data_base64=_text(value.get("data_base64"), "data_base64")
            if has_binary
            else None,
            extensions=extensions,
            specversion=_text(value.get("specversion"), "specversion"),
        )

    @property
    def occurred_at_ms(self) -> int:
        normalized = self.time[:-1] + "+00:00" if self.time.endswith("Z") else self.time
        return int(datetime.fromisoformat(normalized).timestamp() * 1000)

    def payload_for_validation(self) -> Any:
        return self.data if self.data_base64 is None else self.data_base64

    def binary_data(self) -> bytes:
        if self.data_base64 is None:
            raise ContractError("event does not contain binary data")
        return base64.b64decode(self.data_base64, validate=True)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "specversion": self.specversion,
            "id": self.event_id,
            "source": self.source,
            "type": self.event_type,
            "scene": self.scene,
            "edgeid": self.edge_id,
            "time": self.time,
            "datacontenttype": self.datacontenttype,
            "dataschema": self.dataschema,
        }
        if self.subject:
            result["subject"] = self.subject
        if self.data_base64 is None:
            result["data"] = self.data
        else:
            result["data_base64"] = self.data_base64
        result.update(self.extensions)
        return result
