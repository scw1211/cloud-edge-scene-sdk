"""Schema validation is single-pass without becoming a mutable cache."""

from __future__ import annotations

import unittest

from cloud_edge_framework.contracts import ContractError, SemanticEvent
from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.plugins.base import ScenePlugin
from cloud_edge_framework.registry import SceneRegistry
from cloud_edge_framework.runtime import CloudRuntime


class _CountingValidationPlugin(ScenePlugin):
    scene = "validation-fixture"
    aliases = ()
    event_types = ("org.example.validation.v1",)
    data_schema_id = "https://example.org/schemas/validation-v1.json"

    def __init__(self) -> None:
        self.schema_reads = 0
        self.validation_runs = 0

    def payload_schema(self):
        self.schema_reads += 1
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": self.data_schema_id,
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "number"}},
            "additionalProperties": False,
        }

    def validate_envelope(self, envelope: SceneEventEnvelope):
        accepted = getattr(self._validation_scope, "accepted", None)
        if not (
            accepted is not None
            and accepted[0] is self
            and accepted[1] is envelope
        ):
            self.validation_runs += 1
        return super().validate_envelope(envelope)

    def normalize(self, envelope: SceneEventEnvelope) -> SemanticEvent:
        self.validate_envelope(envelope)
        return SemanticEvent.from_dict(
            {
                "schema_version": "1.0",
                "event_id": envelope.event_id,
                "scene": self.scene,
                "task": "validation",
                "edge_id": envelope.edge_id,
                "occurred_at_ms": envelope.occurred_at_ms,
                "scope": {
                    "entity_id": "entity-a",
                    "subsystem": "fixture",
                    "state_variable": "value",
                    "region_id": "region-a",
                    "window_start_ms": envelope.occurred_at_ms,
                    "window_end_ms": envelope.occurred_at_ms,
                },
                "prediction": {
                    "label": "low",
                    "confidence": 0.95,
                    "probabilities": {"low": 0.95},
                },
                "risk": {"level": "low", "score": 0.1},
                "uncertainty": {
                    "confidence": 0.95,
                    "calibrated": True,
                    "prediction_set": ["low"],
                    "method": "fixture",
                },
                "timing": {"deadline_ms": 100.0},
                "evidence": [
                    {
                        "evidence_id": "validation-summary",
                        "level": "summary",
                        "modality": "fixture",
                        "encoding": "json",
                        "inline": {"value": envelope.data["value"]},
                        "size_bytes": 8,
                        "content_type": "application/json",
                    }
                ],
                "candidate_actions": [],
            }
        )

    def edge_decide(self, event):
        raise NotImplementedError

    def cloud_decide(self, event):
        raise NotImplementedError


def _payload():
    return {
        "specversion": "1.0",
        "id": "validation-event",
        "source": "urn:test:edge-a",
        "type": "org.example.validation.v1",
        "subject": "validation-fixture",
        "time": "2026-08-05T00:00:00Z",
        "datacontenttype": "application/json",
        "dataschema": "https://example.org/schemas/validation-v1.json",
        "scene": "validation-fixture",
        "edgeid": "edge-a",
        "data": {"value": 1.0},
    }


class ValidationHotPathTest(unittest.TestCase):
    def test_cloud_runtime_runs_payload_schema_once(self) -> None:
        plugin = _CountingValidationPlugin()
        registry = SceneRegistry([plugin])
        baseline = plugin.validation_runs
        try:
            event = CloudRuntime(registry).normalize(_payload())
            self.assertEqual(event.event_id, "validation-event")
            self.assertEqual(plugin.validation_runs - baseline, 1)
        finally:
            registry.close()

    def test_validation_scope_does_not_cache_mutated_payload(self) -> None:
        plugin = _CountingValidationPlugin()
        envelope = SceneEventEnvelope.from_dict(_payload())
        plugin.normalize_envelope(envelope)
        envelope.data["value"] = "invalid"
        with self.assertRaises(ContractError):
            plugin.validate_envelope(envelope)


if __name__ == "__main__":
    unittest.main()
