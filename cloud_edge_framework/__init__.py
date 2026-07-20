"""用途：提供场景无关的云边协同事件、调度、协调和服务接口。"""

from cloud_edge_framework.contracts import (
    Action,
    DecisionEnvelope,
    Evidence,
    EventScope,
    Prediction,
    Risk,
    SemanticEvent,
    Timing,
    Uncertainty,
)
from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.registry import SceneRegistry, build_default_registry
from cloud_edge_framework.plugin_manager import PluginRuntimeManager
from cloud_edge_framework.runtime import CloudRuntime, EdgeRuntime
from cloud_edge_framework.version import FRAMEWORK_VERSION

__all__ = [
    "Action",
    "CloudRuntime",
    "DecisionEnvelope",
    "EdgeRuntime",
    "Evidence",
    "FRAMEWORK_VERSION",
    "EventScope",
    "Prediction",
    "PluginRuntimeManager",
    "Risk",
    "SceneEventEnvelope",
    "SceneRegistry",
    "SemanticEvent",
    "Timing",
    "Uncertainty",
    "build_default_registry",
]
