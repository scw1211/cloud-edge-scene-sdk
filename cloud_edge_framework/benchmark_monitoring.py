"""用途：在目标设备上对比逐事件全量监测与有界增量监测的持久化开销。"""

import argparse
import copy
import importlib
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.monitoring import CalibrationDriftMonitor, MonitoringPolicy
from cloud_edge_framework.registry import load_registry_config


def _percentile(values: List[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: List[float]) -> Dict[str, float]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "p50": round(_percentile(values, 50.0), 6),
        "p95": round(_percentile(values, 95.0), 6),
        "p99": round(_percentile(values, 99.0), 6),
        "max": round(max(values), 6),
    }


def _load_event_adapter(
    spec: Optional[str],
) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    if not spec:
        return None
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("event adapter must use module:function syntax")
    adapter = getattr(importlib.import_module(module_name), attribute)
    if not callable(adapter):
        raise TypeError("event adapter is not callable: {}".format(spec))
    return adapter


def _load_event(
    path: Path,
    event_adapter: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        event = json.load(file_obj)
    if not isinstance(event, dict):
        raise ValueError("benchmark event must be an object")
    if "specversion" not in event:
        if event_adapter is None:
            raise ValueError(
                "benchmark input is not a public scene envelope; "
                "provide --event-adapter module:function"
            )
        event = event_adapter(event)
    if not isinstance(event, dict) or "specversion" not in event:
        raise ValueError("event adapter must return a public scene envelope")
    return event


def _run_case(
    plugin: Any,
    event_template: Dict[str, Any],
    iterations: int,
    window_size: int,
    bootstrap_reference_size: int,
    evaluation_interval_events: int,
    evaluation_max_staleness_ms: int,
) -> Dict[str, Any]:
    policy = MonitoringPolicy(
        window_size=window_size,
        bootstrap_reference_size=bootstrap_reference_size,
        evaluation_interval_events=evaluation_interval_events,
        evaluation_max_staleness_ms=evaluation_max_staleness_ms,
    )
    values: List[float] = []
    with tempfile.TemporaryDirectory(prefix="monitoring-benchmark-") as directory:
        monitor = CalibrationDriftMonitor(
            Path(directory) / "monitoring.sqlite3", policy
        )
        try:
            for index in range(iterations):
                event_payload = copy.deepcopy(event_template)
                event_payload["id"] = "{}-monitor-{:06d}".format(
                    event_template["id"], index
                )
                envelope = SceneEventEnvelope.from_dict(event_payload)
                event = plugin.normalize(envelope)
                started = time.perf_counter()
                monitor.observe(event, plugin.monitoring_signals(event))
                values.append((time.perf_counter() - started) * 1000.0)
            final_snapshot = monitor.scene_snapshot(plugin.scene)
        finally:
            monitor.close()
    full_window_start = min(window_size, len(values))
    full_window = values[full_window_start:] or values[-min(len(values), window_size) :]
    return {
        "evaluation_interval_events": evaluation_interval_events,
        "all_calls_ms": _summary(values),
        "full_window_ms": _summary(full_window),
        "final_monitoring": final_snapshot,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark persistent calibration/drift monitoring on this device."
    )
    parser.add_argument(
        "--event",
        default="scene_plugin_template/sample_event.json",
    )
    parser.add_argument(
        "--plugin-config",
        default="deployment/framework/scene_plugins.json",
    )
    parser.add_argument(
        "--event-adapter",
        default=None,
        help="Optional module:function converting a scene-native object to an envelope.",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--bootstrap-reference-size", type=int, default=200)
    parser.add_argument("--evaluation-interval-events", type=int, default=25)
    parser.add_argument("--evaluation-max-staleness-ms", type=int, default=1000)
    parser.add_argument(
        "--output",
        default="results/framework/monitoring_hot_path_benchmark.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 2:
        raise ValueError("iterations must be at least 2")
    if args.window_size < 2 or args.window_size > args.iterations:
        raise ValueError("window-size must be within [2, iterations]")
    if args.evaluation_interval_events < 1:
        raise ValueError("evaluation-interval-events must be at least 1")
    if args.evaluation_max_staleness_ms < 1:
        raise ValueError("evaluation-max-staleness-ms must be at least 1")
    project_root = Path(args.project_root).resolve()
    event_path = Path(args.event)
    if not event_path.is_absolute():
        event_path = project_root / event_path
    plugin_config = Path(args.plugin_config)
    if not plugin_config.is_absolute():
        plugin_config = project_root / plugin_config
    event = _load_event(
        event_path.resolve(),
        _load_event_adapter(args.event_adapter),
    )
    registry = load_registry_config(plugin_config.resolve(), project_root)
    try:
        envelope = SceneEventEnvelope.from_dict(event)
        plugin = registry.for_envelope(envelope)
        full_evaluation = _run_case(
            plugin,
            event,
            args.iterations,
            args.window_size,
            args.bootstrap_reference_size,
            1,
            args.evaluation_max_staleness_ms,
        )
        bounded_evaluation = _run_case(
            plugin,
            event,
            args.iterations,
            args.window_size,
            args.bootstrap_reference_size,
            args.evaluation_interval_events,
            args.evaluation_max_staleness_ms,
        )
    finally:
        registry.close()
    old_mean = full_evaluation["full_window_ms"]["mean"]
    new_mean = bounded_evaluation["full_window_ms"]["mean"]
    output = {
        "schema_version": 1,
        "task": "persistent_monitoring_hot_path_benchmark",
        "device": {
            "node": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "event": str(event_path.resolve()),
        "plugin_config": str(plugin_config.resolve()),
        "iterations": args.iterations,
        "window_size": args.window_size,
        "full_evaluation_every_event": full_evaluation,
        "bounded_incremental_evaluation": bounded_evaluation,
        "full_window_mean_speedup": (
            round(old_mean / new_mean, 6) if new_mean > 0 else None
        ),
        "safety_note": (
            "Per-request status is refreshed at the first of the event-count or "
            "wall-clock bounds. Public monitoring queries, reference changes and "
            "delayed-label updates always force a fresh full-window evaluation."
        ),
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(output, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
