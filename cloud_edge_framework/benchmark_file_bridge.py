"""用途：量化缓存 Schema 校验和完整文件入队给实时链路增加的本地开销。"""

import argparse
import copy
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from cloud_edge_framework.file_bridge import (
    FileEventBridge,
    LocalEventValidator,
)


class _UnusedSender:
    def send(self, envelope: Dict[str, Any]):
        raise RuntimeError("benchmark does not execute HTTP delivery")


def _summary(values: Sequence[float]) -> Dict[str, float]:
    data = sorted(float(value) for value in values)
    if not data:
        return {
            "count": 0,
            "average_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    def percentile(fraction: float) -> float:
        index = min(len(data) - 1, max(0, int(round((len(data) - 1) * fraction))))
        return round(data[index], 6)

    return {
        "count": len(data),
        "average_ms": round(statistics.fmean(data), 6),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "min_ms": round(data[0], 6),
        "max_ms": round(data[-1], 6),
    }


def _load_event(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("event must be a JSON object")
    return value


def benchmark(
    event_path: Path,
    envelope_schema_path: Path,
    schema_directories: Sequence[Path],
    iterations: int,
    warmup_iterations: int,
) -> Dict[str, Any]:
    if iterations <= 0 or warmup_iterations < 0:
        raise ValueError("iterations must be positive and warmup must be non-negative")
    event = _load_event(event_path)
    validator = LocalEventValidator(
        envelope_schema_path,
        schema_directories,
        verify_local_evidence=False,
    )
    for _ in range(warmup_iterations):
        validator.validate(event)

    validation_values: List[float] = []
    for _ in range(iterations):
        started_ns = time.perf_counter_ns()
        validator.validate(event)
        validation_values.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)

    ingestion_values: Dict[str, List[float]] = {
        "read_parse_ms": [],
        "envelope_validation_ms": [],
        "payload_validation_ms": [],
        "validation_ms": [],
        "durable_enqueue_ms": [],
        "ingestion_total_ms": [],
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bridge = FileEventBridge(
            input_directory=root / "inbox",
            state_directory=root / "state",
            validator=validator,
            sender=_UnusedSender(),
        )
        for index in range(iterations):
            sample = copy.deepcopy(event)
            sample["id"] = "{}-bridge-benchmark-{:06d}".format(event["id"], index)
            source = bridge.input_directory / "event-{:06d}.json".format(index)
            source.write_text(
                json.dumps(sample, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            outcome = bridge.ingest_file(source)
            if outcome.get("status") != "enqueued":
                raise RuntimeError("benchmark ingestion failed: {}".format(outcome))
            timings = outcome["timings"]
            for name in ingestion_values:
                ingestion_values[name].append(float(timings[name]))
        bridge.close()

    event_bytes = len(
        json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    return {
        "schema_version": 1,
        "task": "file_event_bridge_local_overhead",
        "platform": {
            "hostname": platform.node(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "event_file": str(event_path.resolve()),
        "event_bytes": event_bytes,
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "validator": validator.describe(),
        "cached_in_memory_validation": _summary(validation_values),
        "durable_file_ingestion": {
            name: _summary(values) for name, values in ingestion_values.items()
        },
        "scope": (
            "local JSON read, cached envelope/payload validation, SQLite FULL durable "
            "enqueue and source archive; excludes scene-model inference and HTTP delivery"
        ),
    }


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--schema-dir", action="append", required=True)
    parser.add_argument(
        "--envelope-schema",
        default=str(project_root / "schemas" / "scene_event_envelope.schema.json"),
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument(
        "--output",
        default="results/framework/file_bridge_local_overhead.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = benchmark(
        event_path=Path(args.event),
        envelope_schema_path=Path(args.envelope_schema),
        schema_directories=[Path(value) for value in args.schema_dir],
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
