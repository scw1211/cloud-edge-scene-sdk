"""用途：加载并校验边缘服务和云端服务的版本化运行配置。"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional

from jsonschema import Draft202012Validator


CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ListenConfig:
    host: str
    port: int
    max_body_bytes: int
    access_log: bool


@dataclass(frozen=True)
class StorageConfig:
    outbox: Optional[Path]
    performance_profiles: Optional[Path]
    feedback: Optional[Path]
    idempotency: Optional[Path]


@dataclass(frozen=True)
class SchedulerConfig:
    confidence_threshold: float
    jitter_guard: float


@dataclass(frozen=True)
class CloudClientConfig:
    base_url: str
    timeout_seconds: float
    max_attempts: int
    retry_backoff_seconds: float


@dataclass(frozen=True)
class NetworkProbeConfig:
    interval_seconds: float
    window_size: int
    failure_threshold: int
    uplink_mbps: float
    downlink_mbps: float
    expected_response_bytes: int
    cloud_queue_ms: float
    cloud_compute_ms: float


@dataclass(frozen=True)
class ReplayConfig:
    interval_seconds: float
    batch_size: int
    lease_seconds: float
    max_backoff_seconds: float


@dataclass(frozen=True)
class IdempotencyConfig:
    ttl_seconds: float
    max_entries: int


@dataclass(frozen=True)
class ReleaseWatchConfig:
    enabled: bool
    registry: Optional[Path]
    interval_seconds: float


@dataclass(frozen=True)
class FrameworkServiceConfig:
    role: str
    listen: ListenConfig
    plugin_config: Path
    storage: StorageConfig
    scheduler: SchedulerConfig
    cloud: Optional[CloudClientConfig]
    network_probe: NetworkProbeConfig
    replay: ReplayConfig
    idempotency: IdempotencyConfig
    source_path: Path
    release_watch: Optional[ReleaseWatchConfig] = None


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError("framework service config must contain an object")
    return value


def _resolve_path(project_root: Path, value: Any) -> Optional[Path]:
    if value in {None, ""}:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _validate_config(project_root: Path, value: Dict[str, Any]) -> None:
    schema_path = project_root / "schemas" / "framework_service_config.schema.json"
    schema = _read_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "config"
        raise ValueError("framework config {}: {}".format(location, error.message))


def load_service_config(
    path: Path,
    project_root: Path,
    expected_role: Optional[str] = None,
) -> FrameworkServiceConfig:
    source_path = path.resolve()
    project_root = project_root.resolve()
    value = _read_json(source_path)
    _validate_config(project_root, value)
    role = str(value["role"])
    if expected_role is not None and role != expected_role:
        raise ValueError(
            "framework config role {!r} does not match required role {!r}".format(
                role, expected_role
            )
        )

    listen_raw = dict(value.get("listen", {}))
    storage_raw = dict(value.get("storage", {}))
    scheduler_raw = dict(value.get("scheduler", {}))
    probe_raw = dict(value.get("network_probe", {}))
    replay_raw = dict(value.get("replay", {}))
    idempotency_raw = dict(value.get("idempotency", {}))
    cloud_raw = value.get("cloud")
    release_watch_raw = value.get("release_watch")

    cloud = None
    if cloud_raw is not None:
        cloud_data = dict(cloud_raw)
        cloud = CloudClientConfig(
            base_url=str(cloud_data["base_url"]).rstrip("/"),
            timeout_seconds=float(cloud_data.get("timeout_seconds", 0.5)),
            max_attempts=int(cloud_data.get("max_attempts", 2)),
            retry_backoff_seconds=float(
                cloud_data.get("retry_backoff_seconds", 0.025)
            ),
        )
    if role == "edge" and cloud is None:
        raise ValueError("edge service config must define cloud")

    release_watch = None
    if release_watch_raw is not None:
        release_data = dict(release_watch_raw)
        registry = _resolve_path(project_root, release_data.get("registry"))
        enabled = bool(release_data.get("enabled", False))
        if enabled and registry is None:
            raise ValueError("enabled release_watch requires registry")
        release_watch = ReleaseWatchConfig(
            enabled=enabled,
            registry=registry,
            interval_seconds=float(release_data.get("interval_seconds", 2.0)),
        )

    plugin_path = _resolve_path(project_root, value["plugin_config"])
    if plugin_path is None:
        raise ValueError("plugin_config must not be empty")
    return FrameworkServiceConfig(
        role=role,
        listen=ListenConfig(
            host=str(listen_raw.get("host", "0.0.0.0")),
            port=int(listen_raw.get("port", 18100 if role == "cloud" else 18101)),
            max_body_bytes=int(listen_raw.get("max_body_bytes", 8 * 1024 * 1024)),
            access_log=bool(listen_raw.get("access_log", False)),
        ),
        plugin_config=plugin_path,
        storage=StorageConfig(
            outbox=_resolve_path(project_root, storage_raw.get("outbox")),
            performance_profiles=_resolve_path(
                project_root, storage_raw.get("performance_profiles")
            ),
            feedback=_resolve_path(project_root, storage_raw.get("feedback")),
            idempotency=_resolve_path(project_root, storage_raw.get("idempotency")),
        ),
        scheduler=SchedulerConfig(
            confidence_threshold=float(
                scheduler_raw.get("confidence_threshold", 0.75)
            ),
            jitter_guard=float(scheduler_raw.get("jitter_guard", 1.645)),
        ),
        cloud=cloud,
        network_probe=NetworkProbeConfig(
            interval_seconds=float(probe_raw.get("interval_seconds", 1.0)),
            window_size=int(probe_raw.get("window_size", 20)),
            failure_threshold=int(probe_raw.get("failure_threshold", 3)),
            uplink_mbps=float(probe_raw.get("uplink_mbps", 100.0)),
            downlink_mbps=float(probe_raw.get("downlink_mbps", 100.0)),
            expected_response_bytes=int(
                probe_raw.get("expected_response_bytes", 2048)
            ),
            cloud_queue_ms=float(probe_raw.get("cloud_queue_ms", 1.0)),
            cloud_compute_ms=float(probe_raw.get("cloud_compute_ms", 12.0)),
        ),
        replay=ReplayConfig(
            interval_seconds=float(replay_raw.get("interval_seconds", 1.0)),
            batch_size=int(replay_raw.get("batch_size", 64)),
            lease_seconds=float(replay_raw.get("lease_seconds", 30.0)),
            max_backoff_seconds=float(
                replay_raw.get("max_backoff_seconds", 60.0)
            ),
        ),
        idempotency=IdempotencyConfig(
            ttl_seconds=float(idempotency_raw.get("ttl_seconds", 86400.0)),
            max_entries=int(idempotency_raw.get("max_entries", 100000)),
        ),
        release_watch=release_watch,
        source_path=source_path,
    )
