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
    artifacts: Optional[Path] = None
    reviews: Optional[Path] = None
    aggregations: Optional[Path] = None
    monitoring: Optional[Path] = None


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
    waiting_poll_seconds: float = 0.025
    partial_poll_seconds: float = 1.0
    aggregation_max_wait_seconds: float = 10.0
    reconciliation_poll_seconds: float = 5.0
    reconciliation_max_wait_seconds: float = 60.0


@dataclass(frozen=True)
class IdempotencyConfig:
    ttl_seconds: float
    max_entries: int


@dataclass(frozen=True)
class CloudLLMConfig:
    enabled: bool
    runtime_config: Optional[Path]
    min_risk_level: str


@dataclass(frozen=True)
class MonitoringConfig:
    enabled: bool
    window_size: int
    bins: int
    min_labeled_samples: int
    min_drift_samples: int
    bootstrap_reference_size: int
    max_ece: float
    target_coverage: float
    coverage_tolerance: float
    max_psi: float
    evaluation_interval_events: int
    evaluation_max_staleness_ms: int


@dataclass(frozen=True)
class UtilityRouterConfig:
    enabled: bool
    artifact: Optional[Path]
    mode: str


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
    cloud_llm: Optional[CloudLLMConfig] = None
    monitoring: Optional[MonitoringConfig] = None
    utility_router: Optional[UtilityRouterConfig] = None


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
    cloud_llm_raw = value.get("cloud_llm")
    monitoring_raw = value.get("monitoring")
    utility_router_raw = value.get("utility_router")

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

    cloud_llm = None
    if cloud_llm_raw is not None:
        if role != "cloud":
            raise ValueError("cloud_llm may only be configured on the cloud service")
        llm_data = dict(cloud_llm_raw)
        enabled = bool(llm_data.get("enabled", False))
        runtime_config = _resolve_path(
            project_root, llm_data.get("runtime_config")
        )
        min_risk_level = str(llm_data.get("min_risk_level", "high"))
        if min_risk_level not in {"low", "medium", "high", "severe"}:
            raise ValueError("cloud_llm min_risk_level is invalid")
        if enabled and runtime_config is None:
            raise ValueError("enabled cloud_llm requires runtime_config")
        cloud_llm = CloudLLMConfig(
            enabled=enabled,
            runtime_config=runtime_config,
            min_risk_level=min_risk_level,
        )

    monitoring_data = dict(monitoring_raw or {})
    monitoring = MonitoringConfig(
        enabled=bool(monitoring_data.get("enabled", role == "edge")),
        window_size=int(monitoring_data.get("window_size", 500)),
        bins=int(monitoring_data.get("bins", 10)),
        min_labeled_samples=int(monitoring_data.get("min_labeled_samples", 50)),
        min_drift_samples=int(monitoring_data.get("min_drift_samples", 50)),
        bootstrap_reference_size=int(
            monitoring_data.get("bootstrap_reference_size", 200)
        ),
        max_ece=float(monitoring_data.get("max_ece", 0.10)),
        target_coverage=float(monitoring_data.get("target_coverage", 0.90)),
        coverage_tolerance=float(
            monitoring_data.get("coverage_tolerance", 0.05)
        ),
        max_psi=float(monitoring_data.get("max_psi", 0.20)),
        evaluation_interval_events=int(
            monitoring_data.get("evaluation_interval_events", 25)
        ),
        evaluation_max_staleness_ms=int(
            monitoring_data.get("evaluation_max_staleness_ms", 1000)
        ),
    )

    utility_router = None
    if utility_router_raw is not None:
        if role != "edge":
            raise ValueError("utility_router may only be configured on the edge service")
        router_data = dict(utility_router_raw)
        router_enabled = bool(router_data.get("enabled", False))
        router_artifact = _resolve_path(project_root, router_data.get("artifact"))
        router_mode = str(router_data.get("mode", "shadow"))
        if router_mode not in {"shadow", "active"}:
            raise ValueError("utility_router mode must be shadow or active")
        if router_enabled and router_artifact is None:
            raise ValueError("enabled utility_router requires artifact")
        utility_router = UtilityRouterConfig(
            enabled=router_enabled,
            artifact=router_artifact,
            mode=router_mode,
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
            artifacts=_resolve_path(project_root, storage_raw.get("artifacts")),
            reviews=_resolve_path(project_root, storage_raw.get("reviews")),
            aggregations=_resolve_path(project_root, storage_raw.get("aggregations")),
            monitoring=_resolve_path(project_root, storage_raw.get("monitoring")),
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
            waiting_poll_seconds=float(
                replay_raw.get("waiting_poll_seconds", 0.025)
            ),
            partial_poll_seconds=float(
                replay_raw.get("partial_poll_seconds", 1.0)
            ),
            aggregation_max_wait_seconds=float(
                replay_raw.get("aggregation_max_wait_seconds", 10.0)
            ),
            reconciliation_poll_seconds=float(
                replay_raw.get("reconciliation_poll_seconds", 5.0)
            ),
            reconciliation_max_wait_seconds=float(
                replay_raw.get("reconciliation_max_wait_seconds", 60.0)
            ),
        ),
        idempotency=IdempotencyConfig(
            ttl_seconds=float(idempotency_raw.get("ttl_seconds", 86400.0)),
            max_entries=int(idempotency_raw.get("max_entries", 100000)),
        ),
        release_watch=release_watch,
        cloud_llm=cloud_llm,
        monitoring=monitoring,
        utility_router=utility_router,
        source_path=source_path,
    )
