"""Short-lived edge evidence cache and selective cloud pull protocol.

The normal data plane remains summary-first.  A cloud aggregation may use the
capability locator attached to a member event only after a shared road-set
trigger has been observed.  Locators are short lived, scoped to one
``scene/group/member/event`` tuple, and signed by the edge process that owns the
cache entry.
"""

from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import base64
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from cloud_edge_framework.artifacts import optional_artifact_path
from cloud_edge_framework.contracts import EVIDENCE_LEVELS, Evidence, SemanticEvent


EVIDENCE_CACHE_FETCH_ENDPOINT = "/api/v1/framework/evidence-cache/fetch"
EVIDENCE_CACHE_STATUS_ENDPOINT = "/api/v1/framework/evidence-cache"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _selected_level_evidence(
    event_payload: Mapping[str, Any], requested_level: str
) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in event_payload.get("evidence", [])
        if isinstance(item, dict)
        and str(item.get("level", "")) == requested_level
    ]


def _response_auth_material(
    *,
    cache_id: str,
    scene: str,
    group_key: str,
    member: str,
    event_id: str,
    requested_level: str,
    record_content_sha256: str,
    expires_at_ms: int,
    evidence_content_sha256: str,
) -> Dict[str, Any]:
    """Return the exact edge-secret response authentication contract.

    The response HMAC is deliberately independent of the visible locator
    token.  Its expected value is committed into the locator on the normal
    edge-to-cloud data plane, before any callback takes place.
    """

    return {
        "schema_version": 1,
        "cache_id": str(cache_id),
        "scene": str(scene),
        "group_key": str(group_key),
        "owner_member": str(member),
        "event_id": str(event_id),
        "requested_level": str(requested_level),
        "record_content_sha256": str(record_content_sha256),
        "expires_at_ms": int(expires_at_ms),
        "evidence_content_sha256": str(evidence_content_sha256),
    }


def _response_auth_tag(signing_key: bytes, material: Mapping[str, Any]) -> str:
    return hmac.new(
        signing_key,
        _canonical_bytes(material),
        hashlib.sha256,
    ).hexdigest()


def _normalized_base_url(value: str) -> str:
    raw = str(value).strip().rstrip("/")
    parsed = urlsplit(raw)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("evidence pull base URL must be an HTTP(S) origin")
    return "{}://{}".format(parsed.scheme.lower(), parsed.netloc.lower())


def _aggregation_identity(event: SemanticEvent) -> Tuple[str, str, List[str]]:
    raw = event.metadata.get("aggregation", {})
    if not isinstance(raw, dict):
        return "", "", []
    group_key = str(raw.get("key", raw.get("group_key", ""))).strip()
    member = str(raw.get("member", "")).strip()
    expected_raw = raw.get("expected_members", [])
    expected = (
        [str(value).strip() for value in expected_raw if str(value).strip()]
        if isinstance(expected_raw, list)
        else []
    )
    return group_key, member, expected


def _materialized_evidence(evidence: Evidence) -> Dict[str, Any]:
    """Return a cache-safe evidence descriptor.

    Referenced local files cannot be dereferenced by the cloud callback, so
    their bytes are retained only inside the bounded cache and returned as
    base64.  Inline traffic feature evidence stays in its original encoding.
    """

    value = evidence.to_dict()
    path = optional_artifact_path(evidence.uri)
    if path is None:
        return value
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if evidence.sha256 is not None and evidence.sha256 != digest:
        raise ValueError(
            "local evidence sha256 mismatch for {}".format(evidence.evidence_id)
        )
    if evidence.size_bytes not in {0, len(data)}:
        raise ValueError(
            "local evidence size mismatch for {}".format(evidence.evidence_id)
        )
    value.pop("uri", None)
    value["inline"] = {"base64": base64.b64encode(data).decode("ascii")}
    value["encoding"] = "{}+base64".format(evidence.encoding)
    value["size_bytes"] = len(data)
    value["sha256"] = digest
    return value


@dataclass(frozen=True)
class _CacheRecord:
    cache_id: str
    scene: str
    group_key: str
    member: str
    event_id: str
    expected_members: List[str]
    created_at_ms: int
    expires_at_ms: int
    content_sha256: str
    content_bytes: int
    event_payload: Dict[str, Any]
    allowed_levels: List[str]
    level_content_sha256: Dict[str, str]
    response_auth_tags: Dict[str, str]


class BoundedEvidenceCache:
    """In-memory TTL/LRU cache for evidence that was not sent normally."""

    def __init__(
        self,
        public_base_url: str,
        ttl_seconds: float = 10.0,
        max_entries: int = 2048,
        max_bytes: int = 64 * 1024 * 1024,
        signing_key: Optional[bytes] = None,
        clock_ms: Optional[Callable[[], int]] = None,
    ) -> None:
        self.public_base_url = _normalized_base_url(public_base_url)
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        if (
            not math.isfinite(self.ttl_seconds)
            or self.ttl_seconds <= 0
            or self.max_entries <= 0
            or self.max_bytes <= 0
        ):
            raise ValueError("evidence cache bounds must be positive")
        self._signing_key = signing_key or secrets.token_bytes(32)
        if len(self._signing_key) < 16:
            raise ValueError("evidence cache signing key must contain at least 16 bytes")
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._lock = threading.RLock()
        self._records: "OrderedDict[str, _CacheRecord]" = OrderedDict()
        self._stored_bytes = 0
        self._counters: Counter = Counter()

    @staticmethod
    def _locator_material(locator: Mapping[str, Any]) -> Dict[str, Any]:
        payload = {
            key: locator[key]
            for key in (
                "schema_version",
                "base_url",
                "endpoint",
                "cache_id",
                "scene",
                "group_key",
                "member",
                "event_id",
                "expected_members",
                "created_at_ms",
                "expires_at_ms",
                "allowed_levels",
                "content_sha256",
                "level_content_sha256",
                "response_auth_tags",
            )
        }
        return payload

    def _token(self, locator: Mapping[str, Any]) -> str:
        return hmac.new(
            self._signing_key,
            _canonical_bytes(self._locator_material(locator)),
            hashlib.sha256,
        ).hexdigest()

    def _remove_locked(self, cache_id: str, reason: str) -> None:
        record = self._records.pop(cache_id, None)
        if record is None:
            return
        self._stored_bytes -= record.content_bytes
        self._counters[reason] += 1

    def _purge_locked(self, now_ms: int) -> None:
        for cache_id, record in list(self._records.items()):
            if record.expires_at_ms <= now_ms:
                self._remove_locked(cache_id, "expired_total")

    def store(self, event: SemanticEvent) -> Optional[Dict[str, Any]]:
        group_key, member, expected_members = _aggregation_identity(event)
        if (
            not group_key
            or not member
            or member not in expected_members
            or len(set(expected_members)) != len(expected_members)
        ):
            with self._lock:
                self._counters["not_aggregated_total"] += 1
            return None

        event_payload = event.to_dict(include_scene_payload=True)
        event_payload["evidence"] = [
            _materialized_evidence(item) for item in event.evidence
        ]
        allowed_levels = [
            level
            for level in EVIDENCE_LEVELS
            if any(item.level == level for item in event.evidence)
        ]
        # A locator with no richer evidence would create a callback path that
        # can never improve a summary decision.
        if not any(level in {"feature", "raw"} for level in allowed_levels):
            with self._lock:
                self._counters["no_richer_evidence_total"] += 1
            return None

        content = {
            "scene": event.scene,
            "group_key": group_key,
            "member": member,
            "event_id": event.event_id,
            "expected_members": sorted(expected_members),
            "event": event_payload,
        }
        content_bytes = len(_canonical_bytes(content))
        if content_bytes > self.max_bytes:
            with self._lock:
                self._counters["oversize_rejected_total"] += 1
            return None
        content_sha256 = _sha256(content)
        now_ms = int(self._clock_ms())
        expires_at_ms = now_ms + max(1, int(self.ttl_seconds * 1000.0))
        cache_id = "evidence_cache_{}".format(
            hashlib.sha256(
                "\x1f".join(
                    [
                        event.scene,
                        group_key,
                        member,
                        event.event_id,
                        content_sha256,
                        str(now_ms),
                        secrets.token_hex(8),
                    ]
                ).encode("utf-8")
            ).hexdigest()[:32]
        )
        level_content_sha256 = {
            level: _sha256(_selected_level_evidence(event_payload, level))
            for level in ("feature", "raw")
            if level in allowed_levels
            and _selected_level_evidence(event_payload, level)
        }
        response_auth_tags = {
            level: _response_auth_tag(
                self._signing_key,
                _response_auth_material(
                    cache_id=cache_id,
                    scene=event.scene,
                    group_key=group_key,
                    member=member,
                    event_id=event.event_id,
                    requested_level=level,
                    record_content_sha256=content_sha256,
                    expires_at_ms=expires_at_ms,
                    evidence_content_sha256=evidence_sha256,
                ),
            )
            for level, evidence_sha256 in level_content_sha256.items()
        }
        record = _CacheRecord(
            cache_id=cache_id,
            scene=event.scene,
            group_key=group_key,
            member=member,
            event_id=event.event_id,
            expected_members=sorted(expected_members),
            created_at_ms=now_ms,
            expires_at_ms=expires_at_ms,
            content_sha256=content_sha256,
            content_bytes=content_bytes,
            event_payload=event_payload,
            allowed_levels=allowed_levels,
            level_content_sha256=level_content_sha256,
            response_auth_tags=response_auth_tags,
        )
        with self._lock:
            self._purge_locked(now_ms)
            while (
                len(self._records) >= self.max_entries
                or self._stored_bytes + content_bytes > self.max_bytes
            ):
                oldest = next(iter(self._records), None)
                if oldest is None:
                    break
                self._remove_locked(oldest, "capacity_evictions_total")
            self._records[cache_id] = record
            self._stored_bytes += content_bytes
            self._counters["stored_total"] += 1

        locator: Dict[str, Any] = {
            "schema_version": 1,
            "base_url": self.public_base_url,
            "endpoint": EVIDENCE_CACHE_FETCH_ENDPOINT,
            "cache_id": cache_id,
            "scene": event.scene,
            "group_key": group_key,
            "member": member,
            "event_id": event.event_id,
            "expected_members": sorted(expected_members),
            "created_at_ms": now_ms,
            "expires_at_ms": expires_at_ms,
            "allowed_levels": list(allowed_levels),
            "content_sha256": content_sha256,
            "level_content_sha256": dict(level_content_sha256),
            "response_auth_tags": dict(response_auth_tags),
        }
        locator["token"] = self._token(locator)
        return locator

    def fetch(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("evidence fetch request must be an object")
        required = {
            "cache_id",
            "scene",
            "group_key",
            "member",
            "event_id",
            "expected_members",
            "requested_level",
            "expires_at_ms",
            "content_sha256",
            "level_content_sha256",
            "response_auth_tags",
            "token",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                "evidence fetch request is missing {}".format(", ".join(missing))
            )
        requested_level = str(payload["requested_level"])
        if requested_level not in {"feature", "raw"}:
            raise ValueError("evidence fetch level must be feature or raw")
        cache_id = str(payload["cache_id"])
        now_ms = int(self._clock_ms())
        with self._lock:
            self._purge_locked(now_ms)
            record = self._records.get(cache_id)
            if record is None:
                self._counters["fetch_misses_total"] += 1
                raise KeyError("evidence cache entry is unavailable or expired")

            locator = {
                "schema_version": 1,
                "base_url": self.public_base_url,
                "endpoint": EVIDENCE_CACHE_FETCH_ENDPOINT,
                "cache_id": record.cache_id,
                "scene": record.scene,
                "group_key": record.group_key,
                "member": record.member,
                "event_id": record.event_id,
                "expected_members": list(record.expected_members),
                "created_at_ms": record.created_at_ms,
                "expires_at_ms": record.expires_at_ms,
                "allowed_levels": list(record.allowed_levels),
                "content_sha256": record.content_sha256,
                "level_content_sha256": dict(record.level_content_sha256),
                "response_auth_tags": dict(record.response_auth_tags),
            }
            supplied_token = str(payload["token"])
            if not hmac.compare_digest(supplied_token, self._token(locator)):
                self._counters["authorization_failures_total"] += 1
                raise ValueError("evidence pull capability token is invalid")
            identities = {
                "scene": record.scene,
                "group_key": record.group_key,
                "member": record.member,
                "event_id": record.event_id,
                "content_sha256": record.content_sha256,
            }
            if any(str(payload[key]) != value for key, value in identities.items()):
                self._counters["authorization_failures_total"] += 1
                raise ValueError("evidence pull identity is outside the authorized group")
            expected = payload["expected_members"]
            if (
                not isinstance(expected, list)
                or sorted(str(value) for value in expected)
                != record.expected_members
                or int(payload["expires_at_ms"]) != record.expires_at_ms
                or payload["level_content_sha256"]
                != record.level_content_sha256
                or payload["response_auth_tags"] != record.response_auth_tags
            ):
                self._counters["authorization_failures_total"] += 1
                raise ValueError("evidence pull group authorization is invalid")
            if requested_level not in record.allowed_levels:
                self._counters["unavailable_level_total"] += 1
                raise KeyError(
                    "requested {} evidence was never present at the edge".format(
                        requested_level
                    )
                )

            selected = _selected_level_evidence(
                record.event_payload, requested_level
            )
            if not selected:
                self._counters["unavailable_level_total"] += 1
                raise KeyError("requested evidence level is unavailable")
            self._records.move_to_end(cache_id)
            self._counters["fetch_successes_total"] += 1

        evidence_content_sha256 = _sha256(selected)
        if not hmac.compare_digest(
            evidence_content_sha256,
            record.level_content_sha256.get(requested_level, ""),
        ):
            raise ValueError("cached evidence content changed after locator issue")
        response: Dict[str, Any] = {
            "schema_version": 1,
            "cache_id": record.cache_id,
            "scene": record.scene,
            "group_key": record.group_key,
            "member": record.member,
            "event_id": record.event_id,
            "requested_level": requested_level,
            "evidence": selected,
            "record_content_sha256": record.content_sha256,
            "created_at_ms": record.created_at_ms,
            "expires_at_ms": record.expires_at_ms,
            "evidence_content_sha256": evidence_content_sha256,
            "response_auth_tag": record.response_auth_tags[requested_level],
        }
        response["bundle_sha256"] = _sha256(response)
        return response

    def snapshot(self) -> Dict[str, Any]:
        now_ms = int(self._clock_ms())
        with self._lock:
            self._purge_locked(now_ms)
            return {
                "enabled": True,
                "entry_count": len(self._records),
                "stored_bytes": self._stored_bytes,
                "ttl_seconds": self.ttl_seconds,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "counters": dict(self._counters),
            }


@dataclass(frozen=True)
class PulledEvidence:
    scene: str
    group_key: str
    member: str
    event_id: str
    requested_level: str
    evidence: List[Evidence]
    response_bytes: int
    latency_ms: float


def _validate_pull_response(
    response: Mapping[str, Any],
    locator: Mapping[str, Any],
    requested_level: str,
    latency_ms: float,
) -> PulledEvidence:
    if not isinstance(response, Mapping):
        raise ValueError("edge evidence response must be an object")
    raw = dict(response)
    supplied_bundle_sha = str(raw.pop("bundle_sha256", ""))
    if not supplied_bundle_sha or not hmac.compare_digest(
        supplied_bundle_sha, _sha256(raw)
    ):
        raise ValueError("edge evidence response sha256 is invalid")
    identity_pairs = (
        ("cache_id", "cache_id"),
        ("scene", "scene"),
        ("group_key", "group_key"),
        ("member", "member"),
        ("event_id", "event_id"),
        ("record_content_sha256", "content_sha256"),
    )
    for response_key, locator_key in identity_pairs:
        if str(raw.get(response_key, "")) != str(locator.get(locator_key, "")):
            raise ValueError("edge evidence response identity mismatch")
    if int(raw.get("created_at_ms", -1)) != int(locator.get("created_at_ms", -2)):
        raise ValueError("edge evidence response creation time mismatch")
    if int(raw.get("expires_at_ms", -1)) != int(locator.get("expires_at_ms", -2)):
        raise ValueError("edge evidence response expiry mismatch")
    if str(raw.get("requested_level", "")) != requested_level:
        raise ValueError("edge evidence response level mismatch")
    evidence_raw = raw.get("evidence")
    if not isinstance(evidence_raw, list) or not evidence_raw:
        raise ValueError("edge evidence response contains no evidence")
    evidence_content_sha256 = _sha256(evidence_raw)
    supplied_evidence_sha256 = str(raw.get("evidence_content_sha256", ""))
    level_content_raw = locator.get("level_content_sha256", {})
    if not isinstance(level_content_raw, Mapping):
        raise ValueError("edge evidence locator has no signed level digest")
    committed_evidence_sha256 = str(level_content_raw.get(requested_level, ""))
    if (
        not supplied_evidence_sha256
        or not committed_evidence_sha256
        or not hmac.compare_digest(
            supplied_evidence_sha256, evidence_content_sha256
        )
        or not hmac.compare_digest(
            supplied_evidence_sha256, committed_evidence_sha256
        )
    ):
        raise ValueError(
            "edge evidence response HMAC-bound content sha256 is invalid"
        )
    response_auth_raw = locator.get("response_auth_tags", {})
    if not isinstance(response_auth_raw, Mapping):
        raise ValueError("edge evidence locator has no response HMAC commitment")
    committed_auth_tag = str(response_auth_raw.get(requested_level, ""))
    supplied_auth_tag = str(raw.get("response_auth_tag", ""))
    if (
        not committed_auth_tag
        or not supplied_auth_tag
        or not hmac.compare_digest(supplied_auth_tag, committed_auth_tag)
    ):
        raise ValueError("edge evidence response HMAC is invalid")
    evidence = [Evidence.from_dict(item) for item in evidence_raw]
    if any(item.level != requested_level for item in evidence):
        raise ValueError("edge evidence response contains an unauthorized level")
    return PulledEvidence(
        scene=str(raw["scene"]),
        group_key=str(raw["group_key"]),
        member=str(raw["member"]),
        event_id=str(raw["event_id"]),
        requested_level=requested_level,
        evidence=evidence,
        response_bytes=len(_canonical_bytes(response)),
        latency_ms=max(0.0, float(latency_ms)),
    )


class HttpEvidencePullClient:
    """Fetch capability-bound evidence only from configured edge origins."""

    def __init__(
        self,
        allowed_edge_base_urls: Sequence[str],
        timeout_seconds: float = 0.05,
        max_response_bytes: int = 8 * 1024 * 1024,
        opener: Optional[
            Callable[[str, Dict[str, Any], float, int], Mapping[str, Any]]
        ] = None,
    ) -> None:
        self.allowed_edge_base_urls = {
            _normalized_base_url(value) for value in allowed_edge_base_urls
        }
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        if (
            not self.allowed_edge_base_urls
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or self.max_response_bytes <= 0
        ):
            raise ValueError("evidence pull client settings are invalid")
        self._opener = opener

    @staticmethod
    def _request_payload(
        locator: Mapping[str, Any], requested_level: str
    ) -> Dict[str, Any]:
        required = {
            "schema_version",
            "base_url",
            "endpoint",
            "cache_id",
            "scene",
            "group_key",
            "member",
            "event_id",
            "expected_members",
            "created_at_ms",
            "expires_at_ms",
            "allowed_levels",
            "content_sha256",
            "level_content_sha256",
            "response_auth_tags",
            "token",
        }
        if not isinstance(locator, Mapping) or required - set(locator):
            raise ValueError("edge evidence locator is incomplete")
        if requested_level not in {"feature", "raw"}:
            raise ValueError("requested evidence level must be feature or raw")
        if requested_level not in locator["allowed_levels"]:
            raise KeyError("requested evidence level is unavailable at the edge")
        payload = {
            key: locator[key]
            for key in (
                "cache_id",
                "scene",
                "group_key",
                "member",
                "event_id",
                "expected_members",
                "expires_at_ms",
                "content_sha256",
                "level_content_sha256",
                "response_auth_tags",
                "token",
            )
        }
        payload["requested_level"] = requested_level
        return payload

    def fetch(
        self, locator: Mapping[str, Any], requested_level: str
    ) -> PulledEvidence:
        base_url = _normalized_base_url(str(locator.get("base_url", "")))
        if base_url not in self.allowed_edge_base_urls:
            raise ValueError("edge evidence locator origin is not allowlisted")
        if str(locator.get("endpoint", "")) != EVIDENCE_CACHE_FETCH_ENDPOINT:
            raise ValueError("edge evidence locator endpoint is invalid")
        expires_at_ms = int(locator.get("expires_at_ms", 0))
        if expires_at_ms <= int(time.time() * 1000):
            raise TimeoutError("edge evidence locator has expired")
        payload = self._request_payload(locator, requested_level)
        started = time.perf_counter()
        if self._opener is not None:
            response = self._opener(
                base_url + EVIDENCE_CACHE_FETCH_ENDPOINT,
                payload,
                self.timeout_seconds,
                self.max_response_bytes,
            )
        else:
            body = _canonical_bytes(payload)
            request = Request(
                base_url + EVIDENCE_CACHE_FETCH_ENDPOINT,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as http_response:
                    response_body = http_response.read(self.max_response_bytes + 1)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    "edge evidence pull returned HTTP {}: {}".format(
                        exc.code, detail
                    )
                ) from exc
            except (TimeoutError, URLError, OSError) as exc:
                raise TimeoutError("edge evidence pull failed: {}".format(exc)) from exc
            if len(response_body) > self.max_response_bytes:
                raise ValueError("edge evidence response exceeds configured limit")
            try:
                response = json.loads(response_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("edge evidence response is not valid JSON") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        return _validate_pull_response(
            response, locator, requested_level, latency_ms
        )


@dataclass(frozen=True)
class PullTarget:
    road_set_id: str
    road_set_ids: List[str]
    member: str
    event_id: str
    requested_level: str
    locator: Dict[str, Any]


@dataclass(frozen=True)
class PullPlan:
    triggered: bool
    trigger_reasons: List[str]
    road_set_ids: List[str]
    requested_members: List[str]
    requested_level: Optional[str]
    road_set_members: Dict[str, List[str]]
    road_set_levels: Dict[str, str]
    initial_evidence_sufficient: bool
    targets: List[PullTarget]
    no_pull_reason: str


def _road_set_contexts(event: SemanticEvent) -> List[Dict[str, Any]]:
    raw = event.metadata.get("road_set_contexts")
    if raw is None:
        raw = event.scene_payload.get("road_set_contexts")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _normalized_road_set_id(value: Any, require_resource_prefix: bool = False) -> str:
    raw = str(value).strip()
    prefix = "traffic_road_set:"
    if raw.startswith(prefix):
        return raw[len(prefix):].strip()
    return "" if require_resource_prefix else raw


class SelectiveEvidencePullPlanner:
    """Plan exactly two owners per triggered road set with bounded de-duplication."""

    def __init__(
        self,
        max_members_per_group: int = 2,
        max_unique_members_per_group: int = 4,
        max_road_sets_per_group: int = 5,
    ) -> None:
        # The compatibility/config name means owners *per road set*.  A four
        # edge aggregation may contain several independently triggered road
        # sets; their HTTP targets are de-duplicated by member below.
        self.max_members_per_road_set = int(max_members_per_group)
        self.max_unique_members_per_group = int(max_unique_members_per_group)
        self.max_road_sets_per_group = int(max_road_sets_per_group)
        if self.max_members_per_road_set != 2:
            raise ValueError("selective road-set pull requires exactly two owners")
        if self.max_unique_members_per_group <= 0:
            raise ValueError("selective pull unique-member budget must be positive")
        if self.max_road_sets_per_group <= 0:
            raise ValueError("selective pull road-set budget must be positive")

    def plan(
        self,
        events: Sequence[SemanticEvent],
        initial_coordination: Mapping[str, Any],
    ) -> PullPlan:
        def result(
            triggered: bool,
            trigger_reasons: Optional[List[str]] = None,
            road_set_ids: Optional[List[str]] = None,
            requested_members: Optional[List[str]] = None,
            requested_level: Optional[str] = None,
            road_set_members: Optional[Dict[str, List[str]]] = None,
            road_set_levels: Optional[Dict[str, str]] = None,
            initial_evidence_sufficient: bool = False,
            targets: Optional[List[PullTarget]] = None,
            no_pull_reason: str = "",
        ) -> PullPlan:
            return PullPlan(
                triggered=triggered,
                trigger_reasons=list(trigger_reasons or []),
                road_set_ids=list(road_set_ids or []),
                requested_members=list(requested_members or []),
                requested_level=requested_level,
                road_set_members=dict(road_set_members or {}),
                road_set_levels=dict(road_set_levels or {}),
                initial_evidence_sufficient=bool(initial_evidence_sufficient),
                targets=list(targets or []),
                no_pull_reason=no_pull_reason,
            )

        normalized = list(events)
        if not normalized:
            raise ValueError("selective evidence pull requires events")
        member_events: Dict[str, SemanticEvent] = {}
        group_keys = set()
        for event in normalized:
            group_key, member, expected = _aggregation_identity(event)
            if not group_key or not member or member not in expected:
                return result(
                    False, no_pull_reason="untrusted_aggregation_identity"
                )
            if member in member_events:
                return result(False, no_pull_reason="duplicate_aggregation_member")
            member_events[member] = event
            group_keys.add(group_key)
        if len(group_keys) != 1:
            return result(False, no_pull_reason="mixed_group_keys")

        contexts_by_road_set: Dict[str, Dict[str, Any]] = {}
        for event in normalized:
            _, member, _ = _aggregation_identity(event)
            for context in _road_set_contexts(event):
                road_set_id = _normalized_road_set_id(
                    context.get("road_set_id", "")
                )
                if not road_set_id:
                    continue
                state = contexts_by_road_set.setdefault(
                    road_set_id,
                    {
                        "members": set(),
                        "conflict": False,
                        "levels": [],
                        "reasons": [],
                        "invalid_level": False,
                    },
                )
                raw_members = context.get("required_members", [])
                if isinstance(raw_members, list):
                    state["members"].update(
                        str(value).strip()
                        for value in raw_members
                        if str(value).strip()
                    )
                state["members"].add(member)
                if context.get("road_set_conflict_suspected") is True:
                    state["conflict"] = True
                    state["reasons"].append("road_set_conflict_suspected")
                level = str(context.get("required_evidence_level", "")).strip()
                if level:
                    if level not in {"feature", "raw"}:
                        state["invalid_level"] = True
                        state["reasons"].append(
                            "invalid_required_evidence_level"
                        )
                    else:
                        state["levels"].append(level)
                        state["reasons"].append("road_set_evidence_required")

        event_member = {
            event.event_id: _aggregation_identity(event)[1] for event in normalized
        }
        # Scene fusion runs inside the first cloud coordination pass, so its
        # two-sided road-set assessment is exposed on each decision rather
        # than mutating the durable ingress events.  Consume that audited
        # decision metadata before examining generic action conflicts.
        raw_decisions = initial_coordination.get("decisions", [])
        if isinstance(raw_decisions, list):
            for raw_decision in raw_decisions:
                if not isinstance(raw_decision, dict):
                    continue
                decision_event_ids = raw_decision.get("event_ids", [])
                decision_member = ""
                if isinstance(decision_event_ids, list):
                    for event_id in decision_event_ids:
                        decision_member = event_member.get(str(event_id), "")
                        if decision_member:
                            break
                metadata = raw_decision.get("metadata", {})
                if not isinstance(metadata, dict):
                    continue
                road_set_decisions = metadata.get("road_set_decisions", [])
                if not isinstance(road_set_decisions, list):
                    continue
                for road_set_decision in road_set_decisions:
                    if not isinstance(road_set_decision, dict):
                        continue
                    road_set_id = _normalized_road_set_id(
                        road_set_decision.get("road_set_id", "")
                    )
                    if not road_set_id:
                        continue
                    state = contexts_by_road_set.setdefault(
                        road_set_id,
                        {
                            "members": set(),
                            "conflict": False,
                            "levels": [],
                            "reasons": [],
                            "invalid_level": False,
                        },
                    )
                    raw_members = road_set_decision.get("required_members", [])
                    if isinstance(raw_members, list):
                        state["members"].update(
                            str(value).strip()
                            for value in raw_members
                            if str(value).strip()
                        )
                    if decision_member:
                        state["members"].add(decision_member)
                    level = str(
                        road_set_decision.get("required_evidence_level", "")
                    ).strip()
                    if level:
                        if level not in {"feature", "raw"}:
                            state["invalid_level"] = True
                            state["reasons"].append(
                                "invalid_required_evidence_level"
                            )
                        else:
                            state["levels"].append(level)
                            state["reasons"].append(
                                "road_set_evidence_required"
                            )
                    if road_set_decision.get("road_set_conflict_suspected") is True:
                        state["conflict"] = True
                        state["reasons"].append(
                            "cloud_road_set_assessment_disagreement"
                        )
        conflicts = initial_coordination.get("initial_conflicts", [])
        if isinstance(conflicts, list):
            for conflict in conflicts:
                if not isinstance(conflict, dict):
                    continue
                shared = conflict.get("shared_resources", [])
                if not isinstance(shared, list) or not shared:
                    continue
                left_member = event_member.get(str(conflict.get("left_event_id", "")))
                right_member = event_member.get(str(conflict.get("right_event_id", "")))
                if not left_member or not right_member or left_member == right_member:
                    continue
                for resource in shared:
                    # Other shared actuators (for example a bare traffic node)
                    # are not authority to pull a neighboring member's cached
                    # context.  Only the explicit two-owner road-set namespace
                    # participates in this protocol.
                    road_set_id = _normalized_road_set_id(
                        resource, require_resource_prefix=True
                    )
                    if not road_set_id:
                        continue
                    state = contexts_by_road_set.setdefault(
                        road_set_id,
                        {
                            "members": set(),
                            "conflict": False,
                            "levels": [],
                            "reasons": [],
                            "invalid_level": False,
                        },
                    )
                    state["members"].update([left_member, right_member])
                    state["conflict"] = True
                    state["reasons"].append("cloud_action_conflict")

        # A requested level selects *what* to fetch after a conflict; it is
        # never authority to create a callback on its own.  This keeps the
        # ordinary summary path network-silent even if an edge accidentally
        # carries a stale evidence-level hint.
        triggered = []
        for road_set_id, state in sorted(contexts_by_road_set.items()):
            if state["conflict"]:
                triggered.append((road_set_id, state))
        if not triggered:
            return result(False, no_pull_reason="no_road_set_conflict_trigger")
        road_set_ids = [road_set_id for road_set_id, _ in triggered]
        reasons = sorted(
            {
                str(reason)
                for _, state in triggered
                for reason in state["reasons"]
            }
        )
        road_set_members = {
            road_set_id: sorted(state["members"])
            for road_set_id, state in triggered
        }
        road_set_levels = {
            road_set_id: (
                "raw" if "raw" in state["levels"] else "feature"
            )
            for road_set_id, state in triggered
        }
        requested_members = sorted(
            {
                member
                for members in road_set_members.values()
                for member in members
            }
        )
        if len(triggered) > self.max_road_sets_per_group:
            return result(
                True,
                reasons,
                road_set_ids,
                requested_members,
                road_set_members=road_set_members,
                road_set_levels=road_set_levels,
                no_pull_reason="triggered_road_sets_exceed_group_budget",
            )
        if any(state["invalid_level"] for _, state in triggered):
            return result(
                True,
                reasons,
                road_set_ids,
                requested_members,
                road_set_members=road_set_members,
                road_set_levels=road_set_levels,
                no_pull_reason="invalid_required_evidence_level",
            )
        if any(
            len(members) != self.max_members_per_road_set
            for members in road_set_members.values()
        ):
            return result(
                True,
                reasons,
                road_set_ids,
                requested_members,
                road_set_members=road_set_members,
                road_set_levels=road_set_levels,
                no_pull_reason="road_set_requires_exactly_two_members",
            )
        if any(member not in member_events for member in requested_members):
            return result(
                True,
                reasons,
                road_set_ids,
                requested_members,
                road_set_members=road_set_members,
                road_set_levels=road_set_levels,
                no_pull_reason="road_set_member_not_in_aggregation",
            )
        if len(requested_members) > self.max_unique_members_per_group:
            return result(
                True,
                reasons,
                road_set_ids,
                requested_members,
                road_set_members=road_set_members,
                road_set_levels=road_set_levels,
                no_pull_reason="unique_members_exceed_group_budget",
            )

        member_road_sets: Dict[str, List[str]] = {
            member: sorted(
                road_set_id
                for road_set_id, members in road_set_members.items()
                if member in members
            )
            for member in requested_members
        }
        member_levels = {
            member: (
                "raw"
                if any(
                    road_set_levels[road_set_id] == "raw"
                    for road_set_id in member_road_sets[member]
                )
                else "feature"
            )
            for member in requested_members
        }
        distinct_levels = sorted(set(member_levels.values()))
        requested_level = (
            distinct_levels[0] if len(distinct_levels) == 1 else "mixed"
        )
        targets: List[PullTarget] = []
        for member in requested_members:
            event = member_events[member]
            member_level = member_levels[member]
            if any(item.level == member_level for item in event.evidence):
                # This member already uploaded the exact evidence consumed by
                # the first cloud pass.  Keep it in the road-set audit pair but
                # do not waste a callback on duplicate bytes.
                continue
            locator = event.metadata.get("evidence_pull_locator")
            if not isinstance(locator, dict):
                return result(
                    True,
                    reasons,
                    road_set_ids,
                    requested_members,
                    requested_level,
                    road_set_members,
                    road_set_levels,
                    no_pull_reason="member_locator_unavailable",
                )
            if (
                str(locator.get("scene", "")) != event.scene
                or str(locator.get("group_key", "")) not in group_keys
                or str(locator.get("member", "")) != member
                or str(locator.get("event_id", "")) != event.event_id
            ):
                return result(
                    True,
                    reasons,
                    road_set_ids,
                    requested_members,
                    requested_level,
                    road_set_members,
                    road_set_levels,
                    no_pull_reason="member_locator_identity_mismatch",
                )
            if member_level not in locator.get("allowed_levels", []):
                return result(
                    True,
                    reasons,
                    road_set_ids,
                    requested_members,
                    requested_level,
                    road_set_members,
                    road_set_levels,
                    no_pull_reason="requested_level_unavailable",
                )
            targets.append(
                PullTarget(
                    road_set_id=member_road_sets[member][0],
                    road_set_ids=member_road_sets[member],
                    member=member,
                    event_id=event.event_id,
                    requested_level=member_level,
                    locator=dict(locator),
                )
            )
        return result(
            True,
            reasons,
            road_set_ids,
            requested_members,
            requested_level,
            road_set_members,
            road_set_levels,
            initial_evidence_sufficient=not targets,
            targets=targets,
            no_pull_reason=(
                "required_evidence_already_present" if not targets else ""
            ),
        )


def fetch_pull_plan(
    client: HttpEvidencePullClient,
    plan: PullPlan,
) -> Tuple[Dict[str, PulledEvidence], List[str]]:
    """Fetch the de-duplicated member plan concurrently and isolate failures."""

    if not plan.targets:
        return {}, []
    results: Dict[str, PulledEvidence] = {}
    errors: List[str] = []
    with ThreadPoolExecutor(max_workers=len(plan.targets)) as executor:
        futures = {
            executor.submit(
                client.fetch, target.locator, target.requested_level
            ): target
            for target in plan.targets
        }
        for future in as_completed(futures):
            target = futures[future]
            try:
                results[target.member] = future.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    "{}:{}: {}".format(
                        target.member, type(exc).__name__, exc
                    )
                )
    return results, sorted(errors)


def merge_pulled_evidence(
    events: Sequence[SemanticEvent],
    pulled: Mapping[str, PulledEvidence],
) -> List[SemanticEvent]:
    enriched = []
    for event in events:
        _, member, _ = _aggregation_identity(event)
        item = pulled.get(member)
        if item is None:
            enriched.append(event)
            continue
        if (
            item.event_id != event.event_id
            or item.scene != event.scene
            or item.group_key != _aggregation_identity(event)[0]
        ):
            raise ValueError("pulled evidence cannot be merged across event identity")
        evidence_by_id = {value.evidence_id: value for value in event.evidence}
        for evidence in item.evidence:
            existing = evidence_by_id.get(evidence.evidence_id)
            if existing is not None and existing.to_dict() != evidence.to_dict():
                raise ValueError("pulled evidence id collides with uploaded evidence")
            evidence_by_id[evidence.evidence_id] = evidence
        metadata = dict(event.metadata)
        metadata["selective_evidence_pull"] = {
            "member": member,
            "requested_level": item.requested_level,
            "evidence_ids": [value.evidence_id for value in item.evidence],
            "latency_ms": round(item.latency_ms, 6),
            "response_bytes": item.response_bytes,
        }
        enriched.append(
            replace(
                event,
                evidence=list(evidence_by_id.values()),
                metadata=metadata,
            )
        )
    return enriched
