"""用途：从云端拉取策略与边缘模型清单，校验后原子更新，并在健康检查失败时回滚。"""

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from traffic_system.decision_utils import save_json
from traffic_system.policy_store import PolicyStore, verify_bundle, version_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize a verified cloud policy to an edge node.")
    parser.add_argument("--policy_url", default="http://192.168.31.135:18080/api/v1/policy")
    parser.add_argument("--artifact_base_url", default="")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--current", default="deployment/policy/current_policy.json")
    parser.add_argument("--target", default="edge", choices=("edge", "cloud"))
    parser.add_argument("--health_url", default="http://127.0.0.1:18190/health")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--full_artifact_verify", action="store_true")
    parser.add_argument("--output_json", default="results/edge/remote_policy_sync.json")
    return parser.parse_args()


def default_artifact_base(policy_url: str) -> str:
    parts = urlsplit(policy_url)
    return urlunsplit((parts.scheme, parts.netloc, "/api/v1/artifacts", "", ""))


def fetch_json(url: str, timeout: float) -> Tuple[Dict[str, Any], int]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Cloud policy response must be a JSON object")
    return value, len(body)


def safe_artifact_path(project_root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
        raise ValueError("Unsafe artifact path: {}".format(relative_path))
    target = (project_root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError("Artifact escapes project root: {}".format(relative_path)) from exc
    return target


def file_matches(path: Path, expected_size: int, expected_sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected_sha256


def unchanged_artifact_is_trusted(
    path: Path,
    artifact: Dict[str, Any],
    current_artifacts: Dict[str, Dict[str, Any]],
    full_verify: bool,
) -> Tuple[bool, str]:
    expected_size = int(artifact["size_bytes"])
    expected_sha256 = str(artifact["sha256"])
    if full_verify:
        return file_matches(path, expected_size, expected_sha256), "full_sha256"
    previous = current_artifacts.get(str(artifact["path"]))
    if (
        previous
        and str(previous.get("sha256")) == expected_sha256
        and int(previous.get("size_bytes", -1)) == expected_size
        and path.is_file()
        and path.stat().st_size == expected_size
    ):
        return True, "trusted_unchanged_manifest"
    return file_matches(path, expected_size, expected_sha256), "full_sha256"


def stage_download(
    artifact_base_url: str,
    artifact: Dict[str, Any],
    target: Path,
    timeout: float,
) -> Tuple[Path, int, float]:
    expected_size = int(artifact["size_bytes"])
    expected_sha256 = str(artifact["sha256"])
    relative_path = str(artifact["path"])
    url = artifact_base_url.rstrip("/") + "/" + quote(relative_path, safe="/")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + target.name + ".",
        suffix=".staged",
        dir=str(target.parent),
    )
    started = time.perf_counter()
    downloaded = 0
    digest = hashlib.sha256()
    try:
        request = Request(url, headers={"Accept": "application/octet-stream"})
        with os.fdopen(descriptor, "wb") as file_obj, urlopen(request, timeout=timeout) as response:
            header_size = response.headers.get("Content-Length")
            if header_size is not None and int(header_size) != expected_size:
                raise ValueError("Artifact Content-Length does not match manifest")
            header_hash = response.headers.get("X-Artifact-SHA256")
            if header_hash is not None and header_hash != expected_sha256:
                raise ValueError("Artifact response hash does not match manifest")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                downloaded += len(block)
                if downloaded > expected_size:
                    raise ValueError("Artifact exceeds manifest size")
                digest.update(block)
                file_obj.write(block)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        if downloaded != expected_size:
            raise ValueError("Downloaded artifact size mismatch")
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Downloaded artifact sha256 mismatch")
        return Path(temporary_name), downloaded, (time.perf_counter() - started) * 1000.0
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def health_check(url: str, timeout: float) -> Dict[str, Any]:
    if not url:
        return {"checked": False, "healthy": True, "reason": "disabled"}
    started = time.perf_counter()
    try:
        payload, _ = fetch_json(url, timeout)
        healthy = str(payload.get("status", "")).lower() == "ok"
        return {
            "checked": True,
            "healthy": healthy,
            "reason": "ok" if healthy else "unexpected_health_payload",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 6),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "checked": True,
            "healthy": False,
            "reason": "{}: {}".format(type(exc).__name__, exc),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 6),
        }


def restore_artifacts(promoted: List[Tuple[Path, Optional[Path]]]) -> None:
    for target, backup in reversed(promoted):
        if target.exists():
            target.unlink()
        if backup is not None and backup.exists():
            os.replace(str(backup), str(target))


def sync_policy(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    current_path = safe_artifact_path(project_root, args.current)
    store = PolicyStore(current_path)
    current = store.current()
    current_version = str(current["policy_version"]) if current else None
    continuity_before = current is not None
    artifact_base_url = args.artifact_base_url or default_artifact_base(args.policy_url)

    def rollback_applied_policy() -> Dict[str, Any]:
        if current is not None:
            return store.rollback()
        if current_path.exists():
            current_path.unlink()
        return {
            "rolled_back": True,
            "reason": "failed_initial_install_removed",
            "previous_version": None,
            "current_version": None,
        }

    result: Dict[str, Any] = {
        "task": "remote_edge_policy_and_artifact_sync",
        "policy_url": args.policy_url,
        "artifact_base_url": artifact_base_url,
        "target": args.target,
        "before_version": current_version,
        "full_artifact_verify": bool(args.full_artifact_verify),
        "artifacts": [],
        "downloaded_bytes": 0,
    }
    try:
        fetch_started = time.perf_counter()
        candidate, policy_bytes = fetch_json(args.policy_url, args.timeout)
        result["policy_fetch_ms"] = round((time.perf_counter() - fetch_started) * 1000.0, 6)
        result["policy_bytes"] = policy_bytes
        valid, reason = verify_bundle(candidate)
        if not valid:
            raise ValueError("Invalid cloud policy: {}".format(reason))
        candidate_version = str(candidate["policy_version"])
        result["candidate_version"] = candidate_version
        if current and version_key(candidate_version) < version_key(current_version):
            raise ValueError("stale_policy_version")
        if (
            current
            and version_key(candidate_version) == version_key(current_version)
            and candidate != current
        ):
            raise ValueError("equal_version_content_mismatch")

        current_artifacts = {
            str(item["path"]): item
            for item in (current or {}).get("payload", {}).get("artifacts", [])
            if isinstance(item, dict) and "path" in item
        }
        staged: List[Tuple[Path, Path, Dict[str, Any]]] = []
        for artifact in candidate.get("payload", {}).get("artifacts", []):
            artifact_target = str(artifact.get("target", "edge"))
            if artifact_target not in {args.target, "all"}:
                continue
            target = safe_artifact_path(project_root, str(artifact["path"]))
            matched, verification = unchanged_artifact_is_trusted(
                target,
                artifact,
                current_artifacts,
                bool(args.full_artifact_verify),
            )
            record = {
                "role": artifact["role"],
                "path": artifact["path"],
                "size_bytes": int(artifact["size_bytes"]),
                "sha256": artifact["sha256"],
                "verification": verification,
            }
            if matched:
                record["status"] = "verified_existing"
            else:
                staged_path, downloaded, download_ms = stage_download(
                    artifact_base_url,
                    artifact,
                    target,
                    args.timeout,
                )
                staged.append((staged_path, target, artifact))
                record.update(
                    {
                        "status": "staged_download",
                        "downloaded_bytes": downloaded,
                        "download_ms": round(download_ms, 6),
                    }
                )
                result["downloaded_bytes"] += downloaded
            result["artifacts"].append(record)

        promoted: List[Tuple[Path, Optional[Path]]] = []
        policy_apply: Dict[str, Any] = {
            "applied": False,
            "reason": "not_started",
            "current_version": current_version,
        }
        policy_rolled_back = False
        try:
            for staged_path, target, _ in staged:
                backup = None
                if target.exists():
                    backup = target.with_name("." + target.name + ".rollback")
                    if backup.exists():
                        backup.unlink()
                    os.replace(str(target), str(backup))
                promoted.append((target, backup))
                os.replace(str(staged_path), str(target))

            policy_apply = {
                "applied": False,
                "reason": "already_current",
                "current_version": current_version,
            }
            if not current or version_key(candidate_version) > version_key(current_version):
                policy_apply = store.apply(candidate)
                if not policy_apply["applied"]:
                    raise RuntimeError("Policy apply failed: {}".format(policy_apply["reason"]))
            result["policy_apply"] = policy_apply
            health = health_check(args.health_url, args.timeout)
            result["health_check"] = health
            if not health["healthy"]:
                if policy_apply["applied"]:
                    result["policy_rollback"] = rollback_applied_policy()
                    policy_rolled_back = bool(result["policy_rollback"]["rolled_back"])
                restore_artifacts(promoted)
                promoted = []
                raise RuntimeError("post_update_health_check_failed")
            for _, backup in promoted:
                if backup is not None and backup.exists():
                    backup.unlink()
        except Exception:
            if policy_apply.get("applied") and not policy_rolled_back:
                result["policy_rollback"] = rollback_applied_policy()
            restore_artifacts(promoted)
            for staged_path, _, _ in staged:
                if staged_path.exists():
                    staged_path.unlink()
            raise

        after = store.current()
        result.update(
            {
                "success": True,
                "reason": "synchronized" if policy_apply["applied"] else "already_current",
                "after_version": str(after["policy_version"]) if after else None,
                "artifact_count": len(result["artifacts"]),
                "downloaded_artifact_count": len(staged),
                "business_continuity_preserved": True,
            }
        )
    except Exception as exc:  # noqa: BLE001
        preserved_policy = None
        try:
            preserved_policy = store.current()
        except Exception:  # noqa: BLE001
            preserved_policy = current if continuity_before else None
        preserved = preserved_policy is not None
        result.update(
            {
                "success": False,
                "reason": "{}: {}".format(type(exc).__name__, exc),
                "after_version": (
                    str(preserved_policy["policy_version"])
                    if preserved_policy is not None
                    else None
                ),
                "business_continuity_preserved": preserved,
            }
        )
    result["total_ms"] = round((time.perf_counter() - started) * 1000.0, 6)
    return result


def main() -> None:
    args = parse_args()
    result = sync_policy(args)
    save_json(result, Path(args.output_json))
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "success",
                    "reason",
                    "before_version",
                    "candidate_version",
                    "after_version",
                    "artifact_count",
                    "downloaded_artifact_count",
                    "downloaded_bytes",
                    "business_continuity_preserved",
                    "total_ms",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
