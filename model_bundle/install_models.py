"""Install or verify the frozen cloud and two role-specific edge models."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


CATALOG_PATH = Path(__file__).with_name("catalog.json")
REPOSITORY_ROOT = CATALOG_PATH.parent.parent
BLOCK_BYTES = 8 * 1024 * 1024


class ModelBundleError(RuntimeError):
    """Raised when a frozen model cannot be installed or verified exactly."""


def read_catalog() -> Dict[str, Any]:
    try:
        value = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleError("cannot read model catalog: {}".format(exc)) from exc
    if value.get("schema_version") != "cloud-edge-model-catalog/v2":
        raise ModelBundleError("unsupported model catalog schema")
    for field in (
        "cloud_production_runtime",
        "cloud_reference",
        "edge_general_model",
        "edge_business_model",
        "edge_deployment_boundary",
    ):
        if not isinstance(value.get(field), dict):
            raise ModelBundleError("model catalog is missing {}".format(field))
    return value


def _run(argv: List[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            argv,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise ModelBundleError(
            "command failed ({}):\n{}".format(" ".join(argv), exc.stdout or "")
        ) from exc


def _ollama() -> str:
    executable = shutil.which("ollama")
    if executable is None:
        raise ModelBundleError("ollama is not installed or is not on PATH")
    return executable


def _installed_ollama_models(executable: str) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    for line in _run([executable, "list"]).stdout.splitlines()[1:]:
        columns = line.split()
        if len(columns) >= 2:
            rows[columns[0]] = columns[1]
    return rows


def verify_cloud(executable: str, record: Mapping[str, Any]) -> Dict[str, Any]:
    model = str(record["model"])
    expected = str(record["ollama_manifest_sha256"])
    observed = _installed_ollama_models(executable).get(model)
    if observed is None:
        raise ModelBundleError(
            "required Ollama model is not installed: {}".format(model)
        )
    if not expected.startswith(observed):
        raise ModelBundleError(
            "{} manifest mismatch: expected {}, observed {}".format(
                model, expected, observed
            )
        )
    return {
        "role": str(record["role"]),
        "model": model,
        "manifest_sha256": expected,
        "business_semantics": str(record["business_semantics"]),
        "status": "verified",
    }


def install_cloud(executable: str, record: Mapping[str, Any]) -> Dict[str, Any]:
    _run([executable, "pull", str(record["model"])])
    return verify_cloud(executable, record)


def file_identity(path: Path) -> Tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(BLOCK_BYTES), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def verify_edge(
    record: Mapping[str, Any], override: Optional[Path]
) -> Dict[str, Any]:
    path = (
        override.expanduser().resolve()
        if override is not None
        else (REPOSITORY_ROOT / str(record["artifact_path"])).resolve()
    )
    if not path.is_file() or path.is_symlink():
        raise ModelBundleError(
            "edge model asset is missing or is a symlink: {}".format(path)
        )
    observed_bytes, observed_sha256 = file_identity(path)
    expected_bytes = int(record["artifact_bytes"])
    expected_sha256 = str(record["artifact_sha256"])
    if observed_bytes != expected_bytes:
        raise ModelBundleError(
            "edge model byte count mismatch: expected {}, observed {}".format(
                expected_bytes, observed_bytes
            )
        )
    if observed_sha256 != expected_sha256:
        raise ModelBundleError(
            "edge model SHA-256 mismatch: expected {}, observed {}".format(
                expected_sha256, observed_sha256
            )
        )
    return {
        "role": str(record["role"]),
        "model": str(record["model"]),
        "runtime": str(record["runtime"]),
        "quantization": str(record["quantization"]),
        "path": str(path),
        "bytes": observed_bytes,
        "sha256": observed_sha256,
        "status": "verified",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install or verify the frozen final cloud and edge models."
    )
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--edge-general", action="store_true")
    parser.add_argument("--edge-business", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--edge-general-file", type=Path)
    parser.add_argument("--edge-business-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cloud = bool(args.cloud or args.all)
    use_edge_general = bool(args.edge or args.edge_general or args.all)
    use_edge_business = bool(args.edge or args.edge_business or args.all)
    if not use_cloud and not use_edge_general and not use_edge_business:
        raise SystemExit(
            "select --cloud, --edge, --edge-general, --edge-business, or --all"
        )
    if args.edge_general_file is not None and not use_edge_general:
        raise SystemExit("--edge-general-file requires --edge-general, --edge, or --all")
    if args.edge_business_file is not None and not use_edge_business:
        raise SystemExit("--edge-business-file requires --edge-business, --edge, or --all")

    catalog = read_catalog()
    results: List[Dict[str, Any]] = []
    if use_cloud:
        executable = _ollama()
        cloud = catalog["cloud_production_runtime"]
        results.append(
            verify_cloud(executable, cloud)
            if args.verify_only
            else install_cloud(executable, cloud)
        )
    if use_edge_general:
        results.append(
            verify_edge(catalog["edge_general_model"], args.edge_general_file)
        )
    if use_edge_business:
        results.append(
            verify_edge(catalog["edge_business_model"], args.edge_business_file)
        )

    print(
        json.dumps(
            {
                "bundle_version": catalog["bundle_version"],
                "models": results,
                "status": "model_bundle_ready",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
