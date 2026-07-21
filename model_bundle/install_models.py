"""Install and verify the locked cloud teacher and edge student models."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


CATALOG_PATH = Path(__file__).with_name("catalog.json")
DOWNLOAD_BLOCK_BYTES = 8 * 1024 * 1024
PROGRESS_BLOCK_BYTES = 64 * 1024 * 1024


class ModelBundleError(RuntimeError):
    """Raised when a model cannot be installed without breaking reproducibility."""


def read_catalog() -> Dict[str, Any]:
    try:
        value = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleError("cannot read model catalog: {}".format(exc)) from exc
    if value.get("schema_version") != "cloud-edge-model-catalog/v1":
        raise ModelBundleError("unsupported model catalog schema")
    return value


def require_ollama() -> str:
    executable = shutil.which("ollama")
    if executable is None:
        raise ModelBundleError("ollama is not installed or is not on PATH")
    return executable


def run_checked(argv: List[str]) -> subprocess.CompletedProcess:
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


def installed_models(ollama: str) -> Dict[str, str]:
    output = run_checked([ollama, "list"]).stdout
    records: Dict[str, str] = {}
    for line in output.splitlines()[1:]:
        columns = line.split()
        if len(columns) >= 2:
            records[columns[0]] = columns[1]
    return records


def verify_installed(
    ollama: str,
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    model = str(record["model"])
    expected = str(record["ollama_manifest_sha256"])
    observed = installed_models(ollama).get(model)
    if observed is None:
        raise ModelBundleError("required Ollama model is not installed: {}".format(model))
    if not expected.startswith(observed):
        raise ModelBundleError(
            "{} manifest mismatch: expected {}, observed {}".format(
                model, expected, observed
            )
        )
    return {
        "model": model,
        "manifest_sha256": expected,
        "status": "verified",
    }


def file_identity(path: Path) -> Tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(DOWNLOAD_BLOCK_BYTES), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def verify_edge_asset(path: Path, record: Mapping[str, Any]) -> Dict[str, Any]:
    if not path.is_file():
        raise ModelBundleError("edge model asset does not exist: {}".format(path))
    observed_bytes, observed_sha256 = file_identity(path)
    expected_bytes = int(record["asset_bytes"])
    expected_sha256 = str(record["asset_sha256"])
    if observed_bytes != expected_bytes:
        raise ModelBundleError(
            "edge asset size mismatch: expected {}, observed {}".format(
                expected_bytes, observed_bytes
            )
        )
    if observed_sha256 != expected_sha256:
        raise ModelBundleError(
            "edge asset SHA-256 mismatch: expected {}, observed {}".format(
                expected_sha256, observed_sha256
            )
        )
    return {
        "path": str(path.resolve()),
        "bytes": observed_bytes,
        "sha256": observed_sha256,
        "status": "verified",
    }


def download_edge_asset(
    record: Mapping[str, Any],
    cache_dir: Path,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / str(record["asset_name"])
    if target.is_file():
        verify_edge_asset(target, record)
        return target

    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        str(record["asset_url"]),
        headers={"User-Agent": "cloud-edge-scene-sdk/0.9.0"},
    )
    digest = hashlib.sha256()
    downloaded = 0
    next_progress = PROGRESS_BLOCK_BYTES
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with partial.open("wb") as file_obj:
                while True:
                    block = response.read(DOWNLOAD_BLOCK_BYTES)
                    if not block:
                        break
                    file_obj.write(block)
                    digest.update(block)
                    downloaded += len(block)
                    if downloaded >= next_progress:
                        print("downloaded {:.1f} MiB".format(downloaded / 1024 / 1024))
                        next_progress += PROGRESS_BLOCK_BYTES
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    if downloaded != int(record["asset_bytes"]):
        partial.unlink(missing_ok=True)
        raise ModelBundleError("downloaded edge asset has the wrong size")
    if digest.hexdigest() != str(record["asset_sha256"]):
        partial.unlink(missing_ok=True)
        raise ModelBundleError("downloaded edge asset has the wrong SHA-256")
    partial.replace(target)
    return target


def install_cloud(ollama: str, record: Mapping[str, Any]) -> Dict[str, Any]:
    run_checked([ollama, "pull", str(record["model"])])
    verified = verify_installed(ollama, record)
    return verified


def install_edge(
    ollama: str,
    record: Mapping[str, Any],
    edge_file: Optional[Path],
    cache_dir: Path,
) -> Dict[str, Any]:
    asset = (
        Path(edge_file).expanduser().resolve()
        if edge_file is not None
        else download_edge_asset(record, cache_dir)
    )
    identity = verify_edge_asset(asset, record)
    if " " in str(asset):
        raise ModelBundleError("edge model asset path must not contain spaces")

    modelfile = """FROM {asset}
TEMPLATE {{{{ .Prompt }}}}
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER num_ctx 1024
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
PARAMETER temperature 0
PARAMETER top_p 1
""".format(asset=asset)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".Modelfile",
            delete=False,
        ) as file_obj:
            file_obj.write(modelfile)
            temporary_path = Path(file_obj.name)
        run_checked(
            [
                ollama,
                "create",
                str(record["model"]),
                "-f",
                str(temporary_path),
            ]
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    verified = verify_installed(ollama, record)
    verified["asset"] = identity
    return verified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the locked cloud teacher and edge student models."
    )
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--edge-file", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "cloud-edge-scene-sdk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cloud = bool(args.cloud or args.all)
    use_edge = bool(args.edge or args.all)
    if not use_cloud and not use_edge:
        raise SystemExit("select --cloud, --edge, or --all")
    if args.edge_file is not None and not use_edge:
        raise SystemExit("--edge-file requires --edge or --all")

    catalog = read_catalog()
    ollama = require_ollama()
    results: List[Dict[str, Any]] = []
    if args.verify_only:
        if use_cloud:
            results.append(verify_installed(ollama, catalog["cloud_teacher"]))
        if use_edge:
            results.append(verify_installed(ollama, catalog["edge_general_student"]))
    else:
        if use_cloud:
            results.append(install_cloud(ollama, catalog["cloud_teacher"]))
        if use_edge:
            results.append(
                install_edge(
                    ollama,
                    catalog["edge_general_student"],
                    args.edge_file,
                    args.cache_dir.expanduser().resolve(),
                )
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
