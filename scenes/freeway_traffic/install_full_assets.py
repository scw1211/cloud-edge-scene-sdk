"""用途：下载并逐字节校验真实交通推理数组和 Edge-Qwen GGUF。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "asset_catalog.json"
BLOCK_BYTES = 8 * 1024 * 1024


def _identity(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(BLOCK_BYTES), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _verify(path, record):
    if not path.is_file():
        raise FileNotFoundError(path)
    size, digest = _identity(path)
    if size != int(record["bytes"]):
        raise ValueError(
            "{} 字节数不匹配：{} != {}".format(
                path, size, record["bytes"]
            )
        )
    if digest != str(record["sha256"]):
        raise ValueError("{} SHA-256 不匹配".format(path))
    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "status": "verified",
    }


def _download(record):
    target = ROOT / str(record["file"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return _verify(target, record)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        str(record["url"]),
        headers={"User-Agent": "cloud-edge-scene-sdk/0.13.0"},
    )
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with partial.open("wb") as file_obj:
                while True:
                    block = response.read(BLOCK_BYTES)
                    if not block:
                        break
                    file_obj.write(block)
                    digest.update(block)
                    downloaded += len(block)
                    print(
                        "{}: {:.1f} MiB".format(
                            target.name, downloaded / 1024 / 1024
                        ),
                        flush=True,
                    )
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    if downloaded != int(record["bytes"]):
        partial.unlink()
        raise ValueError("{} 下载字节数不匹配".format(target.name))
    if digest.hexdigest() != str(record["sha256"]):
        partial.unlink()
        raise ValueError("{} 下载 SHA-256 不匹配".format(target.name))
    partial.replace(target)
    return _verify(target, record)


def _embedded(catalog):
    return [
        _verify(ROOT / record["file"], record)
        for record in catalog["embedded_assets"].values()
    ]


def _activate_edge_release(gguf_path):
    from edge_llm_factory.release_store import ReleaseStore

    registry = ROOT / "runtime" / "edge_llm_release_store.json"
    result = ReleaseStore(registry).promote(
        "freeway-traffic-current-state-v2.0.2-q6k",
        ROOT / "assets" / "edge_llm" / "base_manifest.json",
        ROOT / "assets" / "edge_llm" / "adapter_package_current_state_v2",
        gguf_path,
    )
    return {
        "registry": str(registry),
        "status": result["status"],
        "active_release_id": result["active_release_id"],
        "revision": result["revision"],
    }


def main():
    parser = argparse.ArgumentParser(description="安装真实交通全链路资产")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    use_edge = bool(args.edge or args.all)
    use_cloud = bool(args.cloud or args.all)
    if not use_edge and not use_cloud:
        raise SystemExit("请选择 --edge、--cloud 或 --all")

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    result = {
        "bundle_version": catalog["bundle_version"],
        "embedded": _embedded(catalog),
        "downloaded": [],
        "edge_release": None,
        "cloud_qwen": None,
    }
    if use_edge:
        for record in catalog["downloaded_assets"].values():
            path = ROOT / record["file"]
            result["downloaded"].append(
                _verify(path, record)
                if args.verify_only
                else _download(record)
            )
        gguf = ROOT / catalog["downloaded_assets"]["edge_qwen_gguf"]["file"]
        result["edge_release"] = _activate_edge_release(gguf)
    if use_cloud:
        command = [
            sys.executable,
            "-m",
            "model_bundle.install_models",
            "--cloud",
        ]
        if args.verify_only:
            command.append("--verify-only")
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        result["cloud_qwen"] = json.loads(completed.stdout)
    result["status"] = "traffic_full_assets_ready"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
