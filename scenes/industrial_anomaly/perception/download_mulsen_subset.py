#!/usr/bin/env python3
"""Range-download only one RGB/infrared product from MulSen_AD_new.zip."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Sequence

from remotezip import RemoteZip


DEFAULT_URL = (
    "https://huggingface.co/datasets/orgjy314159/MulSen_AD/resolve/main/"
    "MulSen_AD_new.zip?download=true"
)
ARCHIVE_BYTES = 8_973_641_666
ARCHIVE_LFS_SHA256 = "6e7cfec9211e4cbab85118ae2a65dacef2794ab5bb3951ecdf8b0f2ec82a4b7d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(product: str, output: Path, url: str = DEFAULT_URL) -> Dict[str, Any]:
    prefix = "MulSen_AD/{}/".format(product)
    selected = []
    with RemoteZip(url) as archive:
        for member in archive.infolist():
            pure = PurePosixPath(member.filename)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError("unsafe archive member: {}".format(member.filename))
            if not member.is_dir() and member.filename.startswith(prefix) and any(
                token in member.filename for token in ("/RGB/", "/Infrared/")
            ):
                selected.append(member)
        if not selected:
            raise ValueError("product not found in archive: {}".format(product))
        output.mkdir(parents=True, exist_ok=True)
        archive.extractall(output, selected)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "mulsen-ad-subset/v1",
        "source": {
            "url": url,
            "archive": "MulSen_AD_new.zip",
            "archive_bytes": ARCHIVE_BYTES,
            "archive_lfs_sha256": ARCHIVE_LFS_SHA256,
            "selection": "third archive in Hugging Face repository tree",
        },
        "product": product,
        "modalities": ["RGB", "Infrared"],
        "files": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "pointcloud_downloaded": False,
    }
    manifest_path = output / "SUBSET_MANIFEST.json"
    temporary = manifest_path.with_name(manifest_path.name + ".part")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", default="capsule")
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args(argv)
    result = download(args.product, Path(args.output), args.url)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
