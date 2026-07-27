"""用途：以内容寻址方式保存边缘上传的热力图、特征和原始证据。"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Dict, Optional


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ArtifactRecord:
    sha256: str
    size_bytes: int
    content_type: str
    evidence_id: str
    created_at_ms: int
    path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceArtifactStore:
    """Stores immutable evidence blobs under their verified SHA256 digest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def validate_sha256(value: str) -> str:
        digest = str(value).strip().lower()
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("artifact sha256 must contain 64 lowercase hexadecimal characters")
        return digest

    def _blob_path(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def _metadata_path(self, digest: str) -> Path:
        return self.root / digest[:2] / (digest + ".json")

    def put(
        self,
        data: bytes,
        expected_sha256: str,
        content_type: str = "application/octet-stream",
        evidence_id: str = "",
    ) -> Dict[str, Any]:
        if not isinstance(data, bytes):
            raise ValueError("artifact body must be bytes")
        digest = self.validate_sha256(expected_sha256)
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise ValueError(
                "artifact sha256 mismatch: expected {}, received {}".format(
                    digest, actual
                )
            )
        media_type = str(content_type).strip() or "application/octet-stream"
        identifier = str(evidence_id).strip()
        blob_path = self._blob_path(digest)
        metadata_path = self._metadata_path(digest)
        with self._lock:
            if blob_path.is_file():
                if blob_path.stat().st_size != len(data):
                    raise ValueError("existing artifact size does not match uploaded content")
                return {"created": False, "artifact": self.describe(digest)}

            blob_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=digest + ".", dir=str(blob_path.parent)
            )
            try:
                with os.fdopen(descriptor, "wb") as file_obj:
                    file_obj.write(data)
                    file_obj.flush()
                    os.fsync(file_obj.fileno())
                os.replace(temporary_name, blob_path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)

            record = ArtifactRecord(
                sha256=digest,
                size_bytes=len(data),
                content_type=media_type,
                evidence_id=identifier,
                created_at_ms=int(time.time() * 1000),
                path=str(blob_path),
            )
            metadata_path.write_text(
                json.dumps(
                    record.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            return {"created": True, "artifact": record.to_dict()}

    def describe(self, sha256: str) -> Dict[str, Any]:
        digest = self.validate_sha256(sha256)
        blob_path = self._blob_path(digest)
        metadata_path = self._metadata_path(digest)
        if not blob_path.is_file():
            raise KeyError("artifact not found: {}".format(digest))
        if metadata_path.is_file():
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return dict(value)
        return ArtifactRecord(
            sha256=digest,
            size_bytes=blob_path.stat().st_size,
            content_type="application/octet-stream",
            evidence_id="",
            created_at_ms=int(blob_path.stat().st_mtime * 1000),
            path=str(blob_path),
        ).to_dict()

    def path_for(self, sha256: str) -> Path:
        digest = self.validate_sha256(sha256)
        path = self._blob_path(digest)
        if not path.is_file():
            raise KeyError("artifact not found: {}".format(digest))
        return path

    def snapshot(self) -> Dict[str, Any]:
        count = 0
        total_bytes = 0
        for path in self.root.glob("*/*"):
            if path.is_file() and _SHA256_RE.fullmatch(path.name):
                count += 1
                total_bytes += path.stat().st_size
        return {
            "root": str(self.root),
            "artifact_count": count,
            "stored_bytes": total_bytes,
        }


def optional_artifact_path(uri: Optional[str]) -> Optional[Path]:
    """Resolve only explicit local file evidence; other URI schemes stay references."""
    if uri is None:
        return None
    value = str(uri).strip()
    if not value.startswith("file://"):
        return None
    from urllib.parse import unquote, urlsplit

    parsed = urlsplit(value)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError("file evidence URI must reference the local host")
    path = Path(unquote(parsed.path)).resolve()
    if not path.is_file():
        raise FileNotFoundError("evidence file not found: {}".format(path))
    return path
