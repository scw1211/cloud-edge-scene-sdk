"""用途：为 GitHub 分发版本生成可核验的源码文件清单。"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.json"
EXCLUDED_ASSETS = [
    "PEMS08 prepared inference array (downloaded by traffic asset installer)",
    "traffic Edge-Qwen GGUF (downloaded by traffic asset installer)",
    "traffic benchmark results",
    "Qwen 9B weights (installed from the official Ollama registry)",
    "Qwen base weights (manifest only)",
]


def _files():
    completed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    values = []
    for name in completed.stdout.splitlines():
        if not name or name == "MANIFEST.json":
            continue
        path = ROOT / name
        if path.is_file():
            values.append((name, path))
    return sorted(values)


def _record(name, path):
    content = path.read_bytes()
    return {
        "path": name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    manifest = {
        "sdk_version": version,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_requires": ">=3.8",
        "included_files": [
            _record(name, path) for name, path in _files()
        ],
        "excluded_assets": EXCLUDED_ASSETS,
    }
    OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "manifest_built",
                "path": str(OUTPUT),
                "files": len(manifest["included_files"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
