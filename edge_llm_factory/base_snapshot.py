"""用途：在训练前校验本地基座快照、权重分片和动作 token 与清单完全一致。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from edge_llm_factory.contracts import (
    ManifestError,
    read_json_object,
    sha256_file,
    validate_base_manifest,
)


def verify_base_snapshot(
    base: Mapping[str, Any],
    snapshot_dir: Path,
    verify_tokenizer: bool = True,
) -> Dict[str, Any]:
    manifest = validate_base_manifest(base)
    snapshot = snapshot_dir.resolve()
    if not snapshot.is_dir():
        raise ManifestError("基座快照目录不存在: {}".format(snapshot))
    revision = manifest["source"]["revision"]
    if snapshot.name != revision:
        raise ManifestError("快照目录名与锁定 revision 不一致")

    fixed = {
        "config.json": manifest["source"]["config_sha256"],
        "tokenizer.json": manifest["source"]["tokenizer_sha256"],
        "tokenizer_config.json": manifest["source"]["tokenizer_config_sha256"],
    }
    checked = []
    for relative_name, expected_hash in fixed.items():
        path = snapshot / relative_name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ManifestError("基座文件哈希不匹配: {}".format(relative_name))
        checked.append(relative_name)
    for artifact in manifest["source"]["artifacts"]:
        path = snapshot / artifact["path"]
        if not path.is_file():
            raise ManifestError("基座缺少权重产物: {}".format(artifact["path"]))
        if path.stat().st_size != artifact["bytes"]:
            raise ManifestError("基座权重大小不匹配: {}".format(artifact["path"]))
        if sha256_file(path) != artifact["sha256"]:
            raise ManifestError("基座权重哈希不匹配: {}".format(artifact["path"]))
        checked.append(artifact["path"])

    slot_report = None
    if verify_tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ManifestError("验证动作 token 需要 transformers") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False
        )
        slot_report = {}
        for slot in manifest["decision_protocol"]["slots"]:
            encoded = tokenizer.encode(slot["token"], add_special_tokens=False)
            if encoded != [slot["token_id"]]:
                raise ManifestError("动作槽 {} 的 tokenizer ID 不匹配".format(slot["slot"]))
            slot_report[slot["slot"]] = encoded[0]
    return {
        "status": "valid",
        "base_id": manifest["base_id"],
        "revision": revision,
        "snapshot": str(snapshot),
        "checked_files": checked,
        "action_token_ids": slot_report,
    }


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="校验本地边缘大模型基座快照。")
    parser.add_argument("--base", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--skip_tokenizer", action="store_true")
    args = parser.parse_args(argv)
    result = verify_base_snapshot(
        read_json_object(Path(args.base)),
        Path(args.snapshot),
        verify_tokenizer=not args.skip_tokenizer,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
