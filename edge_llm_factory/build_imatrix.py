"""用途：调用 llama-imatrix 为通用文本基座生成可追溯的重要性矩阵。"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from edge_llm_factory.contracts import ManifestError, sha256_file, write_json_object


def imatrix_command(
    binary: Path,
    model: Path,
    calibration_text: Path,
    output: Path,
    threads: int,
    ctx_size: int,
    batch_size: int,
    ubatch_size: int,
    gpu_layers: str,
    chunks: int,
) -> List[str]:
    command = [
        str(binary),
        "-m",
        str(model),
        "-f",
        str(calibration_text),
        "-o",
        str(output),
        "--output-format",
        "gguf",
        "--parse-special",
        "--no-ppl",
        "-t",
        str(threads),
        "-c",
        str(ctx_size),
        "-b",
        str(batch_size),
        "-ub",
        str(ubatch_size),
        "-ngl",
        gpu_layers,
    ]
    if chunks > 0:
        command.extend(["--chunks", str(chunks)])
    return command


def _read_manifest(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError("无法读取校准清单 {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise ManifestError("校准清单必须是 JSON object")
    return value


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="生成通用基座 GGUF importance matrix。")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration_text", required=True)
    parser.add_argument("--calibration_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--ctx_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--ubatch_size", type=int, default=512)
    parser.add_argument("--gpu_layers", default="all")
    parser.add_argument("--chunks", type=int, default=-1)
    args = parser.parse_args(argv)

    binary = Path(args.binary).resolve()
    model = Path(args.model).resolve()
    calibration_text = Path(args.calibration_text).resolve()
    calibration_manifest = Path(args.calibration_manifest).resolve()
    output = Path(args.output).resolve()
    summary_path = Path(args.summary).resolve()
    for path in (binary, model, calibration_text, calibration_manifest):
        if not path.is_file():
            raise ManifestError("imatrix 输入不存在: {}".format(path))
    if output.exists() or summary_path.exists():
        raise ManifestError("拒绝覆盖已有 imatrix 或总结文件")
    if args.threads <= 0 or args.ctx_size <= 0 or args.batch_size <= 0 or args.ubatch_size <= 0:
        raise ManifestError("线程、上下文和批大小必须大于 0")

    calibration = _read_manifest(calibration_manifest)
    if calibration.get("schema_version") != "edge-llm-general-calibration/v1":
        raise ManifestError("校准清单 schema_version 不受支持")
    expected_hash = calibration.get("calibration_text", {}).get("sha256")
    actual_hash = sha256_file(calibration_text)
    if expected_hash != actual_hash:
        raise ManifestError("校准文本哈希与清单不一致")
    if calibration.get("scene_specific_samples") != 0:
        raise ManifestError("通用 imatrix 禁止包含场景专用样本")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = imatrix_command(
        binary=binary,
        model=model,
        calibration_text=calibration_text,
        output=output,
        threads=args.threads,
        ctx_size=args.ctx_size,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        gpu_layers=args.gpu_layers,
        chunks=args.chunks,
    )
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    wall_time_s = time.perf_counter() - started
    if completed.returncode != 0:
        tail = (completed.stdout + "\n" + completed.stderr)[-12000:]
        raise RuntimeError("llama-imatrix 失败，退出码 {}:\n{}".format(completed.returncode, tail))
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("llama-imatrix 未生成有效输出")

    summary = {
        "task": "scene_independent_llm_importance_matrix",
        "command": command,
        "wall_time_s": round(wall_time_s, 6),
        "model": {"path": str(model), "sha256": sha256_file(model), "bytes": model.stat().st_size},
        "calibration_manifest": {
            "path": str(calibration_manifest),
            "sha256": sha256_file(calibration_manifest),
        },
        "calibration_tokens": calibration.get("token_count"),
        "artifact": {
            "path": str(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
        },
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
    }
    write_json_object(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
