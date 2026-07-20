"""用途：调用指定 llama.cpp 工具把合并文本模型转换并量化为边缘 GGUF。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from edge_llm_factory.contracts import ManifestError, sha256_file, write_json_object


QUANTIZATIONS = {"Q3_K_S", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}


def _run(command: List[str], stage: str) -> Dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        tail = (completed.stdout + "\n" + completed.stderr)[-8000:]
        raise RuntimeError("{} 失败，退出码 {}:\n{}".format(stage, completed.returncode, tail))
    return {
        "command": command,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError("无法读取 {}: {}".format(label, exc)) from exc
    if not isinstance(value, dict):
        raise ManifestError("{} 必须是 JSON object".format(label))
    return value


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="转换并量化边缘文本模型 GGUF。")
    parser.add_argument("--converter", required=True)
    parser.add_argument("--quantizer", required=True)
    parser.add_argument("--merged_model", required=True)
    parser.add_argument("--f16_gguf", required=True)
    parser.add_argument("--output_gguf", required=True)
    parser.add_argument("--quantization", choices=sorted(QUANTIZATIONS), required=True)
    parser.add_argument("--imatrix", default="")
    parser.add_argument("--imatrix_summary", default="")
    parser.add_argument("--calibration_manifest", default="")
    parser.add_argument("--keep_f16", action="store_true")
    parser.add_argument("--with_mtp", action="store_true")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args(argv)

    converter = Path(args.converter).resolve()
    quantizer = Path(args.quantizer).resolve()
    merged = Path(args.merged_model).resolve()
    f16 = Path(args.f16_gguf).resolve()
    output = Path(args.output_gguf).resolve()
    imatrix = Path(args.imatrix).resolve() if args.imatrix else None
    imatrix_summary_path = Path(args.imatrix_summary).resolve() if args.imatrix_summary else None
    calibration_manifest = (
        Path(args.calibration_manifest).resolve() if args.calibration_manifest else None
    )
    if not converter.is_file() or not quantizer.is_file() or not merged.is_dir():
        raise ManifestError("converter、quantizer 或 merged_model 不存在")
    if f16.exists() or output.exists():
        raise ManifestError("拒绝覆盖已有 GGUF 产物")

    calibrated_inputs = (imatrix, imatrix_summary_path, calibration_manifest)
    if any(calibrated_inputs) and not all(calibrated_inputs):
        raise ManifestError(
            "--imatrix、--imatrix_summary 与 --calibration_manifest 必须同时提供"
        )
    calibration = None
    imatrix_summary = None
    if imatrix and imatrix_summary_path and calibration_manifest:
        if not all(path.is_file() for path in calibrated_inputs):
            raise ManifestError("imatrix、imatrix summary 或 calibration manifest 不存在")
        calibration = _read_json_object(calibration_manifest, "calibration manifest")
        imatrix_summary = _read_json_object(imatrix_summary_path, "imatrix summary")
        if calibration.get("schema_version") != "edge-llm-general-calibration/v1":
            raise ManifestError("calibration manifest schema_version 不受支持")
        if calibration.get("scene_specific_samples") != 0:
            raise ManifestError("通用基座量化禁止使用场景专用校准样本")
        if imatrix_summary.get("task") != "scene_independent_llm_importance_matrix":
            raise ManifestError("imatrix summary task 不受支持")
        if imatrix_summary.get("artifact", {}).get("sha256") != sha256_file(imatrix):
            raise ManifestError("imatrix 哈希与 summary 不一致")
        if imatrix_summary.get("calibration_manifest", {}).get("sha256") != sha256_file(
            calibration_manifest
        ):
            raise ManifestError("imatrix 使用的 calibration manifest 与当前文件不一致")

    f16.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    convert_command = [
        sys.executable,
        str(converter),
        str(merged),
        "--outfile",
        str(f16),
        "--outtype",
        "f16",
    ]
    if not args.with_mtp:
        convert_command.append("--no-mtp")
    convert_report = _run(convert_command, "HF -> GGUF")
    if not f16.is_file() or f16.stat().st_size == 0:
        raise RuntimeError("转换器未生成 F16 GGUF")
    if imatrix_summary and imatrix_summary.get("model", {}).get("sha256") != sha256_file(f16):
        raise ManifestError("imatrix 对应的 F16 模型与本次待量化模型不一致")

    quantize_command = [str(quantizer)]
    if imatrix:
        quantize_command.extend(["--imatrix", str(imatrix)])
    quantize_command.extend([str(f16), str(output), args.quantization])
    quantize_report = _run(quantize_command, "GGUF quantization")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("量化器未生成目标 GGUF")
    summary = {
        "task": "edge_llm_gguf_export",
        "quantization": args.quantization,
        "calibrated": bool(imatrix),
        "mtp_included": bool(args.with_mtp),
        "convert": convert_report,
        "quantize": quantize_report,
        "artifact": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }
    if imatrix and imatrix_summary_path and calibration_manifest:
        summary["calibration"] = {
            "imatrix": {
                "path": str(imatrix),
                "bytes": imatrix.stat().st_size,
                "sha256": sha256_file(imatrix),
            },
            "imatrix_summary": {
                "path": str(imatrix_summary_path),
                "sha256": sha256_file(imatrix_summary_path),
            },
            "manifest": {
                "path": str(calibration_manifest),
                "sha256": sha256_file(calibration_manifest),
            },
            "token_count": calibration.get("token_count") if calibration else None,
            "scene_specific_samples": 0,
        }
    if args.keep_f16:
        summary["f16_artifact"] = {
            "path": str(f16),
            "bytes": f16.stat().st_size,
            "sha256": sha256_file(f16),
        }
    else:
        f16.unlink()
    write_json_object(Path(args.summary).resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
