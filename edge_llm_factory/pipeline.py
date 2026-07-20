"""用途：按可恢复阶段执行蒸馏、评估、量化与打包，任何失败都会中止并留证。"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from edge_llm_factory.adapter_package import build_adapter_package, validate_adapter_package
from edge_llm_factory.release_store import ReleaseStore
from edge_llm_factory.text_base import verify_text_snapshot
from edge_llm_factory.contracts import (
    PIPELINE_SCHEMA,
    ManifestError,
    canonical_sha256,
    json_path,
    read_json_object,
    sha256_file,
    write_json_object,
)


STAGE_KINDS = {
    "verify_base",
    "command",
    "select_candidate",
    "build_adapter",
    "validate_adapter",
    "publish_release",
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _artifact_digest(path: Path) -> Dict[str, Any]:
    if path.is_file():
        return {"type": "file", "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if path.is_dir():
        records = []
        total = 0
        for item in sorted(value for value in path.rglob("*") if value.is_file()):
            size = item.stat().st_size
            total += size
            records.append(
                {
                    "path": str(item.relative_to(path)),
                    "bytes": size,
                    "sha256": sha256_file(item),
                }
            )
        return {
            "type": "directory",
            "bytes": total,
            "file_count": len(records),
            "sha256": canonical_sha256(records),
        }
    raise ManifestError("阶段产物不存在: {}".format(path))


def _validate_config(value: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(value)
    if config.get("schema_version") != PIPELINE_SCHEMA:
        raise ManifestError("pipeline schema_version 必须是 {}".format(PIPELINE_SCHEMA))
    if not isinstance(config.get("pipeline_id"), str) or not config["pipeline_id"]:
        raise ManifestError("pipeline_id 必须是非空字符串")
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ManifestError("pipeline stages 必须是非空数组")
    identifiers = set()
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ManifestError("stage {} 必须是对象".format(index))
        identifier = stage.get("id")
        kind = stage.get("kind")
        if not isinstance(identifier, str) or not identifier:
            raise ManifestError("stage {} 缺少 id".format(index))
        if identifier in identifiers:
            raise ManifestError("stage id 不能重复: {}".format(identifier))
        identifiers.add(identifier)
        if kind not in STAGE_KINDS:
            raise ManifestError("不支持的 stage kind: {}".format(kind))
        if kind == "verify_base":
            for field in ("base", "snapshot_manifest", "snapshot"):
                if not isinstance(stage.get(field), str) or not stage[field]:
                    raise ManifestError("verify_base 阶段缺少 {}".format(field))
        if kind == "command":
            argv = stage.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and v for v in argv):
                raise ManifestError("command stage argv 必须是非空字符串数组")
            if any("\x00" in value for value in argv):
                raise ManifestError("command argv 含 NUL")
        if kind == "publish_release":
            for field in ("registry", "release_id", "base", "package", "deployment_artifact"):
                if not isinstance(stage.get(field), str) or not stage[field]:
                    raise ManifestError("publish_release 阶段缺少 {}".format(field))
        for field in ("inputs", "outputs"):
            values = stage.get(field, [])
            if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
                raise ManifestError("stage {} {} 必须是路径数组".format(identifier, field))
    return config


def _format_argv(argv: Sequence[str], root: Path) -> List[str]:
    values = {"python": sys.executable, "project_root": str(root)}
    formatted = []
    for part in argv:
        try:
            formatted.append(part.format_map(values))
        except KeyError as exc:
            raise ManifestError("未知命令占位符: {}".format(exc.args[0])) from exc
    return formatted


def _require_inputs(stage: Mapping[str, Any], root: Path) -> None:
    for value in stage.get("inputs", []):
        path = _resolve(root, value)
        if not path.exists():
            raise ManifestError("阶段 {} 缺少输入: {}".format(stage["id"], path))


def _collect_outputs(stage: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    return {
        value: _artifact_digest(_resolve(root, value))
        for value in stage.get("outputs", [])
    }


def _run_command(stage: Mapping[str, Any], root: Path, log_dir: Path) -> Dict[str, Any]:
    command = _format_argv(stage["argv"], root)
    environment = os.environ.copy()
    raw_environment = stage.get("environment", {})
    if not isinstance(raw_environment, dict):
        raise ManifestError("command environment 必须是对象")
    for key, value in raw_environment.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ManifestError("command environment 只允许字符串")
        environment[key] = value.format_map(
            {"python": sys.executable, "project_root": str(root)}
        )
    completed = subprocess.run(
        command,
        cwd=str(root),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "{}.log".format(stage["id"])
    log_path.write_text(
        "$ {}\n\n[stdout]\n{}\n\n[stderr]\n{}".format(
            " ".join(command), completed.stdout, completed.stderr
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "阶段 {} 失败，退出码 {}，日志 {}".format(
                stage["id"], completed.returncode, log_path
            )
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "log": str(log_path),
        "stdout_tail": completed.stdout[-2000:],
    }


def _select_candidate(stage: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    candidates = stage.get("candidates")
    metrics = stage.get("selection_metrics")
    if not isinstance(candidates, list) or not candidates:
        raise ManifestError("select_candidate.candidates 不能为空")
    if not isinstance(metrics, list) or not metrics:
        raise ManifestError("select_candidate.selection_metrics 不能为空")
    scored = []
    for raw in candidates:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            raise ManifestError("candidate 必须包含 id")
        report_path = _resolve(root, raw.get("report", ""))
        report = read_json_object(report_path)
        leakage_path = raw.get("test_set_used_for_training_path")
        if not isinstance(leakage_path, str):
            raise ManifestError("candidate 必须声明 test_set_used_for_training_path")
        if json_path(report, leakage_path) is not False:
            raise ManifestError("候选 {} 使用了验收测试集训练".format(raw["id"]))
        score = []
        observed = {}
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("mode") not in {"max", "min"}:
                raise ManifestError("selection metric 必须声明 path 和 max/min")
            value = json_path(report, metric.get("path", ""))
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ManifestError("候选指标不是数字: {}".format(metric.get("path")))
            number = float(value)
            observed[metric["path"]] = number
            score.append(number if metric["mode"] == "max" else -number)
        scored.append(
            {
                "id": raw["id"],
                "report": str(report_path),
                "artifact": raw.get("artifact"),
                "metrics": observed,
                "score": score,
            }
        )
    winner = max(scored, key=lambda row: tuple(row["score"]))
    output = _resolve(root, stage.get("output", ""))
    result = {
        "task": "edge_llm_candidate_selection",
        "selected": winner,
        "candidates": scored,
        "test_set_used_for_training": False,
    }
    write_json_object(output, result)
    return result


def _execute_stage(stage: Mapping[str, Any], root: Path, log_dir: Path) -> Dict[str, Any]:
    kind = stage["kind"]
    if kind == "verify_base":
        base = read_json_object(_resolve(root, stage["base"]))
        return verify_text_snapshot(
            base,
            read_json_object(_resolve(root, stage["snapshot_manifest"])),
            _resolve(root, stage["snapshot"]),
            True,
        )
    if kind == "command":
        return _run_command(stage, root, log_dir)
    if kind == "select_candidate":
        return _select_candidate(stage, root)
    if kind == "build_adapter":
        return build_adapter_package(
            project_root=root,
            base_manifest_path=_resolve(root, stage["base"]),
            spec_path=_resolve(root, stage["spec"]),
            output_dir=_resolve(root, stage["output"]),
            archive_path=_resolve(root, stage["archive"]) if stage.get("archive") else None,
        )
    if kind == "validate_adapter":
        return validate_adapter_package(
            _resolve(root, stage["package"]),
            _resolve(root, stage["base"]),
            require_gates=True,
        )
    if kind == "publish_release":
        return ReleaseStore(_resolve(root, stage["registry"])).promote(
            stage["release_id"],
            _resolve(root, stage["base"]),
            _resolve(root, stage["package"]),
            _resolve(root, stage["deployment_artifact"]),
        )
    raise ManifestError("未知阶段类型: {}".format(kind))


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json_object(temporary, state)
    os.replace(str(temporary), str(path))


def run_pipeline(
    config_path: Path,
    project_root: Path,
    resume: bool = False,
    dry_run: bool = False,
    from_stage: Optional[str] = None,
    to_stage: Optional[str] = None,
) -> Dict[str, Any]:
    config_file = config_path.resolve()
    config = _validate_config(read_json_object(config_file))
    root = project_root.resolve()
    state_path = _resolve(root, config.get("state_path", "runtime/edge_llm_pipeline_state.json"))
    config_hash = sha256_file(config_file)
    state: Dict[str, Any] = {
        "schema_version": 1,
        "pipeline_id": config["pipeline_id"],
        "config_sha256": config_hash,
        "stages": {},
    }
    if state_path.exists():
        previous = read_json_object(state_path)
        if not resume:
            raise ManifestError("流水线状态已存在；确认后使用 --resume: {}".format(state_path))
        if previous.get("config_sha256") != config_hash:
            raise ManifestError("配置已变化，不能复用旧流水线状态")
        state = previous
    identifiers = [stage["id"] for stage in config["stages"]]
    if from_stage is not None and from_stage not in identifiers:
        raise ManifestError("未知 from_stage: {}".format(from_stage))
    if to_stage is not None and to_stage not in identifiers:
        raise ManifestError("未知 to_stage: {}".format(to_stage))
    start_index = identifiers.index(from_stage) if from_stage else 0
    end_index = identifiers.index(to_stage) if to_stage else len(identifiers) - 1
    if start_index > end_index:
        raise ManifestError("from_stage 不能位于 to_stage 之后")
    selected_stages = config["stages"][start_index : end_index + 1]
    if dry_run:
        return {
            "status": "dry_run",
            "pipeline_id": config["pipeline_id"],
            "stages": [
                {"id": stage["id"], "kind": stage["kind"], "outputs": stage.get("outputs", [])}
                for stage in selected_stages
            ],
        }

    log_dir = state_path.parent / "edge_llm_pipeline_logs"
    for stage in selected_stages:
        identifier = stage["id"]
        previous = state["stages"].get(identifier)
        if resume and previous and previous.get("status") == "completed" and stage["kind"] not in {
            "verify_base",
            "validate_adapter",
            "publish_release",
        }:
            current_outputs = _collect_outputs(stage, root)
            if current_outputs != previous.get("outputs"):
                raise ManifestError("已完成阶段 {} 的产物发生变化".format(identifier))
            continue
        _require_inputs(stage, root)
        started = time.time()
        state["stages"][identifier] = {"status": "running", "started_at_unix": started}
        _write_state(state_path, state)
        try:
            result = _execute_stage(stage, root, log_dir)
            outputs = _collect_outputs(stage, root)
        except Exception as exc:
            state["stages"][identifier] = {
                "status": "failed",
                "started_at_unix": started,
                "finished_at_unix": time.time(),
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
            _write_state(state_path, state)
            raise
        state["stages"][identifier] = {
            "status": "completed",
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "result": result,
            "outputs": outputs,
        }
        _write_state(state_path, state)
    full_pipeline = start_index == 0 and end_index == len(identifiers) - 1
    state["status"] = "completed" if full_pipeline else "completed_selected_range"
    state["selected_range"] = {
        "from_stage": identifiers[start_index],
        "to_stage": identifiers[end_index],
    }
    _write_state(state_path, state)
    return state


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="执行标准边缘大模型蒸馏与发布流水线。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--from_stage", default=None)
    parser.add_argument("--to_stage", default=None)
    args = parser.parse_args(argv)
    result = run_pipeline(
        Path(args.config),
        Path(args.project_root),
        resume=args.resume,
        dry_run=args.dry_run,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
