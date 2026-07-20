"""用途：以通用任务提示测量 Ollama 文本基座的常驻 TTFT、时延和 RSS。"""

import argparse
import json
import statistics
import threading
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from edge_llm_factory.contracts import ManifestError, write_json_object


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError("{}:{} JSON 无效".format(path, line_number)) from exc
            if not isinstance(value, dict):
                raise ManifestError("{}:{} 必须是 JSON object".format(path, line_number))
            rows.append(value)
    return rows


def _select_balanced(rows: Sequence[Dict[str, Any]], per_category: int) -> List[Dict[str, Any]]:
    if per_category <= 0:
        raise ManifestError("limit_per_category 必须大于 0")
    selected = []
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        category = str(row.get("category", ""))
        if category not in {"math", "code", "natural_language_reasoning"}:
            continue
        if counts[category] >= per_category:
            continue
        selected.append(row)
        counts[category] += 1
    if set(counts) != {"math", "code", "natural_language_reasoning"}:
        raise ManifestError("运行时基准必须包含数学、代码和自然语言推理")
    return selected


def _messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    category = row["category"]
    if category == "math":
        system = "Solve the math problem concisely and end with FINAL: <number>."
        user = str(row["prompt"])
    elif category == "code":
        system = "Write only a concise correct Python function without Markdown."
        tests = "\n".join(str(value) for value in row.get("test_list", []))
        user = "{}\nTests:\n{}".format(row["prompt"], tests)
    else:
        system = "回答中文单项选择题，只输出 A、B、C 或 D。"
        user = str(row["prompt"])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _read_rss_kb(pid: int) -> int:
    try:
        for line in (Path("/proc") / str(pid) / "status").read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return 0
    return 0


def _ollama_pids() -> List[int]:
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_text(encoding="utf-8", errors="ignore")
            comm = (proc / "comm").read_text(encoding="utf-8", errors="ignore").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        executable = Path(cmdline.split("\0", 1)[0]).name
        if comm in {"ollama", "llama-server"} or executable in {"ollama", "llama-server"}:
            pids.append(int(proc.name))
    return sorted(set(pids))


def ollama_rss_mb() -> float:
    return sum(_read_rss_kb(pid) for pid in _ollama_pids()) / 1024.0


class MemorySampler:
    def __init__(self, interval_ms: float) -> None:
        self.interval_s = max(0.005, interval_ms / 1000.0)
        self.values: List[float] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            self.values.append(ollama_rss_mb())
            self.stop_event.wait(self.interval_s)

    def start(self) -> None:
        self.values.append(ollama_rss_mb())
        self.thread.start()

    def stop(self) -> None:
        self.values.append(ollama_rss_mb())
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return max(self.values) if self.values else 0.0


def _post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    return value if isinstance(value, dict) else {}


def unload(host: str, model: str) -> None:
    try:
        _post_json(host.rstrip("/") + "/api/generate", {"model": model, "keep_alive": 0}, 60)
    except Exception:  # noqa: BLE001
        return


def _resident_model_info(host: str, model: str) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/ps", timeout=30) as response:
            value = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    expected = model.removesuffix(":latest")
    for item in value.get("models", []):
        name = str(item.get("name", "")).removesuffix(":latest")
        if name == expected:
            return {
                "size_bytes": int(item.get("size", 0)),
                "size_vram_bytes": int(item.get("size_vram", 0)),
                "digest": item.get("digest"),
            }
    return {}


def _run_one(
    host: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    num_ctx: int,
    num_predict: int,
    num_gpu: Optional[int],
    timeout: int,
    sample_interval_ms: float,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": True,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    if num_gpu is not None:
        payload["options"]["num_gpu"] = num_gpu
    request = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    sampler = MemorySampler(sample_interval_ms)
    sampler.start()
    started = time.perf_counter()
    first_content = None
    final: Dict[str, Any] = {}
    parts = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                chunk = json.loads(raw_line.decode("utf-8"))
                content = str(chunk.get("message", {}).get("content", ""))
                if content and first_content is None:
                    first_content = time.perf_counter()
                if content:
                    parts.append(content)
                if chunk.get("done"):
                    final = chunk
    finally:
        sampler.stop()
    finished = time.perf_counter()
    return {
        "ttft_ms": ((first_content or finished) - started) * 1000.0,
        "total_latency_ms": (finished - started) * 1000.0,
        "peak_rss_mb": sampler.peak_mb,
        "prompt_tokens": int(final.get("prompt_eval_count", 0)),
        "output_tokens": int(final.get("eval_count", 0)),
        "load_duration_ms": float(final.get("load_duration", 0)) / 1_000_000.0,
        "prompt_eval_duration_ms": float(final.get("prompt_eval_duration", 0)) / 1_000_000.0,
        "eval_duration_ms": float(final.get("eval_duration", 0)) / 1_000_000.0,
        "output": "".join(parts).strip(),
    }


def _summary(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * 0.95)))
    return {
        "mean": round(statistics.fmean(values), 4),
        "p50": round(statistics.median(values), 4),
        "p95": round(ordered[p95_index], 4),
        "max": round(max(values), 4),
    }


def _parse_model(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ManifestError("model 必须使用 label=name 格式")
    label, model = (part.strip() for part in value.split("=", 1))
    if not label or not model:
        raise ManifestError("model label 和 name 不能为空")
    return label, model


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="测量通用文本基座的常驻 Ollama 资源开销。")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--limit_per_category", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--num_ctx", type=int, default=512)
    parser.add_argument("--num_predict", type=int, default=32)
    parser.add_argument("--num_gpu", type=int)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sample_interval_ms", type=float, default=10.0)
    args = parser.parse_args(argv)

    dataset = Path(args.dataset_jsonl).resolve()
    output = Path(args.output).resolve()
    if not dataset.is_file():
        raise ManifestError("运行时基准数据不存在")
    if output.exists():
        raise ManifestError("拒绝覆盖已有运行时基准结果")
    rows = _select_balanced(_read_jsonl(dataset), args.limit_per_category)
    models = [_parse_model(value) for value in args.model]
    if len({label for label, _ in models}) != len(models):
        raise ManifestError("model label 必须唯一")

    result: Dict[str, Any] = {
        "task": "scene_independent_llm_runtime_benchmark",
        "dataset_jsonl": str(dataset),
        "sample_count": len(rows),
        "category_counts": dict(sorted(defaultdict(int, {category: sum(row["category"] == category for row in rows) for category in {row["category"] for row in rows}}).items())),
        "no_thinking": True,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "num_gpu": args.num_gpu,
        "models": {},
    }
    for label, model in models:
        unload(args.host, model)
        baseline_rss = ollama_rss_mb()
        for index in range(args.warmup):
            _run_one(
                args.host,
                model,
                _messages(rows[index % len(rows)]),
                args.num_ctx,
                args.num_predict,
                args.num_gpu,
                args.timeout,
                args.sample_interval_ms,
            )
        resident = _resident_model_info(args.host, model)
        samples = [
            {
                "sample_id": row.get("sample_id"),
                "category": row.get("category"),
                **_run_one(
                    args.host,
                    model,
                    _messages(row),
                    args.num_ctx,
                    args.num_predict,
                    args.num_gpu,
                    args.timeout,
                    args.sample_interval_ms,
                ),
            }
            for row in rows
        ]
        peak_rss = max(sample["peak_rss_mb"] for sample in samples)
        result["models"][label] = {
            "model": model,
            "baseline_ollama_rss_mb": round(baseline_rss, 4),
            "peak_ollama_rss_mb": round(peak_rss, 4),
            "incremental_peak_rss_mb": round(max(0.0, peak_rss - baseline_rss), 4),
            "resident_model": resident,
            "ttft_ms": _summary([sample["ttft_ms"] for sample in samples]),
            "total_latency_ms": _summary([sample["total_latency_ms"] for sample in samples]),
            "average_prompt_tokens": round(
                statistics.fmean(sample["prompt_tokens"] for sample in samples), 3
            ),
            "average_output_tokens": round(
                statistics.fmean(sample["output_tokens"] for sample in samples), 3
            ),
            "samples": samples,
        }
        unload(args.host, model)
    write_json_object(output, result)
    printable = {
        label: {key: value for key, value in model.items() if key != "samples"}
        for label, model in result["models"].items()
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
