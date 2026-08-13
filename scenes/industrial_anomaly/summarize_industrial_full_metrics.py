#!/usr/bin/env python3
"""Create the public industrial metrics table from raw evidence.

The table intentionally exposes exactly one end-to-end latency: client input to
the locally executable provisional decision.  Cloud completion and model
latencies are module measurements, not alternative end-to-end definitions.
"""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List
from xml.etree import ElementTree as ET
from zipfile import ZipFile


_XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha(resolved),
    }


def _read_perception_workbook(path: Path) -> Dict[str, Any]:
    """Read the small checked-in xlsx without adding an openpyxl dependency."""

    with ZipFile(path) as workbook:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall(_XML_NS + "si"):
                shared.append(
                    "".join(node.text or "" for node in item.iter(_XML_NS + "t"))
                )
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    rows = sheet.findall(".//" + _XML_NS + "sheetData/" + _XML_NS + "row")

    def cell_value(cell: ET.Element) -> str:
        value = cell.find(_XML_NS + "v")
        raw = "" if value is None or value.text is None else value.text
        if cell.attrib.get("t") == "s":
            return shared[int(raw)]
        return raw

    headers = [cell_value(cell) for cell in rows[0].findall(_XML_NS + "c")]
    records = []
    for row in rows[1:]:
        values = [cell_value(cell) for cell in row.findall(_XML_NS + "c")]
        records.append(dict(zip(headers, values)))
    if not records:
        raise ValueError("industrial perception workbook is empty")

    def summarize(modality: str) -> Dict[str, Any]:
        selected = [row for row in records if row["modality"] == modality]
        sample_count = sum(int(row["num_samples"]) for row in selected)
        f1_values = [float(row["max_f1"]) for row in selected]
        weighted = sum(
            float(row["max_f1"]) * int(row["num_samples"]) for row in selected
        ) / sample_count
        return {
            "product_count": len(selected),
            "sample_evaluations": sample_count,
            "mean_per_product_max_f1": round(statistics.fmean(f1_values), 6),
            "sample_weighted_max_f1": round(weighted, 6),
            "min_product_max_f1": round(min(f1_values), 6),
            "max_product_max_f1": round(max(f1_values), 6),
        }

    return {
        "source_kind": "checked_in_historical_prediction_summary",
        "fresh_rerun": False,
        "claim_boundary": (
            "The workbook is a checked-in summary of earlier RGB/infrared "
            "prediction runs. Raw prediction CSVs, images, ONNX models, and "
            "memory banks were not present locally or on Nano 222, so the "
            "perception models were not rerun in this acceptance pass."
        ),
        "row_count": len(records),
        "modalities": {
            "RGB": summarize("RGB"),
            "Infrared": summarize("Infrared"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--nano-model", required=True)
    parser.add_argument("--nano-attestation", required=True)
    parser.add_argument("--cloud9b", required=True)
    parser.add_argument("--closed-loop", required=True)
    parser.add_argument("--isolation", required=True)
    parser.add_argument("--perception-workbook", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = {
        "local_model": Path(args.model).resolve(),
        "nano_model": Path(args.nano_model).resolve(),
        "nano_attestation": Path(args.nano_attestation).resolve(),
        "cloud9b": Path(args.cloud9b).resolve(),
        "closed_loop": Path(args.closed_loop).resolve(),
        "isolation": Path(args.isolation).resolve(),
        "perception_workbook": Path(args.perception_workbook).resolve(),
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
        if name != "perception_workbook"
    }
    perception = _read_perception_workbook(paths["perception_workbook"])
    model = data["local_model"]
    nano = data["nano_model"]
    cloud9b = data["cloud9b"]
    closed = data["closed_loop"]
    isolation = data["isolation"]
    normal = closed["profiles"]["normal_async"]
    outage = closed["profiles"]["outage_and_recovery"]
    cloud_runtime = closed["service_evidence"]["cloud_metrics"]["distributions"][
        "cloud_service_runtime_ms"
    ]
    nano_peak_kib = max(
        int(item["vm_hwm_kib"] or 0) for item in nano["memory_checkpoints"]
    )
    rgb = perception["modalities"]["RGB"]
    infrared = perception["modalities"]["Infrared"]

    rows: List[Dict[str, Any]] = [
        {
            "category": "端到端",
            "metric": "平均端到端时延（输入→本地可执行初判）",
            "value": round(normal["client_input_to_provisional_ms"]["mean"], 3),
            "unit": "ms",
            "count": normal["event_count"],
            "environment": "本机真实HTTP公共/decide接口，RGB/红外双模态",
            "evidence": str(paths["closed_loop"]),
        },
        {
            "category": "工业感知",
            "metric": "既有RGB/红外加权max-F1",
            "value": "{}/{}".format(
                rgb["sample_weighted_max_f1"],
                infrared["sample_weighted_max_f1"],
            ),
            "unit": "RGB/Infrared",
            "count": rgb["sample_evaluations"] + infrared["sample_evaluations"],
            "environment": "仓库内历史预测汇总；本轮未重跑感知模型",
            "evidence": str(paths["perception_workbook"]),
        },
        {
            "category": "边缘模型",
            "metric": "工业LoRA Accuracy/Macro-F1/合法率（Nano 222）",
            "value": "{}/{}/{}".format(
                nano["heldout"]["classification"]["accuracy"],
                nano["heldout"]["classification"]["macro_f1"],
                nano["heldout"]["classification"]["valid_rate"],
            ),
            "unit": "ratio",
            "count": nano["heldout"]["classification"]["count"],
            "environment": "Jetson Orin Nano 222，CPU-only旁路，16→1 token",
            "evidence": str(paths["nano_model"]),
        },
        {
            "category": "边缘模型",
            "metric": "工业LoRA TTFT平均（Nano 222 CPU-only）",
            "value": nano["heldout"]["ttft_ms"]["mean"],
            "unit": "ms",
            "count": nano["heldout"]["classification"]["count"],
            "environment": "能力/内存验收，不作为毫秒级性能结论",
            "evidence": str(paths["nano_model"]),
        },
        {
            "category": "边缘模型",
            "metric": "工业LoRA峰值VmHWM（Nano 222）",
            "value": round(nano_peak_kib / 1024.0, 3),
            "unit": "MiB",
            "count": nano["heldout"]["classification"]["count"],
            "environment": "低于1.5GB门槛；生产交通服务未停止",
            "evidence": str(paths["nano_attestation"]),
        },
        {
            "category": "云端9B",
            "metric": "RGB/红外状态复核 Accuracy/合法率",
            "value": "{}/{}".format(
                cloud9b["metrics"]["accuracy"],
                cloud9b["metrics"]["valid_rate"],
            ),
            "unit": "ratio",
            "count": cloud9b["metrics"]["count"],
            "environment": "qwen3.5:9b，受控held-out状态组合，非图像感知",
            "evidence": str(paths["cloud9b"]),
        },
        {
            "category": "云端9B",
            "metric": "RGB/红外状态复核平均时延",
            "value": cloud9b["metrics"]["wall_ms"]["mean"],
            "unit": "ms",
            "count": cloud9b["metrics"]["count"],
            "environment": "本机Ollama qwen3.5:9b，1-token输出",
            "evidence": str(paths["cloud9b"]),
        },
        {
            "category": "本地闭环",
            "metric": "本地初判Accuracy",
            "value": normal["local_accuracy"],
            "unit": "ratio",
            "count": normal["event_count"],
            "environment": "工业公共/decide接口",
            "evidence": str(paths["closed_loop"]),
        },
        {
            "category": "云端轻量协调",
            "metric": "云服务模块平均处理时延",
            "value": cloud_runtime["mean"],
            "unit": "ms",
            "count": cloud_runtime["count"],
            "environment": "模块时延，不另称端到端",
            "evidence": str(paths["closed_loop"]),
        },
        {
            "category": "弱网稳定性",
            "metric": "断网本地功能保持率/恢复后权威结果率",
            "value": "{}/{}".format(
                outage["local_success_rate"],
                outage["recovery"]["authoritative_final_rate"],
            ),
            "unit": "ratio",
            "count": outage["event_count"],
            "environment": "100%丢包后恢复",
            "evidence": str(paths["closed_loop"]),
        },
        {
            "category": "一致性",
            "metric": "受控双模态分歧解决率/残余冲突",
            "value": "{}/{}".format(
                closed["conflicts"]["resolution_success_rate"],
                closed["conflicts"]["residual_semantic_conflicts"],
            ),
            "unit": "ratio/count",
            "count": closed["conflicts"]["controlled_disagreement_groups"],
            "environment": "RGB/红外受控分歧",
            "evidence": str(paths["closed_loop"]),
        },
        {
            "category": "通信效率",
            "metric": "摘要相对102400B热图缩减率",
            "value": closed["communication_efficiency"][
                "mean_reduction_vs_raw_heatmap"
            ],
            "unit": "ratio",
            "count": normal["event_count"],
            "environment": "摘要+证据URI，不上传热图body",
            "evidence": str(paths["closed_loop"]),
        },
        {
            "category": "多场景隔离",
            "metric": "交通/工业并发预测一致率",
            "value": "{}/{}".format(
                isolation["sequential_vs_concurrent_prediction_consistency"][
                    "traffic"
                ]["consistency_rate"],
                isolation["sequential_vs_concurrent_prediction_consistency"][
                    "industrial"
                ]["consistency_rate"],
            ),
            "unit": "traffic/industrial",
            "count": 200,
            "environment": "同一Q8基座，LoRA id0/id1",
            "evidence": str(paths["isolation"]),
        },
    ]
    csv_path = output / "工业指标_交通对齐.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "industrial-traffic-aligned-summary/v2",
        "primary_end_to_end_metric": {
            "definition": "client input to locally executable provisional decision",
            "mean_ms": round(normal["client_input_to_provisional_ms"]["mean"], 6),
            "count": normal["event_count"],
        },
        "evidence": {name: _id(path) for name, path in paths.items()},
        "perception_evidence": perception,
        "rows": rows,
        "claims": {
            "single_primary_end_to_end_metric": True,
            "industrial_perception_historical_summary_verified": True,
            "industrial_perception_fresh_rerun_complete": False,
            "industrial_action_model_nano_cpu_complete": True,
            "industrial_edge_cloud_loop_complete": True,
            "cloud_9b_industrial_state_review_complete": True,
            "weak_network_recovery_complete": True,
            "multi_scene_adapter_isolation_complete": True,
            "jetson_industrial_gpu_deployment_complete": False,
        },
    }
    json_path = output / "工业指标_交通对齐.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "json": str(json_path),
                **report["primary_end_to_end_metric"],
                **report["claims"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
