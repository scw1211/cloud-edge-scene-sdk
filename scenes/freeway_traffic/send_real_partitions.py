"""用途：在一台边缘节点上感知交通状态，并向本机边缘服务发送指定分区。"""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen
import uuid


SCENE_ROOT = Path(__file__).resolve().parent


def _partition_ids(spec):
    result = []
    for value in str(spec).split(","):
        partition_id = int(value.strip())
        if partition_id not in result:
            result.append(partition_id)
    if not result:
        raise ValueError("partitions 不能为空")
    return result


def _post(url, event, timeout_seconds):
    body = json.dumps(
        {"event": event}, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    request = Request(
        url.rstrip("/") + "/api/v1/collaboration/decide",
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Idempotency-Key": event["id"],
            "X-Trace-Id": "trace_" + event["id"],
        },
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload, round((time.perf_counter() - started) * 1000.0, 6), len(body)


def main():
    parser = argparse.ArgumentParser(description="发送真实交通分区事件")
    parser.add_argument("--edge-url", default="http://127.0.0.1:18101")
    parser.add_argument("--partitions", required=True, help="如 0,1 或 2,3")
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--perception-mode",
        choices=("current-state", "astgcn"),
        default=os.environ.get("TRAFFIC_PERCEPTION_MODE", "current-state"),
        help="current-state直接按观测窗口判断；astgcn保留原预测链路",
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="两台 Jetson 必须填写完全相同的实验编号",
    )
    parser.add_argument(
        "--start-at-ms",
        type=int,
        default=0,
        help="可选：两机共同的 Unix 毫秒发送时刻；模型会提前完成推理后等待",
    )
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument(
        "--aggregation-timeout-ms",
        type=int,
        default=1000,
        help="两机所有分区摘要到齐的窗口；最终报告应记录实测到达偏差。",
    )
    parser.add_argument(
        "--output",
        default=str(SCENE_ROOT / "evidence" / "real_partition_send_latest.json"),
    )
    args = parser.parse_args()

    from traffic_system.scene_event import traffic_event_from_output

    os.chdir(str(SCENE_ROOT.parents[1]))
    data_path = (
        SCENE_ROOT
        / "assets"
        / "downloads"
        / "PEMS08_r1_d0_w0_astcgn_multitask.npz"
    )
    if args.perception_mode == "current-state":
        from traffic_system.current_state_perception_runtime import (
            CurrentStateTrafficPerceptionRuntime,
        )

        runtime = CurrentStateTrafficPerceptionRuntime(
            data_path=data_path,
            rule_config_path=(
                SCENE_ROOT / "assets" / "models" / "current_state_perception_v1.json"
            ),
            topology_path=(
                SCENE_ROOT
                / "assets"
                / "models"
                / "traffic_region_topology_metis4.json"
            ),
            split=args.split,
            top_k=10,
        )
    else:
        import torch
        from traffic_system.traffic_perception_runtime import (
            JointTrafficPerceptionRuntime,
        )

        torch.set_num_threads(args.torch_threads)
        runtime = JointTrafficPerceptionRuntime(
            config_path=SCENE_ROOT / "configurations" / "PEMS08_astgcn.conf",
            data_path=data_path,
            checkpoint_path=(
                SCENE_ROOT
                / "assets"
                / "models"
                / "joint_risk_astgcn_metis4_flowprio2_frozen.pt"
            ),
            risk_calibrator_path=(
                SCENE_ROOT / "assets" / "models" / "region_risk_conformal.json"
            ),
            split=args.split,
            device_name=args.device,
            top_k=10,
        )
    runtime.warmup(args.sample_id)
    perception = runtime.infer_sample(args.sample_id)
    selected = set(_partition_ids(args.partitions))
    invalid = sorted(selected - set(range(runtime.partition_count)))
    if invalid:
        raise ValueError("分区编号越界：{}".format(invalid))

    prepared = []
    for native in perception.events:
        if int(native["partition_id"]) not in selected:
            continue
        measured = copy.deepcopy(native)
        measured["sample_split"] = "{}_{}".format(args.split, args.experiment_id)
        measured["event_id"] = "{}_{}".format(
            measured["event_id"], args.experiment_id
        )
        measured["aggregation_timeout_ms"] = args.aggregation_timeout_ms
        envelope = traffic_event_from_output(measured)
        prepared.append(
            (int(measured["partition_id"]), envelope)
        )
    if not prepared:
        raise RuntimeError("感知结果中没有找到请求的分区")

    if args.start_at_ms:
        remaining_seconds = (args.start_at_ms / 1000.0) - time.time()
        if remaining_seconds < -0.5:
            raise ValueError("start-at-ms 已经过期")
        if remaining_seconds > 0:
            time.sleep(remaining_seconds)

    dispatch_zero = time.perf_counter()

    def submit(partition_id, envelope):
        dispatch_ms = (time.perf_counter() - dispatch_zero) * 1000.0
        response, wall_ms, request_bytes = _post(
            args.edge_url, envelope, args.timeout_seconds
        )
        return {
            "partition_id": partition_id,
            "event_id": envelope["id"],
            "local_dispatch_ms": round(dispatch_ms, 6),
            "client_wall_ms": wall_ms,
            "request_bytes": request_bytes,
            "route": response["final_decision"]["route"],
            "status": response["final_decision"]["status"],
            "edge_decision_path": response["local_decision"]
            .get("metadata", {})
            .get("edge_decision_path"),
        }

    records = []
    with ThreadPoolExecutor(max_workers=len(prepared)) as executor:
        futures = [
            executor.submit(submit, partition_id, envelope)
            for partition_id, envelope in prepared
        ]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["partition_id"])
    dispatch_values = [item["local_dispatch_ms"] for item in records]

    result = {
        "status": "submitted",
        "experiment_id": args.experiment_id,
        "sample_id": args.sample_id,
        "sample_split": "{}_{}".format(args.split, args.experiment_id),
        "edge_url": args.edge_url,
        "perception_mode": args.perception_mode,
        "device": str(runtime.device),
        "model_load_ms": runtime.load_latency_ms,
        "model_forward_ms": perception.model_forward_ms,
        "perception_ms": perception.perception_ms,
        "local_dispatch_spread_ms": round(
            max(dispatch_values) - min(dispatch_values), 6
        )
        if dispatch_values
        else 0.0,
        "records": records,
    }
    output = Path(args.output).resolve()
    if output.name == "real_partition_send_latest.json":
        output = output.with_name(
            "real_partition_send_{}_{}.json".format(
                args.experiment_id, uuid.uuid4().hex[:6]
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**result, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
