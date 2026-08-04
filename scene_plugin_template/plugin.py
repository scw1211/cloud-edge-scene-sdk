"""用途：演示场景原生模型输出如何通过插件接入统一云边运行时。"""

import base64
import fcntl
import hashlib
import json
import socket
import struct
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from cloud_edge_framework.contracts import (
    Action,
    DecisionEnvelope,
    Evidence,
    EventScope,
    Prediction,
    Risk,
    SemanticEvent,
    Timing,
    Uncertainty,
    build_decision,
)
from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.plugins.base import ScenePlugin
import os
import pandas as pd
from pathlib import Path

# 当前代码同级目录下的 xlsx
xlsx_path = Path(__file__).parent / "summary_f1_review_bands.xlsx"

df = pd.read_excel(xlsx_path)

review_bands = {}
history_decision = {}
history_decision["RGB"] = {}
history_decision["Infrared"] = {}

for _, row in df.iterrows():
    modality = row["modality"]
    product = row["product"]

    if modality not in review_bands:
        review_bands[modality] = {}

    if product not in review_bands[modality]:
        review_bands[modality][product] = {}

    review_bands[modality][product]["review_low"] = row["review_low"]
    review_bands[modality][product]["review_high"] = row["review_high"]


def _fetch_edge_heatmap(heatmap_uri, edge_url) -> Dict[str, Any]:
    """请求边缘节点把 heatmap_uri 指向的原始文件字节发送过来，保存到本地并返回信息。"""
    if not heatmap_uri or not edge_url:
        raise ValueError(
            "heatmap_uri and edge_url must both be provided, got {!r} and {!r}".format(
                heatmap_uri, edge_url
            )
        )
    raw_path = heatmap_uri
    if raw_path.startswith("file://"):
        raw_path = raw_path[len("file://"):]
    encoded = base64.urlsafe_b64encode(raw_path.encode("utf-8")).decode("ascii")
    url = "{}/api/v1/collaboration/evidence-file/{}".format(edge_url, encoded)
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/octet-stream"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            content = response.read()  # 直接拿原始字节，不做 base64 编解码
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "edge returned HTTP {} for {}: {}".format(exc.code, url, detail)
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            "edge request failed for {}: {}: {}".format(url, type(exc).__name__, exc)
        ) from exc

    digest = hashlib.sha256(content).hexdigest()
    dest_dir = Path(__file__).parent / "fetched_heatmaps"
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(raw_path).name or "heatmap_{}.bin".format(digest[:8])
    dest = dest_dir / name
    dest.write_bytes(content)
    return {
        "source": heatmap_uri,
        "local_path": str(dest),
        "size_bytes": len(content),
        "sha256": digest,
    }


def _call_ollama_review(
    fetched: Optional[Dict[str, Any]],
    ot_fetched: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """调用云端千问（Ollama），把两个模态热力图文件信息作为 prompt 进行复核。

    返回模型输出文本与调用耗时；配置复用同级 cloud_qwen9b_ollama.json。
    """
    if config_path is None:
        config_path = Path(__file__).parent / "cloud_qwen9b_ollama.json"
    with config_path.open("r", encoding="utf-8") as file_obj:
        config = json.load(file_obj)

    endpoint = str(config["endpoint"]).rstrip("/")
    generation = config.get("generation", {}) or {}

    prompt_context = dict(context or {})
    prompt_context.update(
        {
            "current_modality_heatmap": dict(fetched or {}),
            "other_modality_heatmap": dict(ot_fetched or {}),
        }
    )
    prompt = (
        "请作为工业异常多模态复核专家，根据以下两个模态热力图文件的信息，"
        "判断该样本是否确实存在异常、是否需要触发人工复核或限制措施。\n"
        "只返回一个 JSON 对象，键为："
        "verdict(recommend 或 challenge)、recommended_decision(no_action/review/set_operating_limit)、"
        "confidence(0~1)、reason(简短中文理由)。不要返回 markdown 或其他文字。\n"
        "输入信息：\n"
        + json.dumps(prompt_context, ensure_ascii=False, indent=2)
    )

    body = {
        "model": str(config["model"]),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": bool(generation.get("thinking", False)),
        "keep_alive": str(generation.get("keep_alive", "30m")),
        "options": {
            "temperature": float(generation.get("temperature", 0)),
            "top_p": float(generation.get("top_p", 1)),
            "seed": int(generation.get("seed", 42)),
            "num_ctx": int(generation.get("max_input_tokens", 2048)),
            "num_predict": int(generation.get("max_output_tokens", 128)),
        },
    }
    request = urllib.request.Request(
        endpoint + "/api/chat",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    try:
        started = time.perf_counter()
        with urllib.request.urlopen(
            request, timeout=float(config.get("timeout_seconds", 30))
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
        print(f"Elapsed time: {time.perf_counter()- started} seconds")
    except urllib.error.HTTPError as exc:
        detail = exc.read(1024).decode("utf-8", errors="replace")
        raise RuntimeError("qwen ollama HTTP {}: {}".format(exc.code, detail)) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("qwen ollama request failed: {}".format(exc)) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("qwen ollama returned invalid JSON") from exc

    message = result.get("message", {})
    if not isinstance(message, dict):
        raise RuntimeError("qwen ollama response is missing message")
    print(message.get("content", ""))
    return {
        "provider": "ollama",
        "model": str(config["model"]),
        "text": str(message.get("content", "")).strip(),
        "latency_ms": round((time.perf_counter() - started) * 1000, 4),
        "prompt_tokens": result.get("prompt_eval_count"),
        "output_tokens": result.get("eval_count"),
    }


class ExampleScenePlugin(ScenePlugin):
    """可运行的异常检测接入模板，不代表任何真实场景模型效果。"""

    scene = "industrial_anomaly"
    aliases = ("example",)
    event_types = ("com.example.industrial.anomaly-map.v1",)

    def __init__(
        self,
        policy_version: str = "example-0.2.0",
        template_mode: bool = True,
    ) -> None:
        self.policy_version = str(policy_version)
        self.template_mode = bool(template_mode)
        self._payload_schema = None

    def payload_schema(self) -> Dict[str, Any]:
        if self._payload_schema is None:
            schema_path = Path(__file__).with_name("data_schema.json")
            with schema_path.open("r", encoding="utf-8") as file_obj:
                self._payload_schema = json.load(file_obj)
        return dict(self._payload_schema)

    @staticmethod
    def _risk_level(score: float, threshold: float) -> str:
        margin = score - threshold
        if margin < 0:
            return "low"
        if margin < 0.10:
            return "medium"
        if margin < 0.25:
            return "high"
        return "severe"

    def normalize(self, envelope: SceneEventEnvelope) -> SemanticEvent:
        """把异常分数和热力图引用映射为内部调度语义。"""
        self.validate_envelope(envelope)
        if not isinstance(envelope.data, dict):
            raise ValueError("industrial anomaly data must be an object")
        payload = dict(envelope.data)
        score = float(payload["anomaly_score"])
        threshold = float(payload["threshold"])
        confidence = float(payload.get("confidence", score))
        risk_level = self._risk_level(score, threshold)
        # asset_id = str(payload["asset_id"])
        # resource_id = str(payload["shared_resource"])
        # heatmap = dict(payload["heatmap"])
        window_ms = int(payload.get("window_ms", 5000))

        actions = [
            Action(
                action_type="set_operating_limit",
                # target_ids=[asset_id],
                # resource_ids=[resource_id],
                parameters={
                    "min_risk_level": "high",
                    # "limit_percent": int(payload["proposed_limit_percent"]),
                },
                reason="hold throughput while a human reviews the anomaly evidence",
                priority=80,
            )
        ]
        evidence = [
            Evidence(
                evidence_id=envelope.event_id + "_summary",
                level="summary",
                modality="anomaly_summary",
                encoding="json",
                inline={
                    "anomaly_score": score,
                    "threshold": threshold,
                    "confidence": confidence,
                },
                size_bytes=64,
                content_type="application/json",
            ),
            # Evidence(
            #     evidence_id=envelope.event_id + "_heatmap",
            #     level="feature",
            #     modality="anomaly_heatmap",
            #     encoding=str(heatmap["encoding"]),
            #     uri=str(heatmap["uri"]),
            #     shape=[int(value) for value in heatmap["shape"]],
            #     size_bytes=int(heatmap["size_bytes"]),
            #     content_type=str(heatmap.get("content_type", "application/octet-stream")),
            #     codec={"name": str(heatmap["encoding"]), "version": 1},
            # ),
        ]
        return SemanticEvent(
            event_id=envelope.event_id,
            scene=self.scene,
            task="industrial_anomaly_review",
            edge_id=envelope.edge_id,
            occurred_at_ms=envelope.occurred_at_ms,
            scope=EventScope(
                # entity_id=asset_id,
                # subsystem=str(payload["subsystem"]),
                state_variable="anomaly_state",
                # region_id=str(payload["region_id"]),
                # shared_resources=[resource_id],
                # correlation_keys=[asset_id + ":anomaly", resource_id],
                window_start_ms=envelope.occurred_at_ms - window_ms,
                window_end_ms=envelope.occurred_at_ms,
            ),
            prediction=Prediction(
                label="anomaly" if score >= threshold else "normal",
                confidence=confidence,
                values={"anomaly_score": score, "threshold": threshold},
            ),
            risk=Risk(level=risk_level, score=score),
            uncertainty=Uncertainty(
                confidence=confidence,
                calibrated=bool(payload.get("calibrated", False)),
                prediction_set=[risk_level],
                method=str(payload.get("confidence_method", "model_score")),
            ),
            timing=Timing(
                deadline_ms=float(payload.get("deadline_ms", 200)),
                preprocessing_ms=float(payload.get("preprocessing_latency_ms", 0)),
                edge_inference_ms=float(payload.get("inference_latency_ms", 0)),
            ),
            evidence=evidence,
            candidate_actions=actions,
            model=dict(payload.get("model", {})),
            scene_payload=payload,
            metadata={
                "adapter": "industrial_anomaly_map_v1",
                "ingress_type": envelope.event_type,
                "ingress_dataschema": envelope.dataschema,
                "transport_include_scene_payload": False,
            },
        )

    def edge_decide(self, event: SemanticEvent) -> DecisionEnvelope:
        """正式接入时可替换为场景 LoRA；模板只验证动作边界。"""
        selected = []
        # 获取物品分类
        raw_uri = event.scene_payload["raw_uri"]
        # 去掉 file:// 前缀
        path = raw_uri.replace("file://", "")
        # 获取路径各级目录
        parts = path.split(os.sep)
        class_type = parts[parts.index("MulSen_AD") + 1].lower()
        mode = parts[parts.index("MulSen_AD") + 2]
        index = parts[-1]
        
        # 根据物品分类设置风险等级和置信度
        review_low = review_bands.get(mode, {}).get(class_type, {}).get("review_low")
        review_high = review_bands.get(mode, {}).get(class_type, {}).get("review_high")

        score = event.scene_payload["score"]

        if score < review_low: 
            final_decision = "no_action"        
            selected = []
            reason = "no operational risk detected"
        elif score < review_high:
            final_decision = "review"
        else:
            final_decision = "no_action"
            reason = "high operational risk detected"

        current_url = "http://" + get_ip_address("wlan0") + ":18101"

        event.metadata.update({
            "mode": mode,
            "type": class_type,
            "index": index,
            "decision": final_decision,
            # 与 raw_uri 同结构的 file:// 路径；scene_payload 默认不上云，这里单独随事件带给云端
            "heatmap_uri": event.scene_payload.get("heatmap_uri"),
            "edge_url": current_url
        })

        # TODO 发送云端请求
        # 把当前事件同步 POST 到云端决策接口，云端 CloudRuntime 会调用本插件的 cloud_decide
        cloud_endpoint = "http://192.168.31.160:18100/api/v1/collaboration/cloud-decision"
        request_body = json.dumps(
            {"event": event.to_dict(include_scene_payload=False)},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            cloud_endpoint,
            data=request_body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                cloud_result = json.loads(response.read().decode("utf-8"))
            print(
                "[ExampleScenePlugin] cloud decision for {}: {}".format(
                    event.event_id, cloud_result.get("decision")
                )
            )
        except urllib.error.HTTPError as cloud_exc:
            # 服务端返回了非 2xx，这里把响应体打出来：detail 字段会直接指出是哪个字段不合规
            body_text = cloud_exc.read().decode("utf-8", errors="replace")
            try:
                error_payload = json.loads(body_text)
                error_code = error_payload.get("error", "")
                detail = error_payload.get("detail", "")
            except (json.JSONDecodeError, AttributeError):
                error_code = ""
                detail = body_text
            print(
                "[ExampleScenePlugin] cloud HTTP {} for {}: error={!r} detail={!r}".format(
                    cloud_exc.code, event.event_id, error_code, detail
                )
            )
        except Exception as cloud_exc:  # noqa: BLE001
            print(
                "[ExampleScenePlugin] cloud request failed for {}: {}: {}".format(
                    event.event_id, type(cloud_exc).__name__, cloud_exc
                )
            )

        return self.decision_from_candidates(
            event,
            source="example_edge_placeholder",
            confidence=event.prediction.confidence,
        )

    def cloud_decide(self, event: SemanticEvent) -> DecisionEnvelope:
        # 以下是工业场景
        # 拆包
        scene = event.scene.split("_")[0].lower()
        if scene == "industrial":
            mode = event.metadata.get("mode")
            class_type = event.metadata.get("type")
            index = event.metadata.get("index")
            decision = event.metadata.get("decision")
            heatmap_uri = event.metadata.get("heatmap_uri")
            edge_url = event.metadata.get("edge_url")

            ot_mode = "RGB" if mode == "Infrared" else "Infrared"
            if class_type not in history_decision[mode]:
                history_decision[mode][class_type] = {}

            history_decision[mode][class_type][index] = [decision, edge_url, heatmap_uri]   # 在这里记录边缘判断的结果
            ot_decision = "waiting_decision"
            # 获取上一个模态的决策结果
            if history_decision[ot_mode].get(class_type, {}).get(index) is not None:
                ot_decision = history_decision[ot_mode][class_type][index][0]

            # 测试行
            # try:
            #     fetched = _fetch_edge_heatmap(str(heatmap_uri), edge_url)
            #     try:
            #         llm_result = _call_ollama_review(
            #             fetched,
            #             ot_fetched=None,
            #             context={
            #                 "mode": mode,
            #                 "type": class_type,
            #                 "index": index,
            #                 "edge_decision": decision,
            #             },
            #         )
            #         event.metadata["cloud_llm_review"] = {
            #             "text": llm_result["text"],
            #             "provider": llm_result["provider"],
            #             "model": llm_result["model"],
            #             "latency_ms": llm_result["latency_ms"],
            #         }
            #         print(
            #             "Qwen review for type: {}, index: {}, mode: {}: {} ({:.0f} ms)".format(
            #                 class_type,
            #                 index,
            #                 mode,
            #                 llm_result["text"][:200],
            #                 llm_result["latency_ms"],
            #             )
            #         )
            #     except Exception as llm_exc:  # noqa: BLE001
            #         print(
            #             "Qwen review failed for {}: {}: {}".format(
            #                 event.event_id, type(llm_exc).__name__, llm_exc
            #             )
            #         )
            #     print("[debug] fetched heatmap:", fetched)
            # except Exception as fetch_exc:  # noqa: BLE001
            #     print(
            #         "[debug] test fetch failed for {}: {}: {}".format(
            #             event.event_id, type(fetch_exc).__name__, fetch_exc
            #         )
            #     )

            # 现在判断是否需要复核
            if ot_decision == "review":
                # 上一个模态已经申请过复核，直接跳过
                pass
            elif decision == "review" or (ot_decision != "waiting_decision" and decision != ot_decision):
                # 当前模态申请复核，或者两个模态的决策不一致
                # 请求边缘节点把 heatmap_uri 指向的文件发送过来，供人工/专家复核
                if ot_decision != "waiting_decision":
                    ot_url = history_decision[ot_mode][class_type][index][1]
                    ot_heatmap_uri = history_decision[ot_mode][class_type][index][2]
                    ot_fetched = _fetch_edge_heatmap(str(ot_heatmap_uri), ot_url)
                else:
                    ot_fetched = None
                if heatmap_uri:
                    try:
                        fetched = _fetch_edge_heatmap(str(heatmap_uri), edge_url)
                        # 调用云端千问：把两个模态热力图文件信息作为 prompt
                        try:
                            llm_result = _call_ollama_review(
                                fetched,
                                ot_fetched=ot_fetched,
                                context={
                                    "mode": mode,
                                    "type": class_type,
                                    "index": index,
                                    "edge_decision": decision,
                                },
                            )
                            event.metadata["cloud_llm_review"] = {
                                "text": llm_result["text"],
                                "provider": llm_result["provider"],
                                "model": llm_result["model"],
                                "latency_ms": llm_result["latency_ms"],
                            }
                            print(
                                "Qwen review for type: {}, index: {}, mode: {}: {} ({:.0f} ms)".format(
                                    class_type,
                                    index,
                                    mode,
                                    llm_result["text"][:200],
                                    llm_result["latency_ms"],
                                )
                            )
                        except Exception as llm_exc:  # noqa: BLE001
                            print(
                                "Qwen review failed for {}: {}: {}".format(
                                    event.event_id, type(llm_exc).__name__, llm_exc
                                )
                            )
                        print(
                            "Fetched heatmap from edge for type: {}, index: {}, mode: {}: {}".format(
                                class_type, index, mode, fetched
                            )
                        )
                    except Exception as fetch_exc:  # noqa: BLE001
                        print(
                            "Failed to fetch heatmap from edge for {}: {}: {}".format(
                                event.event_id, type(fetch_exc).__name__, fetch_exc
                            )
                        )
                else:
                    print(
                        "No heatmap_uri in event for type: {}, index: {}, mode: {}".format(
                            class_type, index, mode
                        )
                    )
                print("Triggering review for type: {}, index: {}, mode: {}, decision: {}, {} decision: {}".format(class_type, index, mode, decision, ot_mode, ot_decision))
            else:
                # 当前模态不需要复核，且两个模态的决策一致
                pass
        


        ############################################################################################################

        return self.decision_from_candidates(
            event,
            source="example_cloud_placeholder",
            confidence=event.uncertainty.confidence,
        )

    def prepare_cloud_event(
        self,
        event: SemanticEvent,
        evidence_level: str,
    ) -> SemanticEvent:
        metadata = dict(event.metadata)
        metadata.update(
            {
                "transport_include_scene_payload": False,
                "selected_evidence_level": evidence_level,
            }
        )
        return replace(event, scene_payload={}, metadata=metadata)

    def fuse_cloud_context(
        self,
        events: Sequence[SemanticEvent],
    ) -> Sequence[SemanticEvent]:
        return list(events)

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "template_mode": self.template_mode,
            "policy_version": self.policy_version,
        }
    
def get_ip_address(interface="wlan0"):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ip = fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack("256s", interface.encode("utf-8"))
    )[20:24]

    return socket.inet_ntoa(ip)