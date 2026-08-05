import json
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import joblib
import numpy as np


PRODUCTS = [
    "button_cell", 
    "capsule", 
    "cotton", 
    "cube", 
    "piggy", 
    "plastic_cylinder", 
    "screw", "solar_panel", 
    "toothbrush", "zipper"
    ]

DEFAULT_BANDS = Path(__file__).parent  / "review_bands.json"
DEFAULT_MODEL = Path(__file__).parent  / "extratrees.joblib"
prediction_model = joblib.load(DEFAULT_MODEL)
review_bands = json.loads(Path(DEFAULT_BANDS).read_text(encoding="utf-8"))["bands"]

class CloudReviewModel:
    def __init__(self, 
                 model_path: Path, 
                #  evaluation_path: Path, 
                 confidence_threshold: float = 0.8, 
                 nine_b_enabled: bool = False
                ) -> None:
        
        self.model = prediction_model
        # self.evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
        self.confidence_threshold = float(confidence_threshold)
        self.nine_b_enabled = bool(nine_b_enabled)
        self.calls = 0
        self.nine_b_triggers = 0

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:

        probability = self.model.predict_proba([features])[0]
        anomaly = float(probability[1])
        confidence = max(anomaly, 1.0 - anomaly)
        label = "anomaly" if anomaly >= 0.5 else "normal"
        needs_9b = confidence < self.confidence_threshold
        self.calls += 1
        self.nine_b_triggers += int(needs_9b)

        return {
                "label": label, 
                "probabilities": 
                    {"normal": 1.0-anomaly, "anomaly": anomaly}, 
                "confidence": confidence, 
                "needs_9b_review": needs_9b, 
                "nine_b_enabled": self.nine_b_enabled, 
                "fallback": "extratrees_conservative_review" if needs_9b and not self.nine_b_enabled else "", 
                }

    def snapshot(self):
        return {
                "calls": self.calls, 
                "nine_b_triggers": self.nine_b_triggers, 
                "nine_b_enabled": self.nine_b_enabled
               }

def _map_features(path: Path, prefix: str) -> Tuple[Dict[str, float], np.ndarray]:
    values = np.fromfile(str(path), dtype="<f4")
    if values.size != 160 * 160 or not np.isfinite(values).all():
        raise ValueError("invalid 160x160 f32 evidence: {}".format(path))
    image = values.reshape(160, 160)
    q = np.quantile(values, [0.5, 0.9, 0.95, 0.99])
    y, x = np.unravel_index(int(np.argmax(image)), image.shape)
    features = {
        prefix + "mean": float(values.mean()), prefix + "std": float(values.std()),
        prefix + "min": float(values.min()), prefix + "max": float(values.max()),
        prefix + "q50": float(q[0]), prefix + "q90": float(q[1]),
        prefix + "q95": float(q[2]), prefix + "q99": float(q[3]),
        prefix + "top1pct_mean": float(values[values >= q[3]].mean()),
        prefix + "max_x": x / 159.0, prefix + "max_y": y / 159.0,
        prefix + "center_mean": float(image[40:120, 40:120].mean()),
        prefix + "border_mean": float(np.concatenate((image[:20].ravel(), image[-20:].ravel(), image[:, :20].ravel(), image[:, -20:].ravel())).mean()),
    }
    return features, values

def build_single_features(product, rgb_f32, infrared_f32, rgb_score, infrared_score, bands):
    """Return the 43-dim feature dict for one paired sample (same logic as build_records)."""
    if product not in PRODUCTS:
        raise ValueError("unknown product {!r}; expected one of {}".format(product, PRODUCTS))
    rgb_f32, infrared_f32 = Path(rgb_f32), Path(infrared_f32)
    rgb_stats, rgb_values = _map_features(rgb_f32, "rgb_map_")
    infra_stats, infra_values = _map_features(infrared_f32, "infrared_map_")
    rb, ib = bands[product]["rgb"], bands[product]["infrared"]
    return {
        "product=" + product: 1.0,
        "rgb_score": rgb_score, "infrared_score": infrared_score,
        "rgb_threshold_distance": (rgb_score - rb["review_low"]) / (rb["review_high"] - rb["review_low"]),
        "infrared_threshold_distance": (infrared_score - ib["review_low"]) / (ib["review_high"] - ib["review_low"]),
        "score_abs_difference": abs(rgb_score - infrared_score),
        **rgb_stats, **infra_stats,
        "map_abs_difference_mean": float(np.abs(rgb_values - infra_values).mean()),
        "map_correlation": float(np.corrcoef(rgb_values, infra_values)[0, 1]),
    }


def predict_single(product, rgb_f32, infrared_f32, rgb_score, infrared_score,
                   bands=DEFAULT_BANDS, model_path=DEFAULT_MODEL,
                    confidence_threshold=0.8,
                   nine_b_enabled=False):
    """Build features and run inference for one sample; return dict with both."""
    bands = review_bands
    features = build_single_features(product, rgb_f32, infrared_f32,
                                     float(rgb_score), float(infrared_score), bands)
    reviewer = CloudReviewModel(Path(model_path),
                                confidence_threshold=confidence_threshold,
                                nine_b_enabled=nine_b_enabled)
    return {"features": features, "prediction": reviewer.predict(features)}


# 以下测试
# start_time = time.perf_counter()
# result = predict_single("capsule", 
#                "/home/defaultval/cloud_edgeproj/cloud-edge-scene-sdk-v0131/scene_plugin_template/fetched_heatmaps/RGB_map_1.f32", 
#                "/home/defaultval/cloud_edgeproj/cloud-edge-scene-sdk-v0131/scene_plugin_template/fetched_heatmaps/Infrared_map_1.f32", 
#                0.5, 
#                0.5)
# print(f"Elapsed time: {time.perf_counter()- start_time} seconds")

# print("Prediction result:", result["prediction"])

