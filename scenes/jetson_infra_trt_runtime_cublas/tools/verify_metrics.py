#!/usr/bin/env python3
"""用 C++ 输出的原始预测图计算指标，并和权威 base/Infra 基线做门禁比较。"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score


def labels_for(image_path, img_size):
    # 根据原始 PNG 路径回到 MulSen_AD 的 GT 目录，读取图像级标签和像素 mask。
    path = Path(image_path)
    defect = path.parent.name
    if defect == "good":
        return 0, None
    class_root = path.parents[3]
    csv_path = class_root / "RGB" / "GT" / defect / "data.csv"
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    row = rows[int(path.stem)]
    # Infra baseline 的图像标签采用 any = RGB or infrared or pointcloud；像素指标只评估 infrared mask。
    image_label = int(int(row["RGB"]) or int(row["infrared"]) or int(row["pointcloud"]))
    if int(row["infrared"]) == 0:
        return image_label, None
    mask_path = class_root / "Infrared" / "GT" / defect / path.name
    mask = Image.open(mask_path).convert("L").resize((img_size, img_size), Image.NEAREST)
    return image_label, (np.asarray(mask) > 127).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--class-name", required=True, choices=["capsule", "screw"])
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--output", help="Write rgb_module-compatible metrics.csv")
    parser.add_argument("--no-gate", action="store_true", help="Report metrics without applying deployment thresholds")
    args = parser.parse_args()
    image_scores, image_labels, pixel_scores, pixel_labels = [], [], [], []
    with open(Path(args.predictions) / "predictions.csv") as handle:
        for row in csv.DictReader(handle):
            # predictions.csv 中 map_file 指向 C++ 写出的 img_size x img_size float32 原始得分图。
            label, mask = labels_for(row["path"], args.img_size)
            map_path = Path(args.predictions) / row["map_file"]
            score_map = np.fromfile(str(map_path), dtype=np.float32).reshape(args.img_size, args.img_size)
            if mask is None:
                mask = np.zeros((args.img_size, args.img_size), dtype=np.uint8)
            image_scores.append(float(row["image_score"])); image_labels.append(label)
            pixel_scores.append(score_map.reshape(-1)); pixel_labels.append(mask.reshape(-1))
    pixel_scores = np.concatenate(pixel_scores); pixel_labels = np.concatenate(pixel_labels)
    # 计算图像级 AUROC 和像素级 AUROC/F1/AUPR/AP，字段名对齐 rgb_module 输出。
    precision, recall, _ = precision_recall_curve(pixel_labels, pixel_scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    actual = {"Image_ROCAUC": roc_auc_score(image_labels, image_scores), "Infra_Pixel_ROCAUC": roc_auc_score(pixel_labels, pixel_scores), "Infra_Pixel_F1": float(f1.max()), "Infra_Pixel_AUPR": auc(recall, precision), "Infra_Pixel_AP": average_precision_score(pixel_labels, pixel_scores)}
    with open(args.base) as handle:
        target = next(row for row in csv.DictReader(handle) if row["Method"] == "Infra" and row["Class"].lower() == args.class_name)
    required_ratios = {
        # 160 低延迟部署门禁：Image AUROC 放宽到原版 baseline 的 85%；
        # Infra Pixel AUROC 和 Infra Pixel F1 都要求大于原版 baseline 的 80%。
        "Image_ROCAUC": 0.85,
        "Infra_Pixel_ROCAUC": 0.80,
        "Infra_Pixel_F1": 0.80,
    }
    metric_columns = ["Method", "Image_ROCAUC", "Infra_Pixel_ROCAUC", "Infra_Pixel_F1", "Infra_Pixel_AUPR", "Infra_Pixel_AP", "Class"]
    metric_row = {
        "Method": "Infra",
        "Image_ROCAUC": round(actual["Image_ROCAUC"], 3),
        "Infra_Pixel_ROCAUC": round(actual["Infra_Pixel_ROCAUC"], 3),
        "Infra_Pixel_F1": round(actual["Infra_Pixel_F1"], 3),
        "Infra_Pixel_AUPR": round(actual["Infra_Pixel_AUPR"], 3),
        "Infra_Pixel_AP": round(actual["Infra_Pixel_AP"], 3),
        "Class": args.class_name.title(),
    }
    if args.output:
        # 可选写出 rgb_module 兼容格式，包含 RGB 行和 Mean/Overall 行。
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=metric_columns)
            writer.writeheader()
            writer.writerow(metric_row)
            mean_row = dict(metric_row)
            mean_row["Method"] = "Mean"
            mean_row["Class"] = "Overall"
            writer.writerow(mean_row)
        print("Wrote metrics:", output_path)
    failures = []
    for name, value in actual.items():
        baseline = float(target[name])
        if name not in required_ratios:
            print("%s=%.3f baseline=%.3f (reported only)" % (name, value, baseline))
            continue
        required = baseline * required_ratios[name]
        print("%s=%.3f baseline=%.3f required>%.3f (%.0f%% baseline)" % (
            name, value, baseline, required, required_ratios[name] * 100))
        if value <= required:
            failures.append("%s <= %.3f" % (name, required))
    if failures and not args.no_gate:
        raise SystemExit("Metric gate failed: " + ", ".join(failures))
    print("Metric gate passed" if not failures else "Metrics reported; deployment gate not applied")


if __name__ == "__main__":
    main()
