#!/usr/bin/env python3
"""从原始 Jetson RGB runtime 导出 ImageBackbone ONNX，保持权重和输出布局不变。"""
import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, help="Path to jetson_rgb_runtime")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="vit_small_patch8_224_dino")
    parser.add_argument("--output", required=True)
    parser.add_argument("--img-size", type=int, default=160)
    args = parser.parse_args()
    # 直接导入源 runtime 的 ImageBackbone，避免重新定义模型结构造成导出偏差。
    sys.path.insert(0, str(Path(args.runtime_root).resolve()))
    from image_model import ImageBackbone

    model = ImageBackbone(args.backbone, args.checkpoint, img_size=args.img_size).eval()
    # 固定 1x3ximg_size x img_size 的静态输入，方便 TX2 上构建静态 TensorRT engine。
    sample = torch.zeros(1, 3, args.img_size, args.img_size)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, sample, args.output, opset_version=13,
        input_names=["image"], output_names=["patch_features"],
        # 常量折叠减少推理图中的静态计算节点。
        do_constant_folding=True,
    )
    print("Exported", args.output)


if __name__ == "__main__":
    main()
