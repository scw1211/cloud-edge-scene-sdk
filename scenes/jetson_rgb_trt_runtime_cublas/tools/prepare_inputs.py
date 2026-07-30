#!/usr/bin/env python3
"""把 RGB PNG 预处理成 NCHW float32 张量，供低延迟 TensorRT 路径直接读取。"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prepare_one(source, destination, image_size):
    # 对齐 rgb_module/jetson_rgb_runtime 的 torchvision/PIL 路径：
    # RGB 转换、PIL bicubic resize、ToTensor，再做 ImageNet mean/std 归一化。
    with Image.open(str(source)) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32)
    tensor = ((pixels / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 直接写裸 float32 字节，C++ read_tensor 按固定元素数校验并读取。
    tensor.astype(np.float32, copy=False).tofile(str(destination))


def main():
    parser = argparse.ArgumentParser(description="Convert PNG images to normalized NCHW float32 tensors.")
    parser.add_argument("--input-dir", required=True, help="RGB/test directory containing defect subdirectories")
    parser.add_argument("--output-dir", required=True, help="Destination directory for .nchw.f32 tensors")
    parser.add_argument("--img-size", type=int, default=160)
    args = parser.parse_args()
    source_root, destination_root = Path(args.input_dir), Path(args.output_dir)
    # 保留原始 defect 子目录相对路径，便于 C++ 从预处理张量反推出原 PNG 路径。
    images = sorted(source_root.rglob("*.png"))
    if not images:
        raise SystemExit("No PNG files found under %s" % source_root)
    for source in images:
        relative = source.relative_to(source_root)
        prepare_one(source, destination_root / relative.with_suffix(".nchw.f32"), args.img_size)
    print("Prepared %d tensors in %s" % (len(images), destination_root))


if __name__ == "__main__":
    main()
