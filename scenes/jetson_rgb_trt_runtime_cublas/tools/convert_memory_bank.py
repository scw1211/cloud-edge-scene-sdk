#!/usr/bin/env python3
"""把 PyTorch PatchCore 记忆库转换为 C++ 运行时可直接读取的紧凑 FP16 .pcbank 文件。"""
import argparse
import struct
from pathlib import Path

import torch

MAGIC = b"PCBNK01\0"
# 文件头布局必须与 src/main.cpp 的 BankHeader 保持一致：magic/version/rows/cols/mean/std。
HEADER = struct.Struct("<8sIIIff")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bank = torch.load(args.input, map_location="cpu")
    # patch_lib 是正常样本 patch 特征集合；部署时以 FP16 存储，降低磁盘和 GPU 显存占用。
    patches = bank["patch_lib"].contiguous().to(dtype=torch.float16)
    if patches.ndim != 2:
        raise RuntimeError("patch_lib must be a [rows, channels] tensor")
    mean = float(torch.as_tensor(bank["mean"]).item())
    stddev = float(torch.as_tensor(bank["std"]).item())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        # 先写固定长度头部，再顺序写入 FP16 tensor 原始字节，C++ 端按相同布局 memcpy 解析。
        handle.write(HEADER.pack(MAGIC, 1, patches.shape[0], patches.shape[1], mean, stddev))
        handle.write(patches.numpy().tobytes())
    print("Converted", args.input, "->", args.output, "rows=", patches.shape[0], "cols=", patches.shape[1])


if __name__ == "__main__":
    main()
