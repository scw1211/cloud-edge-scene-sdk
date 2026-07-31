#!/usr/bin/env python3
"""对比 TensorRT 特征导出和 PyTorch backbone，确认 ONNX/TRT 转换没有改变特征语义。"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-tensor", required=True, help="Prepared .nchw.f32 input")
    parser.add_argument("--trt-features", required=True, help="trt_features_0.f16 emitted by the C++ runtime")
    parser.add_argument("--memory-bank", required=True, help="Original PyTorch .pt memory bank")
    parser.add_argument("--trt-distances", help="Optional trt_distances_0.f32 emitted by the C++ runtime")
    parser.add_argument("--img-size", type=int, default=160)
    args = parser.parse_args()
    # 复用原始 runtime 的 image_model.py，保证 PyTorch 参考模型和部署模型来自同一套实现。
    sys.path.insert(0, str(Path(args.runtime_root).resolve()))
    from image_model import ImageBackbone

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImageBackbone("vit_small_patch8_224_dino", args.checkpoint, img_size=args.img_size).to(device).half().eval()
    # 输入是 prepare_inputs.py 生成的 NCHW float32；PyTorch 侧转成 FP16 以匹配 TensorRT engine。
    input_array = np.fromfile(args.input_tensor, dtype=np.float32).reshape(1, 3, args.img_size, args.img_size)
    with torch.no_grad():
        expected = model(torch.from_numpy(input_array).to(device).half()).cpu().numpy().astype(np.float16)
    actual = np.fromfile(args.trt_features, dtype=np.float16).reshape(expected.shape)
    delta = actual.astype(np.float32) - expected.astype(np.float32)
    cosine = float(np.dot(actual.reshape(-1).astype(np.float32), expected.reshape(-1).astype(np.float32)) /
                   (np.linalg.norm(actual.reshape(-1).astype(np.float32)) * np.linalg.norm(expected.reshape(-1).astype(np.float32))))
    print("shape=%s" % (expected.shape,))
    print("max_abs_error=%.8f" % np.abs(delta).max())
    print("mean_abs_error=%.8f" % np.abs(delta).mean())
    print("cosine_similarity=%.8f" % cosine)
    # 继续用同一份记忆库计算 PatchCore 分数，区分“特征误差”和“CUDA 距离误差”。
    bank_data = torch.load(args.memory_bank, map_location="cpu")
    bank = bank_data["patch_lib"].to(device=device, dtype=torch.float16)
    mean = torch.as_tensor(bank_data["mean"], device=device, dtype=torch.float16)
    stddev = torch.as_tensor(bank_data["std"], device=device, dtype=torch.float16)

    def patchcore_score(feature_map, return_min_values=False):
        # 这里复刻 Python PatchCore 的 cdist 最近邻和 reweight 逻辑，作为 C++/CUDA 后处理的参照。
        patch = torch.from_numpy(feature_map).to(device).reshape(feature_map.shape[1], -1).T
        patch = ((patch - mean) / stddev).half()
        distances = torch.cdist(patch, bank)
        min_values, min_indices = torch.min(distances, dim=1)
        patch_index = torch.argmax(min_values)
        s_star = min_values[patch_index] / 1000
        m_test = patch[patch_index].unsqueeze(0)
        m_star = bank[min_indices[patch_index]].unsqueeze(0)
        _, nn_indices = torch.topk(torch.cdist(m_star, bank), k=3, largest=False)
        neighbour_distances = torch.linalg.norm(m_test - bank[nn_indices[0, 1:]], dim=1) / 1000
        dimension = torch.sqrt(torch.tensor(float(patch.shape[1]), device=device))
        weight = 1 - torch.exp(s_star / dimension) / torch.sum(torch.exp(neighbour_distances / dimension))
        score = float((weight * s_star).float().cpu())
        max_distance = float(min_values.max().float().cpu())
        return (score, max_distance, min_values.float().cpu().numpy()) if return_min_values else (score, max_distance)

    pytorch_score, pytorch_max_distance, pytorch_distances = patchcore_score(expected, return_min_values=True)
    trt_score, trt_max_distance = patchcore_score(actual)
    print("pytorch_feature_score=%.8f max_patch_distance=%.6f" % (pytorch_score, pytorch_max_distance))
    print("trt_feature_torch_cdist_score=%.8f max_patch_distance=%.6f" % (trt_score, trt_max_distance))
    if args.trt_distances:
        # 如果 C++ 设置 DUMP_TRT_DISTANCES，可直接比较 CUDA kernel 输出的每 patch 最近距离。
        trt_distances = np.fromfile(args.trt_distances, dtype=np.float32)
        if trt_distances.shape != pytorch_distances.shape:
            raise RuntimeError("TRT distance shape mismatch: %s != %s" % (trt_distances.shape, pytorch_distances.shape))
        distance_delta = trt_distances - pytorch_distances
        print("trt_distance_max_abs_error=%.8f" % np.abs(distance_delta).max())
        print("trt_distance_mean_abs_error=%.8f" % np.abs(distance_delta).mean())
        print("trt_distance_max=%.6f torch_cdist_max=%.6f" % (trt_distances.max(), pytorch_distances.max()))


if __name__ == "__main__":
    main()
