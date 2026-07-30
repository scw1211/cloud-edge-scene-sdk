#pragma once

#include <cuda_fp16.h>

// PatchCore 最近邻距离计算的工作区句柄。
// 头文件只暴露不完整类型，调用方不需要知道内部字段，避免把实现细节泄漏到 main.cpp。
struct PatchDistanceWorkspace;

// 创建/销毁距离计算工作区：当前实现主要保存形状信息，用于运行时校验。
PatchDistanceWorkspace* create_patch_distance_workspace(int channels, int patches, int bank_rows,
                                                         const __half* bank);
void destroy_patch_distance_workspace(PatchDistanceWorkspace* workspace);

// 对每个 patch 特征，在 FP16 记忆库中查找最近邻。
// features: TensorRT 输出的 [channels, patches] FP16 特征。
// bank: 训练阶段保存的正常样本 patch 记忆库。
// min_distances/min_indices: 每个 patch 对应的最近距离和最近记忆库行号。
void patch_min_distances(const __half* features, int channels, int patches,
                         const __half* bank, int bank_rows, float mean, float stddev,
                         float* min_distances, int* min_indices,
                         PatchDistanceWorkspace* workspace);
