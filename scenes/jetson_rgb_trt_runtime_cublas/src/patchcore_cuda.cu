#include "patchcore_cuda.h"

#include <cfloat>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace {
void cuda_check(cudaError_t status, const char* action) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

void cublas_check(cublasStatus_t status, const char* action) {
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(action) + ": cuBLAS status " + std::to_string(status));
}

__global__ void prepare_bank_kernel(const __half* bank_row_major, int channels, int bank_rows,
                                    __half* bank_col_major, float* bank_norms) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= bank_rows) return;

  float sum = 0.0f;
  for (int channel = tid; channel < channels; channel += blockDim.x) {
    const float value = __half2float(bank_row_major[row * channels + channel]);
    bank_col_major[row + channel * bank_rows] = __float2half(value);
    sum += value * value;
  }

  extern __shared__ float partial[];
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }
  if (tid == 0) bank_norms[row] = partial[0];
}

__global__ void prepare_features_kernel(const __half* features, int channels, int patches,
                                        float mean, float stddev,
                                        __half* feature_col_major, float* feature_norms) {
  const int patch = blockIdx.x;
  const int tid = threadIdx.x;
  if (patch >= patches) return;

  float sum = 0.0f;
  for (int channel = tid; channel < channels; channel += blockDim.x) {
    const float value = (__half2float(features[channel * patches + patch]) - mean) / stddev;
    feature_col_major[channel + patch * channels] = __float2half(value);
    sum += value * value;
  }

  extern __shared__ float partial[];
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }
  if (tid == 0) feature_norms[patch] = partial[0];
}

__global__ void reduce_distances_kernel(const float* dots, const float* bank_norms,
                                        const float* feature_norms, int bank_rows, int patches,
                                        float* output_distances, int* output_indices) {
  const int patch = blockIdx.x;
  const int tid = threadIdx.x;
  if (patch >= patches) return;

  float best = FLT_MAX;
  int best_index = 0;
  const float feature_norm = feature_norms[patch];
  for (int row = tid; row < bank_rows; row += blockDim.x) {
    float squared = bank_norms[row] + feature_norm - 2.0f * dots[row + patch * bank_rows];
    squared = fmaxf(squared, 0.0f);
    if (squared < best) {
      best = squared;
      best_index = row;
    }
  }

  extern __shared__ unsigned char scratch[];
  float* distances = reinterpret_cast<float*>(scratch);
  int* indices = reinterpret_cast<int*>(distances + blockDim.x);
  distances[tid] = best;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride && distances[tid + stride] < distances[tid]) {
      distances[tid] = distances[tid + stride];
      indices[tid] = indices[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output_distances[patch] = sqrtf(distances[0]);
    output_indices[patch] = indices[0];
  }
}
}  // namespace

struct PatchDistanceWorkspace {
  int channels;
  int patches;
  int bank_rows;
  cublasHandle_t handle = nullptr;
  __half* bank_col_major = nullptr;
  __half* feature_col_major = nullptr;
  float* bank_norms = nullptr;
  float* feature_norms = nullptr;
  float* dots = nullptr;
};

PatchDistanceWorkspace* create_patch_distance_workspace(int channels, int patches, int bank_rows,
                                                         const __half* bank) {
  constexpr int threads = 256;
  PatchDistanceWorkspace* workspace = new PatchDistanceWorkspace();
  workspace->channels = channels;
  workspace->patches = patches;
  workspace->bank_rows = bank_rows;
  cublas_check(cublasCreate(&workspace->handle), "Create cuBLAS handle");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&workspace->bank_col_major),
                        static_cast<size_t>(bank_rows) * channels * sizeof(__half)), "Allocate cuBLAS bank");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&workspace->feature_col_major),
                        static_cast<size_t>(channels) * patches * sizeof(__half)), "Allocate cuBLAS features");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&workspace->bank_norms),
                        static_cast<size_t>(bank_rows) * sizeof(float)), "Allocate bank norms");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&workspace->feature_norms),
                        static_cast<size_t>(patches) * sizeof(float)), "Allocate feature norms");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&workspace->dots),
                        static_cast<size_t>(bank_rows) * patches * sizeof(float)), "Allocate cuBLAS dot matrix");

  prepare_bank_kernel<<<bank_rows, threads, threads * sizeof(float)>>>(bank, channels, bank_rows,
                                                                       workspace->bank_col_major,
                                                                       workspace->bank_norms);
  cuda_check(cudaGetLastError(), "Prepare cuBLAS memory bank");
  cuda_check(cudaDeviceSynchronize(), "Synchronize cuBLAS memory bank preparation");
  return workspace;
}

void destroy_patch_distance_workspace(PatchDistanceWorkspace* workspace) {
  if (!workspace) return;
  if (workspace->bank_col_major) cudaFree(workspace->bank_col_major);
  if (workspace->feature_col_major) cudaFree(workspace->feature_col_major);
  if (workspace->bank_norms) cudaFree(workspace->bank_norms);
  if (workspace->feature_norms) cudaFree(workspace->feature_norms);
  if (workspace->dots) cudaFree(workspace->dots);
  if (workspace->handle) cublasDestroy(workspace->handle);
  delete workspace;
}

void patch_min_distances(const __half* features, int channels, int patches,
                         const __half* bank, int bank_rows, float mean, float stddev,
                         float* min_distances, int* min_indices,
                         PatchDistanceWorkspace* workspace) {
  if (!workspace || workspace->channels != channels || workspace->patches != patches || workspace->bank_rows != bank_rows)
    throw std::runtime_error("Patch distance workspace shape mismatch");

  constexpr int threads = 256;
  prepare_features_kernel<<<patches, threads, threads * sizeof(float)>>>(features, channels, patches, mean, stddev,
                                                                         workspace->feature_col_major,
                                                                         workspace->feature_norms);
  cuda_check(cudaGetLastError(), "Prepare cuBLAS features");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublas_check(cublasGemmEx(workspace->handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            bank_rows, patches, channels,
                            &alpha,
                            workspace->bank_col_major, CUDA_R_16F, bank_rows,
                            workspace->feature_col_major, CUDA_R_16F, channels,
                            &beta,
                            workspace->dots, CUDA_R_32F, bank_rows,
                            CUDA_R_32F, CUBLAS_GEMM_DEFAULT),
               "Compute PatchCore dot products with cuBLAS");

  const size_t shared = threads * (sizeof(float) + sizeof(int));
  reduce_distances_kernel<<<patches, threads, shared>>>(workspace->dots, workspace->bank_norms,
                                                        workspace->feature_norms, bank_rows, patches,
                                                        min_distances, min_indices);
  cuda_check(cudaGetLastError(), "Reduce PatchCore minimum distances");
}
