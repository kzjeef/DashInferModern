/*!
 * Copyright (c) 2026 Segno System.
 * @file    fp8_blockwise_activation.cu
 */

#include "fp8_blockwise_activation.h"

#if defined(ENABLE_FP8) && (defined(ENABLE_FP16) || defined(ENABLE_BF16))

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <limits>

#ifdef ENABLE_BF16
#include <common/hie_bfloat16.hpp>
#endif

namespace allspark {
namespace cuda {
namespace {

constexpr int kThreadCount = 256;
constexpr float kFP8E4M3Maximum = 448.0f;

template <typename InputType>
__device__ __forceinline__ float InputToFloat(InputType value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float InputToFloat<half>(half value) {
  return __half2float(value);
}

template <typename InputType>
__global__ __launch_bounds__(kThreadCount) void QuantizeFP8BlockwiseKernel(
    const InputType* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output, float* __restrict__ scale,
    int row_count, int k_dim, int block_k, int k_group_count) {
  const int group_index = static_cast<int>(blockIdx.x);
  const int row = group_index / k_group_count;
  const int k_group = group_index % k_group_count;
  if (row >= row_count) {
    return;
  }

  const int k_begin = k_group * block_k;
  const int k_end = min(k_begin + block_k, k_dim);
  const int64_t row_offset = static_cast<int64_t>(row) * k_dim;
  float local_max = 0.0f;
  for (int k_index = k_begin + threadIdx.x; k_index < k_end;
       k_index += blockDim.x) {
    local_max = fmaxf(
        local_max, fabsf(InputToFloat(input[row_offset + k_index])));
  }

  __shared__ float block_max[kThreadCount];
  block_max[threadIdx.x] = local_max;
  __syncthreads();
  for (int offset = kThreadCount / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      block_max[threadIdx.x] =
          fmaxf(block_max[threadIdx.x], block_max[threadIdx.x + offset]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scale[group_index] = block_max[0] == 0.0f
                             ? 1.0f
                             : block_max[0] / kFP8E4M3Maximum;
  }
  __syncthreads();
  const float inverse_scale = 1.0f / scale[group_index];
  for (int k_index = k_begin + threadIdx.x; k_index < k_end;
       k_index += blockDim.x) {
    const float value = InputToFloat(input[row_offset + k_index]);
    output[row_offset + k_index] = __nv_fp8_e4m3(value * inverse_scale);
  }
}

template <typename InputType>
cudaError_t LaunchQuantizeFP8Blockwise(
    const void* input, void* output, float* scale, int row_count, int k_dim,
    int block_k, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scale == nullptr ||
      row_count <= 0 || k_dim <= 0 || block_k <= 0) {
    return cudaErrorInvalidValue;
  }
  const int k_group_count = (k_dim + block_k - 1) / block_k;
  const int64_t group_count =
      static_cast<int64_t>(row_count) * k_group_count;
  if (group_count > std::numeric_limits<unsigned int>::max()) {
    return cudaErrorInvalidConfiguration;
  }
  QuantizeFP8BlockwiseKernel<<<static_cast<unsigned int>(group_count),
                               kThreadCount, 0, stream>>>(
      static_cast<const InputType*>(input),
      static_cast<__nv_fp8_e4m3*>(output), scale, row_count, k_dim, block_k,
      k_group_count);
  return cudaGetLastError();
}

}  // namespace

#ifdef ENABLE_FP16
cudaError_t QuantizeFP8BlockwiseFP16(
    const void* input, void* output, float* scale, int row_count, int k_dim,
    int block_k, cudaStream_t stream) {
  return LaunchQuantizeFP8Blockwise<half>(
      input, output, scale, row_count, k_dim, block_k, stream);
}
#endif

#ifdef ENABLE_BF16
cudaError_t QuantizeFP8BlockwiseBF16(
    const void* input, void* output, float* scale, int row_count, int k_dim,
    int block_k, cudaStream_t stream) {
  return LaunchQuantizeFP8Blockwise<hie::bfloat16>(
      input, output, scale, row_count, k_dim, block_k, stream);
}
#endif

}  // namespace cuda
}  // namespace allspark

#endif  // ENABLE_FP8 && (ENABLE_FP16 || ENABLE_BF16)
