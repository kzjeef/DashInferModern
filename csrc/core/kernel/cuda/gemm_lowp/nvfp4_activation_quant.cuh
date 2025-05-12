/*!
 * Copyright (c) 2026 Segno System.
 * @file    nvfp4_activation_quant.cuh
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "nvfp4_scale_repack.cuh"

namespace allspark {
namespace cuda {

constexpr int kNvFP4BlockSize = 16;

__device__ __forceinline__ uint32_t NvFP4ConvertEight(const float* values) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t result;
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(result)
      : "f"(values[0]), "f"(values[1]), "f"(values[2]),
        "f"(values[3]), "f"(values[4]), "f"(values[5]),
        "f"(values[6]), "f"(values[7]));
  return result;
#else
  asm volatile("trap;\n");
  return 0;
#endif
}

__device__ __forceinline__ float NvFP4Reciprocal(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n"
               : "=f"(result) : "f"(value));
  return result;
}

template <typename InputType>
__global__ __launch_bounds__(256, 8) void NvFP4QuantizeActivationKernel(
    const InputType* __restrict__ input,
    uint8_t* __restrict__ packed_output,
    uint8_t* __restrict__ scale_output,
    int row_count, int k_dim, float scale_multiplier) {
  const int block_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int k_block_count = k_dim / kNvFP4BlockSize;
  const int block_count = row_count * k_block_count;
  if (block_index >= block_count) {
    return;
  }

  const int row = block_index / k_block_count;
  const int k_block = block_index % k_block_count;
  const int k_start = k_block * kNvFP4BlockSize;
  const InputType* row_input = input +
      static_cast<int64_t>(row) * k_dim + k_start;

  float values[kNvFP4BlockSize];
  float absolute_max = 0.0f;
#pragma unroll
  for (int index = 0; index < kNvFP4BlockSize; ++index) {
    values[index] = static_cast<float>(row_input[index]);
    absolute_max = fmaxf(absolute_max, fabsf(values[index]));
  }

  float block_scale =
      scale_multiplier * absolute_max * NvFP4Reciprocal(6.0f);
  if (block_scale < 1.0e-12f) {
    block_scale = 1.0f;
  }
  __nv_fp8_e4m3 fp8_scale(block_scale);
  const float rounded_scale = static_cast<float>(fp8_scale);
  scale_output[NvFP4ScaleInterleavedOffset(
      row, k_block, k_block_count)] =
      *reinterpret_cast<const uint8_t*>(&fp8_scale);

  const float output_multiplier =
      rounded_scale == 0.0f
          ? 0.0f
          : scale_multiplier * NvFP4Reciprocal(rounded_scale);
#pragma unroll
  for (int index = 0; index < kNvFP4BlockSize; ++index) {
    values[index] *= output_multiplier;
  }

  const uint32_t first = NvFP4ConvertEight(values);
  const uint32_t second = NvFP4ConvertEight(values + 8);
  const int64_t byte_offset =
      (static_cast<int64_t>(row) * k_dim + k_start) / 2;
  reinterpret_cast<uint32_t*>(packed_output + byte_offset)[0] = first;
  reinterpret_cast<uint32_t*>(packed_output + byte_offset)[1] = second;
}

template <typename InputType>
inline void NvFP4QuantizeActivation(const void* input, void* packed_output,
                                    void* scale_output, int row_count,
                                    int k_dim, float scale_multiplier,
                                    cudaStream_t stream) {
  const int block_count = row_count * (k_dim / kNvFP4BlockSize);
  constexpr int kThreadCount = 256;
  const int grid_size =
      (block_count + kThreadCount - 1) / kThreadCount;
  NvFP4QuantizeActivationKernel<InputType>
      <<<grid_size, kThreadCount, 0, stream>>>(
          static_cast<const InputType*>(input),
          static_cast<uint8_t*>(packed_output),
          static_cast<uint8_t*>(scale_output), row_count, k_dim,
          scale_multiplier);
}

}  // namespace cuda
}  // namespace allspark
