/*!
 * Copyright (c) 2026 Segno System.
 * @file    nvfp4_scale_repack.cuh
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace allspark {
namespace cuda {

// CUTLASS Sm1xxBlkScaledConfig groups 128 rows by four consecutive K blocks.
__host__ __device__ inline int64_t NvFP4ScaleInterleavedOffset(
    int row, int k_block, int k_block_count) {
  const int k_tile_count = (k_block_count + 3) / 4;
  return static_cast<int64_t>(row % 32) * 16 +
         static_cast<int64_t>((row / 32) % 4) * 4 +
         static_cast<int64_t>(k_block % 4) +
         static_cast<int64_t>(k_block / 4) * 512 +
         static_cast<int64_t>(row / 128) * k_tile_count * 512;
}

__global__ void NvFP4RepackScaleKernel(
    const uint8_t* __restrict__ source,
    uint8_t* __restrict__ destination,
    int row_count, int k_block_count) {
  const int64_t linear_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t element_count =
      static_cast<int64_t>(row_count) * k_block_count;
  if (linear_index >= element_count) {
    return;
  }

  const int row = linear_index / k_block_count;
  const int k_block = linear_index % k_block_count;
  const int64_t destination_index = NvFP4ScaleInterleavedOffset(
      row, k_block, k_block_count);
  destination[destination_index] = source[linear_index];
}

inline void NvFP4RepackScales(const void* source, void* destination,
                              int row_count, int k_block_count,
                              cudaStream_t stream) {
  const int64_t element_count =
      static_cast<int64_t>(row_count) * k_block_count;
  constexpr int kThreadCount = 256;
  const int block_count =
      static_cast<int>((element_count + kThreadCount - 1) / kThreadCount);
  NvFP4RepackScaleKernel<<<block_count, kThreadCount, 0, stream>>>(
      static_cast<const uint8_t*>(source),
      static_cast<uint8_t*>(destination), row_count, k_block_count);
}

}  // namespace cuda
}  // namespace allspark
