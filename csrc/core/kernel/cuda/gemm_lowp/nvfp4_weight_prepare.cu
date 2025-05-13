/*!
 * Copyright (c) 2026 Segno System.
 * @file    nvfp4_weight_prepare.cu
 */

#include "gemm_nvfp4_blockwise.h"

#include "nvfp4_scale_repack.cuh"

namespace allspark {
namespace cuda {

cudaError_t NvFP4PrepareWeight(const void* packed_weight,
                               const void* block_scale,
                               void* prepared_weight,
                               void* prepared_scale,
                               int n_dim, int k_dim,
                               cudaStream_t stream) {
  if (packed_weight == nullptr || block_scale == nullptr ||
      prepared_weight == nullptr || prepared_scale == nullptr ||
      n_dim <= 0 || k_dim <= 0 || n_dim % 128 != 0 || k_dim % 64 != 0) {
    return cudaErrorInvalidValue;
  }

  const size_t packed_bytes =
      static_cast<size_t>(n_dim) * k_dim / 2;
  cudaError_t status = cudaMemcpyAsync(
      prepared_weight, packed_weight, packed_bytes,
      cudaMemcpyDeviceToDevice, stream);
  if (status != cudaSuccess) {
    return status;
  }

  NvFP4RepackScales(block_scale, prepared_scale, n_dim, k_dim / 16, stream);
  return cudaGetLastError();
}

}  // namespace cuda
}  // namespace allspark
