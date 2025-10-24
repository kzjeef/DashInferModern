/*!
 * Copyright (c) 2026 Segno System.
 * @file    fp8_blockwise_activation.h
 */

#pragma once

#include <cuda_runtime.h>

namespace allspark {
namespace cuda {

#ifdef ENABLE_FP16
cudaError_t QuantizeFP8BlockwiseFP16(
    const void* input, void* output, float* scale, int row_count, int k_dim,
    int block_k, cudaStream_t stream);
#endif

#ifdef ENABLE_BF16
cudaError_t QuantizeFP8BlockwiseBF16(
    const void* input, void* output, float* scale, int row_count, int k_dim,
    int block_k, cudaStream_t stream);
#endif

}  // namespace cuda
}  // namespace allspark
