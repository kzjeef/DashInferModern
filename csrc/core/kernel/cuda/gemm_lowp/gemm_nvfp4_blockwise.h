/*!
 * Copyright (c) 2026 Segno System.
 * @file    gemm_nvfp4_blockwise.h
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace allspark {
namespace cuda {

cudaError_t NvFP4PrepareWeight(const void* packed_weight,
                               const void* block_scale,
                               void* prepared_weight,
                               void* prepared_scale,
                               int n_dim, int k_dim,
                               cudaStream_t stream);

void* NvFP4GemmCreate();
void NvFP4GemmDestroy(void* state);

size_t NvFP4GemmWorkspaceSize(int m_dim, int n_dim, int k_dim);

int NvFP4GemmPrepare(void* state, const void* prepared_weight,
                     const void* prepared_scale, float weight_global_scale,
                     float activation_scale_multiplier, void* output,
                     int m_dim, int n_dim, int k_dim,
                     void* workspace, size_t workspace_size,
                     cudaStream_t stream);

int NvFP4GemmRun(void* state, const void* activation, void* output,
                 int m_dim, int n_dim, int k_dim,
                 void* workspace, size_t workspace_size,
                 cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
