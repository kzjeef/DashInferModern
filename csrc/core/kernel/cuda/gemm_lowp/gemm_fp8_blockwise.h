/*!
 * Copyright (c) 2026 Segno System.
 * @file    gemm_fp8_blockwise.h
 */

#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace allspark {
namespace cuda {

enum class FP8BlockwiseActivationType : int {
  FP16 = 0,
  BF16 = 1,
};

void* FP8BlockwiseGemmCreate();
void FP8BlockwiseGemmDestroy(void* state);

size_t FP8BlockwiseGemmWorkspaceSize(int m_dim, int n_dim, int k_dim,
                                     int block_k);

int FP8BlockwiseGemmPrepare(void* state, cublasLtHandle_t handle,
                            void* workspace, size_t workspace_size,
                            int m_dim, int n_dim, int k_dim,
                            int block_n, int block_k);

int FP8BlockwiseGemmRun(void* state, cublasLtHandle_t handle,
                        const void* activation, const void* weight,
                        const float* weight_scale, const void* bias,
                        void* output,
                        FP8BlockwiseActivationType activation_type,
                        cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
