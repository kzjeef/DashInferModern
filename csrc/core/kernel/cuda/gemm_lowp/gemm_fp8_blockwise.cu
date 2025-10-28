/*!
 * Copyright (c) 2026 Segno System.
 * @file    gemm_fp8_blockwise.cu
 */

#include "gemm_fp8_blockwise.h"

#ifdef ENABLE_FP8

#include "fp8_blockwise_activation.h"

#include <cuda_fp16.h>

#include <cstdint>
#include <initializer_list>
#include <limits>

#ifdef ENABLE_BF16
#include <common/hie_bfloat16.hpp>
#endif

namespace {

constexpr size_t kCublasWorkspaceBytes = 8 * 1024 * 1024;
constexpr int kThreadCount = 256;

size_t Align256(size_t value) { return (value + 255) & ~size_t{255}; }

struct WorkspaceLayout {
  size_t activation_offset = 0;
  size_t activation_scale_offset = 0;
  size_t partial_offset = 0;
  size_t accumulator_offset = 0;
  size_t cublas_offset = 0;
  size_t total_bytes = 0;
};

WorkspaceLayout MakeWorkspaceLayout(int m_dim, int n_dim, int k_dim,
                                    int block_k) {
  const int padded_m = ((m_dim + 15) / 16) * 16;
  const int k_group_count = (k_dim + block_k - 1) / block_k;
  WorkspaceLayout layout{};
  size_t cursor = 0;
  layout.activation_offset = cursor;
  cursor += Align256(static_cast<size_t>(padded_m) * k_dim);
  layout.activation_scale_offset = cursor;
  cursor += Align256(static_cast<size_t>(m_dim) * k_group_count *
                     sizeof(float));
  cursor += Align256(sizeof(float));
  layout.partial_offset = cursor;
  cursor += Align256(static_cast<size_t>(padded_m) * n_dim * sizeof(float));
  layout.accumulator_offset = cursor;
  cursor += Align256(static_cast<size_t>(m_dim) * n_dim * sizeof(float));
  layout.cublas_offset = cursor;
  cursor += Align256(kCublasWorkspaceBytes);
  layout.total_bytes = cursor;
  return layout;
}

struct FP8BlockwiseGemmState {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t activation_layout = nullptr;
  cublasLtMatrixLayout_t weight_layout = nullptr;
  cublasLtMatrixLayout_t partial_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  uint8_t* quantized_activation = nullptr;
  float* activation_scale = nullptr;
  float* partial = nullptr;
  float* accumulator = nullptr;
  float* unit_scale = nullptr;
  void* cublas_workspace = nullptr;
  size_t cublas_workspace_size = 0;
  void* workspace = nullptr;
  size_t workspace_size = 0;
  int m_dim = 0;
  int padded_m = 0;
  int n_dim = 0;
  int k_dim = 0;
  int block_n = 0;
  int block_k = 0;
  int k_group_count = 0;
  bool ready = false;
};

__global__ void SetUnitScaleKernel(float* unit_scale) {
  if (threadIdx.x == 0) {
    *unit_scale = 1.0f;
  }
}

void DestroyDescriptors(FP8BlockwiseGemmState* state) {
  if (state->preference != nullptr) {
    cublasLtMatmulPreferenceDestroy(state->preference);
    state->preference = nullptr;
  }
  if (state->partial_layout != nullptr) {
    cublasLtMatrixLayoutDestroy(state->partial_layout);
    state->partial_layout = nullptr;
  }
  if (state->weight_layout != nullptr) {
    cublasLtMatrixLayoutDestroy(state->weight_layout);
    state->weight_layout = nullptr;
  }
  if (state->activation_layout != nullptr) {
    cublasLtMatrixLayoutDestroy(state->activation_layout);
    state->activation_layout = nullptr;
  }
  if (state->operation != nullptr) {
    cublasLtMatmulDescDestroy(state->operation);
    state->operation = nullptr;
  }
  state->ready = false;
}

template <typename OutputType>
__global__ void AccumulateFP8BlockKernel(
    const float* __restrict__ partial,
    const float* __restrict__ activation_scale,
    const float* __restrict__ weight_scale,
    float* __restrict__ accumulator, int element_count, int n_dim,
    int k_group_count, int k_group, int block_n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const int row = index / n_dim;
  const int column = index % n_dim;
  const float combined_scale =
      activation_scale[row * k_group_count + k_group] *
      weight_scale[(column / block_n) * k_group_count + k_group];
  const float value = partial[index] * combined_scale;
  accumulator[index] = k_group == 0 ? value : accumulator[index] + value;
}

template <typename OutputType>
__device__ __forceinline__ float OutputToFloat(OutputType value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float OutputToFloat<half>(half value) {
  return __half2float(value);
}

template <typename OutputType>
__global__ void FinalizeFP8BlockKernel(
    const float* __restrict__ accumulator,
    const OutputType* __restrict__ bias,
    OutputType* __restrict__ output, int element_count, int n_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  float value = accumulator[index];
  if (bias != nullptr) {
    value += OutputToFloat(bias[index % n_dim]);
  }
  output[index] = static_cast<OutputType>(value);
}

template <typename OutputType>
cudaError_t LaunchAccumulateAndFinalize(
    FP8BlockwiseGemmState* state, const float* weight_scale,
    const void* bias, void* output, int k_group, cudaStream_t stream) {
  const int64_t element_count_64 =
      static_cast<int64_t>(state->m_dim) * state->n_dim;
  if (element_count_64 > std::numeric_limits<int>::max()) {
    return cudaErrorInvalidConfiguration;
  }
  const int element_count = static_cast<int>(element_count_64);
  const int grid_size = (element_count + kThreadCount - 1) / kThreadCount;
  AccumulateFP8BlockKernel<OutputType><<<grid_size, kThreadCount, 0, stream>>>(
      state->partial, state->activation_scale, weight_scale,
      state->accumulator, element_count, state->n_dim, state->k_group_count,
      k_group, state->block_n);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess || k_group + 1 != state->k_group_count) {
    return status;
  }
  FinalizeFP8BlockKernel<OutputType><<<grid_size, kThreadCount, 0, stream>>>(
      state->accumulator, static_cast<const OutputType*>(bias),
      static_cast<OutputType*>(output), element_count, state->n_dim);
  return cudaGetLastError();
}

}  // namespace

namespace allspark {
namespace cuda {

void* FP8BlockwiseGemmCreate() { return new FP8BlockwiseGemmState(); }

void FP8BlockwiseGemmDestroy(void* state) {
  auto* gemm_state = static_cast<FP8BlockwiseGemmState*>(state);
  if (gemm_state != nullptr) {
    DestroyDescriptors(gemm_state);
    delete gemm_state;
  }
}

size_t FP8BlockwiseGemmWorkspaceSize(int m_dim, int n_dim, int k_dim,
                                     int block_k) {
  if (m_dim <= 0 || n_dim <= 0 || k_dim <= 0 || block_k <= 0) {
    return 0;
  }
  return MakeWorkspaceLayout(m_dim, n_dim, k_dim, block_k).total_bytes;
}

int FP8BlockwiseGemmPrepare(void* state, cublasLtHandle_t handle,
                            void* workspace, size_t workspace_size,
                            int m_dim, int n_dim, int k_dim,
                            int block_n, int block_k) {
  auto* gemm_state = static_cast<FP8BlockwiseGemmState*>(state);
  if (gemm_state == nullptr || handle == nullptr || workspace == nullptr ||
      m_dim <= 0 || n_dim <= 0 || k_dim <= 0 || block_n <= 0 ||
      block_k <= 0 || n_dim % 16 != 0 || k_dim % 16 != 0 ||
      block_k % 16 != 0 || k_dim % block_k != 0) {
    return -1;
  }
  DestroyDescriptors(gemm_state);

  const WorkspaceLayout layout =
      MakeWorkspaceLayout(m_dim, n_dim, k_dim, block_k);
  if (workspace_size < layout.total_bytes) {
    return -2;
  }
  auto* workspace_bytes = static_cast<uint8_t*>(workspace);
  gemm_state->quantized_activation =
      workspace_bytes + layout.activation_offset;
  gemm_state->activation_scale = reinterpret_cast<float*>(
      workspace_bytes + layout.activation_scale_offset);
  gemm_state->unit_scale = reinterpret_cast<float*>(
      workspace_bytes + layout.activation_scale_offset +
      Align256(static_cast<size_t>(m_dim) * (k_dim / block_k) *
               sizeof(float)));
  gemm_state->partial =
      reinterpret_cast<float*>(workspace_bytes + layout.partial_offset);
  gemm_state->accumulator =
      reinterpret_cast<float*>(workspace_bytes + layout.accumulator_offset);
  gemm_state->cublas_workspace = workspace_bytes + layout.cublas_offset;
  gemm_state->cublas_workspace_size = kCublasWorkspaceBytes;
  gemm_state->workspace = workspace;
  gemm_state->workspace_size = workspace_size;
  gemm_state->m_dim = m_dim;
  gemm_state->padded_m = ((m_dim + 15) / 16) * 16;
  gemm_state->n_dim = n_dim;
  gemm_state->k_dim = k_dim;
  gemm_state->block_n = block_n;
  gemm_state->block_k = block_k;
  gemm_state->k_group_count = k_dim / block_k;

  cublasStatus_t status = cublasLtMatmulDescCreate(
      &gemm_state->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t trans_a = CUBLAS_OP_N;
  cublasOperation_t trans_b = CUBLAS_OP_T;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        gemm_state->operation, CUBLASLT_MATMUL_DESC_TRANSA,
        &trans_a, sizeof(trans_a));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        gemm_state->operation, CUBLASLT_MATMUL_DESC_TRANSB,
        &trans_b, sizeof(trans_b));
  }
  const float* unit_scale = gemm_state->unit_scale;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        gemm_state->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &unit_scale, sizeof(unit_scale));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        gemm_state->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &unit_scale, sizeof(unit_scale));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &gemm_state->activation_layout, CUDA_R_8F_E4M3,
        gemm_state->padded_m, block_k, k_dim);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &gemm_state->weight_layout, CUDA_R_8F_E4M3,
        n_dim, block_k, k_dim);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &gemm_state->partial_layout, CUDA_R_32F,
        gemm_state->padded_m, n_dim, n_dim);
  }
  const cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
  for (cublasLtMatrixLayout_t matrix_layout :
       {gemm_state->activation_layout, gemm_state->weight_layout,
        gemm_state->partial_layout}) {
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatrixLayoutSetAttribute(
          matrix_layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
          &row_major, sizeof(row_major));
    }
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceCreate(&gemm_state->preference);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceSetAttribute(
        gemm_state->preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &gemm_state->cublas_workspace_size,
        sizeof(gemm_state->cublas_workspace_size));
  }
  cublasLtMatmulHeuristicResult_t heuristic{};
  int result_count = 0;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoGetHeuristic(
        handle, gemm_state->operation, gemm_state->activation_layout,
        gemm_state->weight_layout, gemm_state->partial_layout,
        gemm_state->partial_layout, gemm_state->preference, 1,
        &heuristic, &result_count);
  }
  if (status != CUBLAS_STATUS_SUCCESS || result_count == 0) {
    DestroyDescriptors(gemm_state);
    return -3;
  }
  gemm_state->algorithm = heuristic.algo;
  gemm_state->ready = true;
  return 0;
}

int FP8BlockwiseGemmRun(void* state, cublasLtHandle_t handle,
                        const void* activation, const void* weight,
                        const float* weight_scale, const void* bias,
                        void* output,
                        FP8BlockwiseActivationType activation_type,
                        cudaStream_t stream) {
  auto* gemm_state = static_cast<FP8BlockwiseGemmState*>(state);
  if (gemm_state == nullptr || !gemm_state->ready || handle == nullptr ||
      activation == nullptr || weight == nullptr || weight_scale == nullptr ||
      output == nullptr) {
    return -1;
  }

  SetUnitScaleKernel<<<1, 1, 0, stream>>>(gemm_state->unit_scale);
  cudaError_t quantize_status = cudaGetLastError();
  if (quantize_status != cudaSuccess) {
    return -3;
  }
  if (gemm_state->padded_m != gemm_state->m_dim) {
    const size_t padding_offset =
        static_cast<size_t>(gemm_state->m_dim) * gemm_state->k_dim;
    const size_t padding_bytes =
        static_cast<size_t>(gemm_state->padded_m - gemm_state->m_dim) *
        gemm_state->k_dim;
    quantize_status = cudaMemsetAsync(
        gemm_state->quantized_activation + padding_offset, 0,
        padding_bytes, stream);
    if (quantize_status != cudaSuccess) {
      return -3;
    }
  }

  quantize_status = cudaErrorInvalidValue;
  switch (activation_type) {
#ifdef ENABLE_FP16
    case FP8BlockwiseActivationType::FP16:
      quantize_status = QuantizeFP8BlockwiseFP16(
          activation, gemm_state->quantized_activation,
          gemm_state->activation_scale, gemm_state->m_dim,
          gemm_state->k_dim, gemm_state->block_k, stream);
      break;
#endif
#ifdef ENABLE_BF16
    case FP8BlockwiseActivationType::BF16:
      quantize_status = QuantizeFP8BlockwiseBF16(
          activation, gemm_state->quantized_activation,
          gemm_state->activation_scale, gemm_state->m_dim,
          gemm_state->k_dim, gemm_state->block_k, stream);
      break;
#endif
    default:
      return -2;
  }
  if (quantize_status != cudaSuccess) {
    return -3;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  for (int k_group = 0; k_group < gemm_state->k_group_count; ++k_group) {
    const size_t k_offset = static_cast<size_t>(k_group) *
                            gemm_state->block_k;
    const auto* activation_block =
        gemm_state->quantized_activation + k_offset;
    const auto* weight_block = static_cast<const uint8_t*>(weight) + k_offset;
    const cublasStatus_t gemm_status = cublasLtMatmul(
        handle, gemm_state->operation, &alpha,
        activation_block, gemm_state->activation_layout,
        weight_block, gemm_state->weight_layout, &beta,
        gemm_state->partial, gemm_state->partial_layout,
        gemm_state->partial, gemm_state->partial_layout,
        &gemm_state->algorithm, gemm_state->cublas_workspace,
        gemm_state->cublas_workspace_size, stream);
    if (gemm_status != CUBLAS_STATUS_SUCCESS) {
      return -4;
    }

    cudaError_t accumulate_status = cudaErrorInvalidValue;
    switch (activation_type) {
#ifdef ENABLE_FP16
      case FP8BlockwiseActivationType::FP16:
        accumulate_status = LaunchAccumulateAndFinalize<half>(
            gemm_state, weight_scale, bias, output, k_group, stream);
        break;
#endif
#ifdef ENABLE_BF16
      case FP8BlockwiseActivationType::BF16:
        accumulate_status = LaunchAccumulateAndFinalize<hie::bfloat16>(
            gemm_state, weight_scale, bias, output, k_group, stream);
        break;
#endif
      default:
        return -2;
    }
    if (accumulate_status != cudaSuccess) {
      return -5;
    }
  }
  return 0;
}

}  // namespace cuda
}  // namespace allspark

#endif  // ENABLE_FP8
