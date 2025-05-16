/*!
 * Copyright (c) 2026 Segno System.
 * @file    gemm_nvfp4_blockwise_sm100.cu
 */

#include "gemm_nvfp4_blockwise.h"

#include "nvfp4_activation_quant.cuh"

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/float_subbyte.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <cstdint>
#include <type_traits>

using namespace cute;

namespace {

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementOutput = cutlass::bfloat16_t;
using ScaleFactor = ElementA::ScaleFactorType;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
        Shape<_128, _256, _256>, Shape<_1, _1, _1>,
        cutlass::epilogue::collective::EpilogueTileAuto,
        float, float,
        ElementOutput, cutlass::layout::RowMajor,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementOutput, cutlass::layout::RowMajor,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
        ElementA, cutlass::layout::RowMajor, 32,
        ElementB, cutlass::layout::ColumnMajor, 32,
        float, Shape<_128, _256, _256>, Shape<_1, _1, _1>,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(
                typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using BlockScaleConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

constexpr size_t kCutlassWorkspaceBytes = 16 * 1024 * 1024;

size_t Align128(size_t value) { return (value + 127) & ~size_t{127}; }

struct WorkspaceLayout {
  size_t activation_bytes;
  size_t scale_bytes;
  size_t cutlass_bytes;
  size_t output_bytes;
  size_t total_bytes;
};

WorkspaceLayout MakeWorkspaceLayout(int m_dim, int n_dim, int k_dim) {
  const int padded_m = ((m_dim + 255) / 256) * 256;
  WorkspaceLayout layout{};
  layout.activation_bytes = Align128(
      static_cast<size_t>(padded_m) * k_dim / 2);
  layout.scale_bytes = Align128(
      static_cast<size_t>(padded_m) * (k_dim / 16));
  layout.cutlass_bytes = Align128(kCutlassWorkspaceBytes);
  layout.output_bytes = Align128(
      static_cast<size_t>(padded_m) * n_dim * sizeof(uint16_t));
  layout.total_bytes = layout.activation_bytes + layout.scale_bytes +
                       layout.cutlass_bytes + layout.output_bytes;
  return layout;
}

struct NvFP4GemmState {
  Gemm gemm;
  typename Gemm::Arguments arguments{};
  void* workspace = nullptr;
  uint8_t* packed_activation = nullptr;
  uint8_t* activation_scale = nullptr;
  uint8_t* cutlass_workspace = nullptr;
  ElementOutput* padded_output = nullptr;
  float activation_scale_multiplier = 1.0f;
  int m_dim = 0;
  int padded_m = 0;
  int n_dim = 0;
  int k_dim = 0;
  size_t workspace_size = 0;
  size_t cutlass_workspace_size = 0;
  bool ready = false;
};

}  // namespace

namespace allspark {
namespace cuda {

void* NvFP4GemmCreate() { return new NvFP4GemmState(); }

void NvFP4GemmDestroy(void* state) {
  delete static_cast<NvFP4GemmState*>(state);
}

size_t NvFP4GemmWorkspaceSize(int m_dim, int n_dim, int k_dim) {
  if (m_dim <= 0 || n_dim <= 0 || k_dim <= 0) {
    return 0;
  }
  return MakeWorkspaceLayout(m_dim, n_dim, k_dim).total_bytes;
}

int NvFP4GemmPrepare(void* state, const void* prepared_weight,
                     const void* prepared_scale, float weight_global_scale,
                     float activation_scale_multiplier, void* output,
                     int m_dim, int n_dim, int k_dim,
                     void* workspace, size_t workspace_size,
                     cudaStream_t stream) {
  auto* gemm_state = static_cast<NvFP4GemmState*>(state);
  if (gemm_state == nullptr || prepared_weight == nullptr ||
      prepared_scale == nullptr || output == nullptr || workspace == nullptr ||
      m_dim <= 0 || n_dim <= 0 || k_dim <= 0 ||
      n_dim % 128 != 0 || k_dim % 64 != 0 ||
      activation_scale_multiplier <= 0.0f) {
    return -1;
  }
  gemm_state->ready = false;

  const WorkspaceLayout layout =
      MakeWorkspaceLayout(m_dim, n_dim, k_dim);
  if (workspace_size < layout.total_bytes) {
    return -2;
  }

  auto* cursor = static_cast<uint8_t*>(workspace);
  gemm_state->packed_activation = cursor;
  cursor += layout.activation_bytes;
  gemm_state->activation_scale = cursor;
  cursor += layout.scale_bytes;
  gemm_state->cutlass_workspace = cursor;
  cursor += layout.cutlass_bytes;
  gemm_state->padded_output = reinterpret_cast<ElementOutput*>(cursor);

  gemm_state->workspace = workspace;
  gemm_state->workspace_size = layout.total_bytes;
  gemm_state->m_dim = m_dim;
  gemm_state->padded_m = ((m_dim + 255) / 256) * 256;
  gemm_state->n_dim = n_dim;
  gemm_state->k_dim = k_dim;
  gemm_state->activation_scale_multiplier = activation_scale_multiplier;

  const auto stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, {gemm_state->padded_m, k_dim, 1});
  const auto stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, {n_dim, k_dim, 1});
  const auto stride_c = cutlass::make_cute_packed_stride(
      StrideC{}, {gemm_state->padded_m, n_dim, 1});
  const auto stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, {gemm_state->padded_m, n_dim, 1});
  const auto layout_sfa = BlockScaleConfig::tile_atom_to_shape_SFA(
      make_shape(gemm_state->padded_m, n_dim, k_dim, 1));
  const auto layout_sfb = BlockScaleConfig::tile_atom_to_shape_SFB(
      make_shape(gemm_state->padded_m, n_dim, k_dim, 1));
  const float alpha = weight_global_scale / activation_scale_multiplier;

  gemm_state->arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {gemm_state->padded_m, n_dim, k_dim, 1},
      {reinterpret_cast<ElementA::DataType*>(
           gemm_state->packed_activation),
       stride_a,
       reinterpret_cast<const ElementB::DataType*>(prepared_weight),
       stride_b,
       reinterpret_cast<const ScaleFactor*>(gemm_state->activation_scale),
       layout_sfa,
       reinterpret_cast<const ScaleFactor*>(prepared_scale),
       layout_sfb},
      {{alpha, 0.0f}, nullptr, stride_c, gemm_state->padded_output, stride_d}};

  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&gemm_state->arguments.hw_info.sm_count,
                             cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess) {
    return -3;
  }
  gemm_state->arguments.hw_info.cluster_shape = dim3(1, 1, 1);
  gemm_state->arguments.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  if constexpr (!std::is_const_v<decltype(
                    gemm_state->arguments.scheduler.max_swizzle_size)>) {
    gemm_state->arguments.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(
                    gemm_state->arguments.scheduler.raster_order)>) {
    using RasterOrder =
        decltype(gemm_state->arguments.scheduler.raster_order);
    gemm_state->arguments.scheduler.raster_order = RasterOrder::Heuristic;
  }

  if (gemm_state->gemm.can_implement(gemm_state->arguments) !=
      cutlass::Status::kSuccess) {
    return -4;
  }
  gemm_state->cutlass_workspace_size =
      Gemm::get_workspace_size(gemm_state->arguments);
  if (gemm_state->cutlass_workspace_size > layout.cutlass_bytes) {
    return -5;
  }
  if (gemm_state->gemm.initialize(
          gemm_state->arguments, gemm_state->cutlass_workspace, stream) !=
      cutlass::Status::kSuccess) {
    return -6;
  }

  gemm_state->ready = true;
  return 0;
}

int NvFP4GemmRun(void* state, const void* activation, void* output,
                 int m_dim, int n_dim, int k_dim,
                 void* workspace, size_t workspace_size,
                 cudaStream_t stream) {
  auto* gemm_state = static_cast<NvFP4GemmState*>(state);
  if (gemm_state == nullptr || !gemm_state->ready || activation == nullptr ||
      output == nullptr || workspace != gemm_state->workspace ||
      workspace_size < gemm_state->workspace_size ||
      m_dim != gemm_state->m_dim || n_dim != gemm_state->n_dim ||
      k_dim != gemm_state->k_dim) {
    return -1;
  }

  const size_t packed_activation_bytes =
      static_cast<size_t>(gemm_state->padded_m) * k_dim / 2;
  const size_t activation_scale_bytes =
      static_cast<size_t>(gemm_state->padded_m) * (k_dim / 16);
  cudaError_t cuda_status = cudaMemsetAsync(
      gemm_state->packed_activation, 0, packed_activation_bytes, stream);
  if (cuda_status != cudaSuccess) {
    return -2;
  }
  cuda_status = cudaMemsetAsync(
      gemm_state->activation_scale, 0, activation_scale_bytes, stream);
  if (cuda_status != cudaSuccess) {
    return -2;
  }

  NvFP4QuantizeActivation<__nv_bfloat16>(
      activation, gemm_state->packed_activation,
      gemm_state->activation_scale, m_dim, k_dim,
      gemm_state->activation_scale_multiplier, stream);
  if (cudaGetLastError() != cudaSuccess) {
    return -3;
  }

  if (gemm_state->gemm.run(stream) != cutlass::Status::kSuccess) {
    return -4;
  }
  cuda_status = cudaMemcpyAsync(
      output, gemm_state->padded_output,
      static_cast<size_t>(m_dim) * n_dim * sizeof(uint16_t),
      cudaMemcpyDeviceToDevice, stream);
  return cuda_status == cudaSuccess ? 0 : -5;
}

}  // namespace cuda
}  // namespace allspark
