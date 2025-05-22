/*!
 * Copyright (c) 2026 Segno System.
 * @file    gemm_nvfp4_blockwise_gpu.cpp
 */

#if defined(ENABLE_CUDA) && defined(ENABLE_SM100_NVFP4)

#include "gemm_nvfp4_blockwise_gpu.h"

#include <core/kernel/cuda/gemm_lowp/gemm_nvfp4_blockwise.h>
#include <cuda/cuda_context.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace allspark {

GemmNVFP4BlockwiseGPU::~GemmNVFP4BlockwiseGPU() {
  if (gemm_state_ != nullptr) {
    cuda::NvFP4GemmDestroy(gemm_state_);
    gemm_state_ = nullptr;
  }
}

AsStatus GemmNVFP4BlockwiseGPU::Init(const OperatorProto& op_proto,
                                     const DeviceContext& ctx,
                                     const TensorMap& weights_map,
                                     TensorMap* tensor_map) {
  LOG(ERROR) << "GemmNVFP4BlockwiseGPU requires InitV2" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmNVFP4BlockwiseGPU::InitV2(
    const OperatorProto& op_proto, const DeviceContext& ctx,
    const TensorMap& weights_map, TensorMap& weights_buffer,
    TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));

  if (weights_.size() != 4) {
    LOG(ERROR) << "GemmNVFP4BlockwiseGPU expects packed weight, block scale, "
                  "global scale, and input scale"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  const auto block_size_attr = op_proto.attr().find("nvfp4_block_size");
  if (block_size_attr != op_proto.attr().end()) {
    if (block_size_attr->second.size() != sizeof(block_size_)) {
      LOG(ERROR) << "GemmNVFP4BlockwiseGPU has an invalid block size attribute"
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    std::memcpy(&block_size_, block_size_attr->second.data(),
                sizeof(block_size_));
  }
  if (block_size_ != 16) {
    LOG(ERROR) << "GemmNVFP4BlockwiseGPU supports ModelOpt block size 16, got "
               << block_size_ << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  const Shape& packed_shape = weights_[0]->GetShape();
  if (weights_[0]->GetDataType() != DataType::UINT8 ||
      packed_shape.Size() != 2) {
    LOG(ERROR) << "NVFP4 weight must be a 2D UINT8 packed tensor"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  n_ = packed_shape[0];
  k_ = packed_shape[1] * 2;
  if (n_ % 128 != 0 || k_ % 64 != 0) {
    LOG(ERROR) << "NVFP4 requires N divisible by 128 and K divisible by 64"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  const Shape& scale_shape = weights_[1]->GetShape();
  if (weights_[1]->GetDataType() != DataType::FLOAT8E4M3 ||
      scale_shape.Size() != 2 || scale_shape[0] != n_ ||
      scale_shape[1] != k_ / block_size_) {
    LOG(ERROR) << "NVFP4 block scale must be FLOAT8E4M3 [N, K/16]"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  for (int index : {2, 3}) {
    if (weights_[index]->GetDataType() != DataType::FLOAT32 ||
        weights_[index]->GetShape().Count() != 1) {
      LOG(ERROR) << "NVFP4 global and input scales must be FP32 scalars"
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }

  activation_type_ = tensor_map_->at(in_names_[0])->GetDataType();
  if (activation_type_ != DataType::BFLOAT16) {
    LOG(ERROR) << "NVFP4 currently requires BFLOAT16 activation" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (ctx.GetNranks() != 1) {
    LOG(ERROR) << "NVFP4 mini-model path currently supports one GPU"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  int device_id = -1;
  cudaDeviceProp device_prop{};
  if (cudaGetDevice(&device_id) != cudaSuccess ||
      cudaGetDeviceProperties(&device_prop, device_id) != cudaSuccess ||
      device_prop.major != 10 || device_prop.minor != 0) {
    LOG(ERROR) << "NVFP4 native GEMM requires an SM100 GPU" << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  const cudaStream_t stream = gpu_ctx->GetStream();
  if (cudaMemcpyAsync(&weight_global_scale_, weights_[2]->GetDataPtr(),
                      sizeof(float), cudaMemcpyDeviceToHost, stream) !=
          cudaSuccess ||
      cudaMemcpyAsync(&activation_scale_multiplier_,
                      weights_[3]->GetDataPtr(), sizeof(float),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    LOG(ERROR) << "failed to read NVFP4 scalar scales" << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  gpu_ctx->Synchronize();
  if (!std::isfinite(weight_global_scale_) ||
      !std::isfinite(activation_scale_multiplier_) ||
      weight_global_scale_ <= 0.0f || activation_scale_multiplier_ <= 0.0f) {
    LOG(ERROR) << "NVFP4 scalar scales must be finite and positive"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  activation_scale_multiplier_ = 1.0f / activation_scale_multiplier_;

  const int64_t packed_bytes = n_ * k_ / 2;
  const int64_t scale_bytes = n_ * (k_ / block_size_);
  prepared_weight_ = std::make_unique<AsTensor>(
      weights_[0]->GetName() + ".nvfp4_packed", DeviceType::CUDA,
      DataType::UINT8, DataMode::DENSE, Shape({packed_bytes}));
  prepared_scale_ = std::make_unique<AsTensor>(
      weights_[1]->GetName() + ".nvfp4_interleaved", DeviceType::CUDA,
      DataType::UINT8, DataMode::DENSE, Shape({scale_bytes}));
  const cudaError_t prepare_status = cuda::NvFP4PrepareWeight(
      weights_[0]->GetDataPtr(), weights_[1]->GetDataPtr(),
      prepared_weight_->GetDataPtr(), prepared_scale_->GetDataPtr(),
      static_cast<int>(n_), static_cast<int>(k_), stream);
  if (prepare_status != cudaSuccess) {
    LOG(ERROR) << "failed to prepare packed NVFP4 weight: "
               << cudaGetErrorString(prepare_status) << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  gpu_ctx->Synchronize();

  workspace_m_capacity_ = std::max<int64_t>(
      {1, ctx.GetModelMaxBatch(), ctx.GetModelMaxLength(),
       ctx.GetModelMaxPrefillLength()});
  if (workspace_m_capacity_ > std::numeric_limits<int>::max()) {
    LOG(ERROR) << "NVFP4 workspace row capacity exceeds INT_MAX" << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  workspace_size_ = cuda::NvFP4GemmWorkspaceSize(
      static_cast<int>(workspace_m_capacity_), static_cast<int>(n_),
      static_cast<int>(k_));
  if (workspace_size_ == 0 ||
      workspace_size_ >
          static_cast<size_t>(std::numeric_limits<dim_t>::max())) {
    LOG(ERROR) << "invalid NVFP4 workspace size" << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  workspace_ = std::make_unique<AsTensor>(
      weights_[0]->GetName() + ".nvfp4_workspace", DeviceType::CUDA,
      DataType::UINT8, DataMode::DENSE,
      Shape({static_cast<dim_t>(workspace_size_)}));
  gemm_state_ = cuda::NvFP4GemmCreate();
  if (gemm_state_ == nullptr) {
    LOG(ERROR) << "failed to create NVFP4 GEMM state" << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  tensor_map_->at(out_names_[0])->SetDataType(activation_type_);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmNVFP4BlockwiseGPU::Reshape() {
  const Shape& input_shape = tensor_map_->at(in_names_[0])->GetShape();
  if (input_shape.Size() == 0 || input_shape[-1] != k_) {
    LOG(ERROR) << "NVFP4 input K does not match packed weight K" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  m_ = input_shape.Count(0, input_shape.Size() - 1);
  if (m_ <= 0 || m_ > workspace_m_capacity_ ||
      m_ > std::numeric_limits<int>::max()) {
    LOG(ERROR) << "NVFP4 input rows exceed the preallocated workspace"
               << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  Shape output_shape;
  for (int index = 0; index < input_shape.Size() - 1; ++index) {
    output_shape.Append(input_shape[index]);
  }
  output_shape.Append(n_);
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(output_shape)));

  if (prepared_m_ != m_) {
    const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
    const int prepare_status = cuda::NvFP4GemmPrepare(
        gemm_state_, prepared_weight_->GetDataPtr(),
        prepared_scale_->GetDataPtr(), weight_global_scale_,
        activation_scale_multiplier_,
        tensor_map_->at(out_names_[0])->GetDataPtr(), static_cast<int>(m_),
        static_cast<int>(n_), static_cast<int>(k_), workspace_->GetDataPtr(),
        workspace_size_, gpu_ctx->GetStream());
    if (prepare_status != 0) {
      LOG(ERROR) << "failed to prepare NVFP4 GEMM: " << prepare_status
                 << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
    prepared_m_ = m_;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmNVFP4BlockwiseGPU::Forward() {
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  const int run_status = cuda::NvFP4GemmRun(
      gemm_state_, tensor_map_->at(in_names_[0])->GetDataPtr(),
      tensor_map_->at(out_names_[0])->GetDataPtr(), static_cast<int>(m_),
      static_cast<int>(n_), static_cast<int>(k_), workspace_->GetDataPtr(),
      workspace_size_, gpu_ctx->GetStream());
  if (run_status != 0) {
    LOG(ERROR) << "NVFP4 GEMM launch failed: " << run_status << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(GemmNVFP4Blockwise, CUDA, GemmNVFP4BlockwiseGPU)

}  // namespace allspark

#endif  // ENABLE_CUDA && ENABLE_SM100_NVFP4
