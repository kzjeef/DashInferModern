/*!
 * Copyright (c) 2026 Segno System.
 * @file    gemm_fp8_blockwise_gpu.cpp
 */

#if defined(ENABLE_CUDA) && defined(ENABLE_FP8)

#include "gemm_fp8_blockwise_gpu.h"

#include <core/kernel/cuda/gemm_lowp/gemm_fp8_blockwise.h>
#include <cuda/cuda_context.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace allspark {
namespace {

template <typename T>
bool ReadAttr(const OperatorProto& op_proto, const char* name, T* value) {
  const auto found = op_proto.attr().find(name);
  if (found == op_proto.attr().end()) {
    return true;
  }
  if (found->second.size() != sizeof(T)) {
    return false;
  }
  std::memcpy(value, found->second.data(), sizeof(T));
  return true;
}

}  // namespace

GemmFP8BlockwiseGPU::~GemmFP8BlockwiseGPU() {
  if (gemm_state_ != nullptr) {
    cuda::FP8BlockwiseGemmDestroy(gemm_state_);
    gemm_state_ = nullptr;
  }
}

AsStatus GemmFP8BlockwiseGPU::Init(const OperatorProto& op_proto,
                                   const DeviceContext& ctx,
                                   const TensorMap& weights_map,
                                   TensorMap* tensor_map) {
  (void)op_proto;
  (void)ctx;
  (void)weights_map;
  (void)tensor_map;
  LOG(ERROR) << "GemmFP8BlockwiseGPU requires InitV2" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmFP8BlockwiseGPU::InitV2(
    const OperatorProto& op_proto, const DeviceContext& ctx,
    const TensorMap& weights_map, TensorMap& weights_buffer,
    TensorMap* tensor_map) {
  (void)weights_buffer;
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));

  if (weights_.size() != 2 && weights_.size() != 3) {
    LOG(ERROR) << "GemmFP8BlockwiseGPU expects weight, block scale, and an "
                  "optional bias"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  bool trans_a = false;
  bool trans_b = false;
  bool is_pooler = false;
  float alpha = 1.0f;
  UnaryType activation = UnaryType::UNARYTYPE_UNDEFINED;
  BinaryType binary_type = BinaryType::BINARYTYPE_UNDEFINED;
  if (!ReadAttr(op_proto, "transA", &trans_a) ||
      !ReadAttr(op_proto, "transB", &trans_b) ||
      !ReadAttr(op_proto, "is_pooler", &is_pooler) ||
      !ReadAttr(op_proto, "alpha", &alpha) ||
      !ReadAttr(op_proto, "activation", &activation) ||
      !ReadAttr(op_proto, "binary_type", &binary_type) ||
      !ReadAttr(op_proto, "fp8_block_m", &block_n_) ||
      !ReadAttr(op_proto, "fp8_block_n", &block_k_)) {
    LOG(ERROR) << "GemmFP8BlockwiseGPU received a malformed attribute"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (trans_a || trans_b || is_pooler || alpha != 1.0f ||
      activation != UnaryType::UNARYTYPE_UNDEFINED ||
      binary_type != BinaryType::BINARYTYPE_UNDEFINED) {
    LOG(ERROR) << "GemmFP8BlockwiseGPU supports only plain A x W^T GEMM"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (block_n_ <= 0 || block_k_ <= 0) {
    LOG(ERROR) << "FP8 block dimensions must be positive" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  const Shape& weight_shape = weights_[0]->GetShape();
  if (weights_[0]->GetDataType() != DataType::FLOAT8E4M3 ||
      weight_shape.Size() != 2 || weight_shape[0] <= 0 ||
      weight_shape[1] <= 0) {
    LOG(ERROR) << "FP8 weight must be a non-empty E4M3 tensor in [N, K] "
                  "checkpoint layout"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  n_ = weight_shape[0];
  k_ = weight_shape[1];

  const Shape& scale_shape = weights_[1]->GetShape();
  const int64_t scale_n = (n_ + block_n_ - 1) / block_n_;
  const int64_t scale_k = (k_ + block_k_ - 1) / block_k_;
  if (weights_[1]->GetDataType() != DataType::FLOAT32 ||
      scale_shape.Size() != 2 || scale_shape[0] != scale_n ||
      scale_shape[1] != scale_k) {
    LOG(ERROR) << "FP8 inverse scale must be FP32 [ceil(N/block_n), "
                  "ceil(K/block_k)]"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  activation_type_ = tensor_map_->at(in_names_[0])->GetDataType();
  if (activation_type_ != DataType::FLOAT16 &&
      activation_type_ != DataType::BFLOAT16) {
    LOG(ERROR) << "FP8 block GEMM requires FP16 or BF16 activations"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (weights_.size() == 3 &&
      (weights_[2]->GetDataType() != activation_type_ ||
       weights_[2]->GetShape().Count() != n_)) {
    LOG(ERROR) << "FP8 block GEMM bias must match the activation type and N"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (ctx.GetNranks() != 1) {
    LOG(ERROR) << "FP8 block GEMM miniature path currently supports one GPU"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (n_ > std::numeric_limits<int>::max() ||
      k_ > std::numeric_limits<int>::max()) {
    LOG(ERROR) << "FP8 block GEMM dimensions exceed INT_MAX" << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  if (n_ % 16 != 0 || k_ % 16 != 0 || block_k_ % 16 != 0 ||
      k_ % block_k_ != 0) {
    LOG(ERROR) << "FP8 block GEMM requires N, K, and block K aligned to 16 "
                  "with K divisible by block K"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  int device_id = -1;
  cudaDeviceProp device_prop{};
  if (cudaGetDevice(&device_id) != cudaSuccess ||
      cudaGetDeviceProperties(&device_prop, device_id) != cudaSuccess ||
      device_prop.major * 10 + device_prop.minor < 89) {
    LOG(ERROR) << "FP8 block GEMM requires an SM89 or newer GPU" << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  workspace_m_capacity_ = std::max<int64_t>(
      {1, ctx.GetModelMaxBatch(), ctx.GetModelMaxLength(),
       ctx.GetModelMaxPrefillLength()});
  if (workspace_m_capacity_ > std::numeric_limits<int>::max()) {
    LOG(ERROR) << "FP8 block GEMM workspace row capacity exceeds INT_MAX"
               << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  workspace_size_ = cuda::FP8BlockwiseGemmWorkspaceSize(
      static_cast<int>(workspace_m_capacity_), static_cast<int>(n_),
      static_cast<int>(k_), block_k_);
  if (workspace_size_ == 0 ||
      workspace_size_ >
          static_cast<size_t>(std::numeric_limits<dim_t>::max())) {
    LOG(ERROR) << "invalid FP8 block GEMM workspace size" << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  workspace_ = std::make_unique<AsTensor>(
      weights_[0]->GetName() + ".fp8_block_workspace", DeviceType::CUDA,
      DataType::UINT8, DataMode::DENSE,
      Shape({static_cast<dim_t>(workspace_size_)}));
  gemm_state_ = cuda::FP8BlockwiseGemmCreate();
  if (gemm_state_ == nullptr) {
    LOG(ERROR) << "failed to create FP8 block GEMM state" << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  tensor_map_->at(out_names_[0])->SetDataType(activation_type_);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmFP8BlockwiseGPU::Reshape() {
  const Shape& input_shape = tensor_map_->at(in_names_[0])->GetShape();
  if (input_shape.Size() == 0 || input_shape[-1] != k_) {
    LOG(ERROR) << "FP8 block GEMM input K does not match weight K"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  m_ = input_shape.Count(0, input_shape.Size() - 1);
  if (m_ <= 0 || m_ > workspace_m_capacity_ ||
      m_ > std::numeric_limits<int>::max()) {
    LOG(ERROR) << "FP8 block GEMM rows exceed the preallocated workspace"
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
    const int prepare_status = cuda::FP8BlockwiseGemmPrepare(
        gemm_state_, gpu_ctx->GetCublasLtHandle(), workspace_->GetDataPtr(),
        workspace_size_, static_cast<int>(m_), static_cast<int>(n_),
        static_cast<int>(k_), block_n_, block_k_);
    if (prepare_status != 0) {
      LOG(ERROR) << "failed to prepare FP8 block GEMM: " << prepare_status
                 << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
    prepared_m_ = m_;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmFP8BlockwiseGPU::Forward() {
  cuda::FP8BlockwiseActivationType activation_type;
  switch (activation_type_) {
#ifdef ENABLE_FP16
    case DataType::FLOAT16:
      activation_type = cuda::FP8BlockwiseActivationType::FP16;
      break;
#endif
#ifdef ENABLE_BF16
    case DataType::BFLOAT16:
      activation_type = cuda::FP8BlockwiseActivationType::BF16;
      break;
#endif
    default:
      LOG(ERROR) << "FP8 block GEMM activation type is not compiled"
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  const void* bias =
      weights_.size() == 3 ? weights_[2]->GetDataPtr() : nullptr;
  const int run_status = cuda::FP8BlockwiseGemmRun(
      gemm_state_, gpu_ctx->GetCublasLtHandle(),
      tensor_map_->at(in_names_[0])->GetDataPtr(),
      weights_[0]->GetDataPtr(),
      static_cast<const float*>(weights_[1]->GetDataPtr()), bias,
      tensor_map_->at(out_names_[0])->GetDataPtr(), activation_type,
      gpu_ctx->GetStream());
  if (run_status != 0) {
    LOG(ERROR) << "FP8 block GEMM launch failed: " << run_status << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(GemmFP8Blockwise, CUDA, GemmFP8BlockwiseGPU)

}  // namespace allspark

#endif  // ENABLE_CUDA && ENABLE_FP8
