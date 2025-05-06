/*!
 * Copyright (c) 2026 Segno System.
 * @file    gemm_nvfp4_blockwise_gpu.cpp
 */

#ifdef ENABLE_CUDA

#include "gemm_nvfp4_blockwise_gpu.h"

#include <cstring>
#include <utility>

namespace allspark {

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
  if (k_ % block_size_ != 0) {
    LOG(ERROR) << "NVFP4 logical K must be divisible by block size"
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
  if (activation_type_ != DataType::FLOAT16 &&
      activation_type_ != DataType::BFLOAT16) {
    LOG(ERROR) << "NVFP4 activation must be FLOAT16 or BFLOAT16"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
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
  Shape output_shape;
  for (int index = 0; index < input_shape.Size() - 1; ++index) {
    output_shape.Append(input_shape[index]);
  }
  output_shape.Append(n_);
  tensor_map_->at(out_names_[0])->SetShape(std::move(output_shape));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmNVFP4BlockwiseGPU::Forward() {
  LOG(ERROR) << "native SM100 NVFP4 kernel is not linked" << std::endl;
  return AsStatus::ALLSPARK_RUNTIME_ERROR;
}

}  // namespace allspark

#endif  // ENABLE_CUDA
