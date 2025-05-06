/*!
 * Copyright (c) 2026 Segno System.
 * @file    gemm_nvfp4_blockwise_gpu.h
 */

#pragma once

#ifdef ENABLE_CUDA

#include <core/operator/operator.h>

#include <cstdint>
#include <string>

namespace allspark {

class GemmNVFP4BlockwiseGPU : public AsOperator {
 public:
  explicit GemmNVFP4BlockwiseGPU(const std::string& op_type = "")
      : AsOperator(op_type) {}

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map) override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override { return Reshape(); }
  AsStatus Forward(RuntimeContext* runtime_ctx) override { return Forward(); }
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  DataType activation_type_ = DataType::DATATYPE_UNDEFINED;
  int64_t m_ = 0;
  int64_t n_ = 0;
  int64_t k_ = 0;
  int32_t block_size_ = 16;
};

}  // namespace allspark

#endif  // ENABLE_CUDA
