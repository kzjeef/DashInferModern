/*!
 * Copyright (c) 2026 Segno System.
 * @file    gemm_fp8_blockwise_gpu.h
 */

#pragma once

#if defined(ENABLE_CUDA) && defined(ENABLE_FP8)

#include <core/operator/operator.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace allspark {

class GemmFP8BlockwiseGPU : public AsOperator {
 public:
  explicit GemmFP8BlockwiseGPU(const std::string& op_type = "")
      : AsOperator(op_type) {}
  ~GemmFP8BlockwiseGPU() override;

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
  int32_t block_n_ = 128;
  int32_t block_k_ = 128;
  int64_t workspace_m_capacity_ = 0;
  int64_t prepared_m_ = 0;
  size_t workspace_size_ = 0;
  std::unique_ptr<AsTensor> workspace_;
  void* gemm_state_ = nullptr;
};

}  // namespace allspark

#endif  // ENABLE_CUDA && ENABLE_FP8
