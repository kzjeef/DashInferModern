/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_request_context.cpp
 */

#include "model.h"  // NOLINT

#include <common/device_context.h>

#include <cstring>
#include <random>
#include <sstream>

#ifdef ENABLE_CUDA
#include <core/kernel/cuda/sample.h>
#include <cuda/cuda_context.h>
#include <curand_kernel.h>
#endif

namespace allspark {

AsStatus AsModel::buildGenContext(
    std::shared_ptr<GenerateContext>& gen_ctx,
    const std::shared_ptr<Request>& request) const {
  gen_ctx->gen_cfg = request->gen_cfg;
  gen_ctx->request = request;
  gen_ctx->k_cache_list = std::vector<std::unique_ptr<CacheMemory>>();
  gen_ctx->v_cache_list = std::vector<std::unique_ptr<CacheMemory>>();

  gen_ctx->engine_max_length = ctx_->GetModelMaxLength();
  gen_ctx->input_len = request->inputs.at("input_ids")->GetShape()[1];
  gen_ctx->real_input_len = gen_ctx->input_len;
  gen_ctx->gen_cfg.input_len = gen_ctx->input_len;
  gen_ctx->max_length = ctx_->GetModelMaxLength();

#ifdef ENABLE_JSON_MODE
  if (request->gen_cfg.response_format.count("type") &&
      request->gen_cfg.response_format["type"] == "json_object") {
    gen_ctx->format_enforcer = request->format_enforcer;
    DLOG(INFO)
        << "Request:" << request->request_id
        << " FormatEnforcer pointer successfully passed to GenerateContext\n";
  }
#endif

  std::stringstream k_cache_tag;
  std::stringstream v_cache_tag;
  k_cache_tag << request->request_id << "_k";
  v_cache_tag << request->request_id << "_v";

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    switch (ctx_->GetCacheSpanSize()) {
      case 0:
        break;
      default:
        gen_ctx->virtual_k_cache = std::make_unique<SpannedVirtualCache>(
            cache_span_manager_, ctx_->GetCacheConfig(), k_cache_tag.str(),
            ctx_->GetDecoderLayer());
        gen_ctx->virtual_v_cache = std::make_unique<SpannedVirtualCache>(
            cache_span_manager_, ctx_->GetCacheConfig(), v_cache_tag.str(),
            ctx_->GetDecoderLayer());
        break;
    }
  }
#endif

  int state_size = sizeof(std::mt19937);
#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (ctx_->GetUseTorchSample()) {
      state_size = static_cast<int>(sizeof(cuda::PhiloxCudaState));
    } else {
      state_size = static_cast<int>(sizeof(curandState_t));
      gen_ctx->sample_state = std::make_unique<AsTensor>(
          "sample_state:" + request->request_id, DeviceType::CUDA,
          DataType::INT8, DataMode::DENSE, Shape({state_size}));
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      cudaStream_t cu_stream = gpu_ctx->GetStream();
      AS_CHECK_CUDA(cudaMemsetAsync(gen_ctx->sample_state->GetDataPtr(), 0,
                                    gen_ctx->sample_state->GetSizeInByte(),
                                    cu_stream));
    }
  }
#endif

  if (gen_ctx->sample_state == nullptr) {
    gen_ctx->sample_state = std::make_unique<AsTensor>(
        "sample_state:" + request->request_id, DeviceType::CPU, DataType::INT8,
        DataMode::DENSE, Shape({state_size}));
    memset(gen_ctx->sample_state->GetDataPtr(), 0,
           gen_ctx->sample_state->GetSizeInByte());
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark
