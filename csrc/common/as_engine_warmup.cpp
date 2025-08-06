/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_warmup.cpp
 */

#include "as_engine_impl.h"

#include <device/memory_func.h>

#include <chrono>

namespace allspark {

AsStatus AsEngineImpl::TunePrefixCache(const char* model_name) {
#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA) {
    int tokens_per_span = device_ctx_->GetCacheSpanSize();
    int request_id = engine_max_batch_ + 1;

    for (int i = 0; i < nranks_; ++i) {
      workers_[i]->ResetPrefixCache();
    }

    for (int cache_len = tokens_per_span;
         cache_len + 1 < engine_max_length_ - 2;
         cache_len *= 2, request_id++) {
      int seq_len = cache_len + 1;
      AsTensor input_ids("input_ids", DeviceType::CPU, DataType::INT64,
                         DataMode::DENSE, Shape({1, seq_len}));
      TensorUtils::Memset(input_ids, cache_len);
      const DLTensorMap warmup_inputs = {
          {"input_ids", input_ids.ToDLPack(nullptr)}};

      auto warmup_request = std::make_shared<AsEngine::RequestContent>();
      warmup_request->config.max_length = seq_len + 1;
      warmup_request->config.top_k = 0;
      warmup_request->config.top_p = 0.5;
      warmup_request->infer_type = AsEngine::RequestInferType::Generate;
      warmup_request->inputs = std::make_shared<DLTensorMap>(warmup_inputs);
      warmup_request->mm_type = AsEngine::RequestMMType::TextInput;

      float duration_ms[2] = {0};
      for (int run = 0; run < 2; run++) {
        warmup_request->config.uuid = "warmup_request_" +
                                      std::to_string(request_id) + "_" +
                                      std::to_string(run);
        RequestHandle* request_handle{nullptr};
        AsEngine::ResultQueue* result_queue{nullptr};

        auto start = std::chrono::steady_clock::now();
        AS_CHECK_STATUS(StartRequest(model_name, warmup_request,
                                     &request_handle, &result_queue));
        AS_CHECK_STATUS(SyncRequest(model_name, request_handle));
        AS_CHECK_STATUS(ReleaseRequest(model_name, request_handle));
        auto duration = std::chrono::steady_clock::now() - start;
        duration_ms[run] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
                .count() /
            1000000.0;
      }

      LOG(INFO) << __FUNCTION__ << ": cache_len: " << cache_len
                << ", duration_ms[0]: " << duration_ms[0] << " ms"
                << ", duration_ms[1]: " << duration_ms[1] << " ms";
      if (duration_ms[0] > duration_ms[1]) {
        for (int i = 0; i < nranks_; ++i) {
          workers_[i]->SetPrefixCacheSeqlenThre(cache_len);
        }
        break;
      }
    }
  }
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark
