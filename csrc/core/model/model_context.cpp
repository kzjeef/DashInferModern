/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_context.cpp
 */

#include "model.h"  // NOLINT

#include <common/device_context.h>
#include <common/engine_runtime.h>
#include <common/extra_embedding.hpp>
#include <utility/timer.h>

#include <algorithm>

namespace allspark {

AsStatus AsModel::GenerateContinueContext(bool is_new_context) {
  util::Timer t0;
  std::unique_lock<std::mutex> lock(gen_ctx_lock_);

  if (!runtime_ctx_) {
    return AsStatus::ALLSPARK_EMPTY_REQUEST;
  }

  DLOG(INFO) << " gen ctx list " << runtime_ctx_->GetGenCtxListSize()
             << " pending init " << pending_request_queue_.size();
  if (is_new_context) {
    already_context_length_ = 0;
  }
  if (!pending_request_queue_.empty() &&
      runtime_ctx_->GetGenCtxListSize() < ctx_->GetModelMaxBatch()) {
    std::shared_ptr<Request> request = pending_request_queue_.front();
    if (already_context_length_ != 0 &&
        already_context_length_ + request->input_len >
            ctx_->GetModelMaxPrefillLength()) {
      return AsStatus::ALLSPARK_EMPTY_REQUEST;
    }
#if ENABLE_SPAN_ATTENTION
    if (ctx_->GetDeviceType() == DeviceType::CUDA) {
      if (!cache_frame_manager_) {
        pending_request_queue_.pop();
        StartRequest(request);
        DLOG(INFO) << "RunContext SUCCESS ,request id = "
                   << request->request_id;

        current_unfinished_request_.store(pending_request_queue_.size() +
                                          runtime_ctx_->GetGenCtxListSize());
        return AsStatus::ALLSPARK_SUCCESS;
      } else {
        int model_layer = ctx_->GetDecoderLayer();
        int min_gen_length = 10;
        int span_size = ctx_->GetCacheSpanSize();
        size_t context_frame = 0;
        if (prefix_cache_manager_ != nullptr) {
          AS_CHECK_STATUS(ExtraEmbeddingUtils::CreateTensorForHash(
              request, request->interim, request->inputs, "input_ids"));

          std::vector<PrefixCacheManager::PrefixNodePtr> prefix_cache_node_list;
          int prefix_len = 0;
          int gpu_cached_len = 0;
          prefix_cache_manager_->RefOnly(
              request->interim.at("input_ids_for_hash"), request->start_ts,
              prefix_len, gpu_cached_len, prefix_cache_node_list);

          int real_input = request->input_len - prefix_len;
          if (request->prefill_chunk_len == 0) {
            prefix_cache_manager_->UpdateCnt(prefix_len, real_input);
            request->prefill_chunk_len = prefix_len;
            request->prefix_len = prefix_len;
            request->prefix_len_gpu = gpu_cached_len;
            if (prefix_len != 0) {
              LOG(INFO) << "request: " << request->request_id << ", "
                        << "find prefix cache, len: " << prefix_len << ", "
                        << "total tokens: " << request->input_len;
            }
          }
          context_frame =
              (std::min((request->input_len - gpu_cached_len) + min_gen_length,
                        ctx_->GetModelMaxLength()) /
                   span_size +
               1) *
              2 * model_layer;
          if (context_frame > cache_frame_manager_->CountFreeFrame()) {
            LOG(INFO) << "Not enough frame for new request, "
                      << "need frame vs free frame: " << context_frame << "/ "
                      << cache_frame_manager_->CountFreeFrame()
                      << ", swap unrefered prefix cache to cpu memory";
            prefix_cache_manager_->EvictUnrefered(context_frame);
          }
          prefix_cache_manager_->UnRef(prefix_cache_node_list);
        } else {
          context_frame = (std::min(request->input_len + min_gen_length,
                                    ctx_->GetModelMaxLength()) /
                               span_size +
                           1) *
                          2 * model_layer;
        }

        if (context_frame <= cache_frame_manager_->CountFreeFrame()) {
          PrefillChunkRequest(request);
          DLOG(INFO) << "RunContext SUCCESS ,request id = "
                     << request->request_id;
          current_unfinished_request_.store(pending_request_queue_.size() +
                                            runtime_ctx_->GetGenCtxListSize());
          if (request->prefill_chunk_len != request->input_len) {
            return AsStatus::ALLSPARK_CHUNK_PREFILL;
          } else {
            already_context_length_ += request->input_len;
            return AsStatus::ALLSPARK_SUCCESS;
          }
        } else {
          LOG(ERROR) << "Try RunContext " << request->request_id
                     << ", but not enough frame, so RunContext failed";
          pending_request_queue_.pop();
          request->finish = true;
          request->status =
              AsEngine::GenerateRequestStatus::GenerateInterrupted;
          LOG(INFO) << request->request_id << "GenerateInterrupted";
          current_unfinished_request_.store(pending_request_queue_.size() +
                                            runtime_ctx_->GetGenCtxListSize());
          return AsStatus::ALLSPARK_EMPTY_REQUEST;
        }
      }
    } else
#endif
    {
      pending_request_queue_.pop();
      StartRequest(request);
      DLOG(INFO) << "RunContext SUCCESS ,request id = " << request->request_id;
    }

    current_unfinished_request_.store(pending_request_queue_.size() +
                                      runtime_ctx_->GetGenCtxListSize());
  } else {
    return AsStatus::ALLSPARK_EMPTY_REQUEST;
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark
