/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_generation.cpp
 */

#include "model.h"  // NOLINT

#include <stdexcept>

namespace allspark {

// Called by every worker thread; ownership changes stay behind request_map_lock_.
AsStatus AsModel::StartRequestImpl(
    const std::shared_ptr<RequestHandle> request_handle,
    const std::string request_id, TensorMap* outputs,
    const GenerateConfig& gen_cfg) {
  DLOG(INFO) << "AsModel::StartRequestImpl()" << std::endl;
  std::shared_ptr<Request> request = std::make_shared<Request>(
      request_id, *request_handle->inputs_internal, *outputs, gen_cfg);
  request->input_len = request->inputs.at("input_ids")->GetShape()[1];
  request->origin_len = request->input_len;
  request->extra_embedding = request_handle->mm_embedding_internal;
  request->enqueue_ts = request_handle->create_ts;
#ifdef ENABLE_JSON_MODE
  if (gen_cfg.response_format.count("type")) {
    try {
      if (gen_cfg.response_format.at("type") == "json_object") {
        request->format_enforcer = request_handle->format_enforcer;
      }
    } catch (const std::out_of_range&) {
      // Missing response format is equivalent to the default text mode.
    }
  }
#endif
  DLOG(INFO) << "AsModel::StartRequestImpl(): input length:"
             << request->input_len;

  std::unique_lock<std::mutex> lock(request_map_lock_);
  pending_request_queue_.push(request);
  all_request_map_[request_id] = request;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::GenerateContinue() {
  AsStatus status = GenerateContinueDecoder();
  AS_CHECK_STATUS(status);
  return status;
}

}  // namespace allspark
