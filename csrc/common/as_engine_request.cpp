/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_request.cpp
 */

#include "as_engine_impl.h"
#include "engine_control_message.h"
#include "extra_embedding.hpp"

#include <device/memory_func.h>
#include <utility/timer.h>
#include <utility/uuid.h>

#include "runtime/weight/weight_manager_lora.h"

#include <cassert>
#include <future>

#ifdef ENABLE_JSON_MODE
#include <utility/format_enforcer.h>
#endif

namespace allspark {

AsStatus AsEngineImpl::StartRequest(
    const char* model_name,
    std::shared_ptr<AsEngine::RequestContent> request_info,
    RequestHandle** request_handle, AsEngine::ResultQueue** queue,
    const std::string customized_uuid) {
  std::string uuid =
      customized_uuid.empty() ? GenNewUUID() : customized_uuid;
  std::lock_guard<std::mutex> lora_guard(lora_lock_);
  TracerLog trace(device_ctx_->GetDeviceType(), "StartRequest", 1);

  if (!request_info) {
    LOG(ERROR) << "[" << model_name
               << "] StartRequest with null request_info "
               << static_cast<int>(AsStatus::ALLSPARK_PARAM_ERROR);
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  LOG(INFO) << "[" << model_name << "] StartRequest Received: "
            << to_string(*request_info) << " uuid: " << uuid;

  AsStatus status = InputParamsVerify(model_name, request_info);
  if (status != AsStatus::ALLSPARK_SUCCESS) {
    LOG(ERROR) << "[" << model_name << "] StartRequest failed with error "
               << static_cast<int>(status);
    return status;
  }

  TensorListMap extra_embedding;
  ExtraEmbeddingUtils::ParseMmInfo(extra_embedding,
                                   request_info->config.mm_info);
  status = RichInputVerify(extra_embedding, request_info);
  if (status != AsStatus::ALLSPARK_SUCCESS) {
    LOG(ERROR) << "[" << model_name
               << "] StartRequest failed with rich input verify error "
               << static_cast<int>(status);
    return status;
  }

  const auto& lora_name = request_info->config.lora_name;
  if (!lora_name.empty()) {
    LOG(INFO) << "req lora_name=" << lora_name;
    if (!workers_[0]->GetModel()->GetLoraManager()->IsLoraExists(lora_name)) {
      LOG(ERROR) << "LoRA " << lora_name << " not found, cannot StartRequest!";
      return AsStatus::ALLSPARK_LORA_NOT_FOUND;
    }
    std::lock_guard<std::mutex> usage_guard(lora_usage_lock_);
    lora_use_count_++;
    loras_in_use_[model_name].insert(lora_name);
  }

  auto reply_promise = std::make_shared<std::promise<AsStatus>>();
  auto handle = std::make_shared<RequestHandle>();
  handle->request_uuid = uuid;
  handle->context_length =
      (*request_info->inputs)["input_ids"]->dl_tensor.shape[1];
  handle->inputs_internal = TensorUtils::DeepCopyDLTensorMapToTensorMap(
      request_info->inputs, DeviceType::CPU);
  handle->mm_type_internal = request_info->mm_type;
  handle->mm_embedding_internal = extra_embedding;

#ifdef ENABLE_JSON_MODE
  if (request_info->config.response_format.count("type") &&
      request_info->config.response_format["type"] == "json_object") {
    if (util::FormatEnforcer::vocab_.empty() &&
        request_info->config.vocab.empty()) {
      request_info->config.response_format["type"] = "";
    } else {
      std::string schema =
          request_info->config.response_format.find("json_schema") !=
                  request_info->config.response_format.end()
              ? request_info->config.response_format["json_schema"]
              : "";
      handle->format_enforcer = std::make_shared<util::FormatEnforcer>(
          request_info->config.vocab, schema, request_info->config.vocab_type,
          request_info->config.eos_token_id);
    }
  }
#endif

  auto result_queue = std::make_shared<ResultQueueImpl>(uuid);
  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  if (model_state->model_stopping.load()) {
    LOG(INFO) << "model is stopping, access denied";
    return AsStatus::ALLSPARK_REQUEST_DENIED;
  }

#ifndef ENABLE_CUDA
  workers_[0]->GetDeviceContext()->SemPostInterProcess();
#endif
  auto message = EngineControlMessage(EngineControlMessageId::StartRequest,
                                      reply_promise, uuid, handle, result_queue,
                                      request_info);
  model_state->msg_queue.enqueue(std::move(message));
#ifndef ENABLE_CUDA
  workers_[0]->GetDeviceContext()->SemWaitSendInterProcess();
#endif

  DLOG(INFO) << "[" << model_name << "] request with uuid "
             << handle->request_uuid << " notified";
  *request_handle = handle.get();
  *queue = result_queue.get();
  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark
