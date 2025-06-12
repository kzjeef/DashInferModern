/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_request.cpp
 */

#include "model.h"  // NOLINT

#include <common/device_context.h>
#include <common/engine_runtime.h>

#include <cassert>
#include <chrono>
#include <exception>

#include "runtime/weight/weight_manager_lora.h"

namespace allspark {

AsStatus AsModel::StartRequest(std::shared_ptr<Request> request) {
  DLOG(INFO) << "AsModel:StartRequest()" << std::endl;

  auto& lora_name = request->gen_cfg.lora_name;
  DLOG(INFO) << "Incoming request: " << request->request_id
             << " lora_name=" << lora_name << std::endl;
  if (!lora_name.empty()) {
    assert(lora_manager_ != nullptr);
    if (!lora_manager_->IsLoraExists(lora_name)) {
      LOG(ERROR) << "check lora in AsModel::StartRequest failed, LoRA "
                 << lora_name << " not loaded!";
      StopRequest(request->request_id);
      request->status = AsEngine::GenerateRequestStatus::GenerateFinished;
      request->finish = true;
      return AsStatus::ALLSPARK_LORA_NOT_FOUND;
    }
  }

  int batch = runtime_ctx_->GetGenCtxListSize();

  std::shared_ptr<GenerateContext> gen_ctx =
      std::make_shared<GenerateContext>();
  AS_CHECK_STATUS(buildGenContext(gen_ctx, request));
  runtime_ctx_->PushBackGenCtx(gen_ctx);

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    const int max_seq_len = ctx_->GetModelMaxLength();
    const int span_len = ctx_->GetCacheSpanSize();
    const int max_num_spans = (max_seq_len + span_len - 1) / span_len;
    for (int layer_id = 0; layer_id < ctx_->GetDecoderLayer(); layer_id++) {
      AS_CHECK_STATUS(gen_ctx->virtual_k_cache->InitLayer(
          layer_id, ctx_->GetNumberHeads(), ctx_->GetSizePerHead(), 0,
          max_num_spans));
      AS_CHECK_STATUS(gen_ctx->virtual_v_cache->InitLayer(
          layer_id, ctx_->GetNumberHeads(), ctx_->GetSizePerHead(), 0,
          max_num_spans));
    }
  }

  if (prefix_cache_manager_ != nullptr &&
      ctx_->GetDeviceType() == DeviceType::CUDA) {
    std::shared_ptr<AsTensor> new_input_ids_tensor;
    prefix_cache_manager_->RefFill(
        gen_ctx->request->inputs.at("input_ids"),
        gen_ctx->request->interim.at("input_ids_for_hash"),
        new_input_ids_tensor, gen_ctx->request->start_ts, gen_ctx->prefix_len,
        gen_ctx->virtual_k_cache, gen_ctx->virtual_v_cache,
        gen_ctx->prefix_cache_node_list);
    gen_ctx->request->interim.insert({"new_input_ids", new_input_ids_tensor});

    if (ctx_->GetRank() == 0) {
      LOG(INFO) << "[" << __FUNCTION__ << "] "
                << "request id: " << gen_ctx->request->request_id << ", "
                << "cached prefix_len: " << gen_ctx->prefix_len << ", "
                << "total tokens: "
                << gen_ctx->request->inputs.at("input_ids")->GetShape()[1];
    }
  } else {
#endif

    {
      int batch_now = request->inputs.at("input_ids")->GetShape()[0];
      int seq_now = request->inputs.at("input_ids")->GetShape()[1];
      gen_ctx->request->interim.insert(
          {"new_input_ids", request->inputs.at("input_ids")});
      tensors_["attention_mask"]->SetShape(Shape({batch_now, seq_now}));
    }

#if ENABLE_SPAN_ATTENTION
  }
#endif

  runtime_ctx_->is_context = true;
  runtime_ctx_->current_batch = batch;
  try {
    DLOG(INFO) << "before enter runcontext for request: " << request->request_id
               << " lora_name " << lora_name
               << " exist: " << lora_manager_->IsLoraExists(lora_name)
               << std::endl;
    runDecoderContext();
  } catch (std::exception& e) {
    LOG(ERROR) << "runDecoderContext() Failed: " << std::string(e.what())
               << ", "
               << "request_id = " << request->request_id;
    StopRequest(request->request_id);
    request->status = AsEngine::GenerateRequestStatus::GenerateInterrupted;
    throw e;
  }

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (prefix_cache_manager_ != nullptr) {
      prefix_cache_manager_->Insert(
          gen_ctx->request->interim.at("input_ids_for_hash"),
          gen_ctx->prefix_len, gen_ctx->request->start_ts,
          gen_ctx->virtual_k_cache->GetLayerCache(),
          gen_ctx->virtual_v_cache->GetLayerCache(),
          gen_ctx->prefix_cache_node_list);
    }
  }
#endif

  runtime_ctx_->is_context = false;
  runtime_ctx_->current_batch = 0;
  DLOG(INFO)
      << "AsModel::StartRequest: context finish, restore ops with Reshape";
  for (AsOperator* op : topo_ops_) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in topo_ops" << std::endl;
      return ErrorProcess(status);
    }
  }

  request->status = AsEngine::GenerateRequestStatus::ContextFinished;

  request->context_ts = std::chrono::steady_clock::now();
  auto duration = request->context_ts - request->start_ts;
  auto duration_in_milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  if (ctx_->GetRank() == 0) {
    LOG(INFO) << "[" << __FUNCTION__ << "] "
              << "Context Success, request id: " << request->request_id << ", "
              << "context phase time(ms): " << duration_in_milliseconds;
  }

  for (int i = runtime_ctx_->GetGenCtxListSize() - 1; i >= 0; i--) {
    if (runtime_ctx_->GetGenCtx(i)->finish) {
      auto ret = StopRequest(runtime_ctx_->GetGenCtx(i)->request->request_id);
      if (ret != AsStatus::ALLSPARK_SUCCESS) {
        return ret;
      }
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark
