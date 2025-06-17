/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_request.cpp
 */

#include "model.h"  // NOLINT
#include "model_internal.h"  // NOLINT

#include <common/device_context.h>
#include <common/engine_runtime.h>
#include <common/extra_embedding.hpp>

#include <cassert>
#include <chrono>
#include <exception>
#include <queue>
#include <sstream>

#include "runtime/weight/weight_manager_lora.h"

namespace allspark {

namespace {

bool StopPendingRequest(
    const std::string& request_id,
    std::queue<std::shared_ptr<Request>>& pending_request_queue) {
  std::queue<std::shared_ptr<Request>> remaining_requests;
  bool found_request = false;
  while (!pending_request_queue.empty()) {
    std::shared_ptr<Request> request = pending_request_queue.front();
    if (request->request_id == request_id) {
      found_request = true;
      request->status = AsEngine::GenerateRequestStatus::GenerateFinished;
      LOG(INFO) << "Request " << request_id << " stop before running";
    } else {
      remaining_requests.push(request);
    }
    pending_request_queue.pop();
  }
  pending_request_queue = std::move(remaining_requests);
  return found_request;
}

}  // namespace

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

AsStatus AsModel::StopRequest(const std::string& request_id) {
  if (StopPendingRequest(request_id, pending_request_queue_)) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  int request_idx = -1;
  for (int i = runtime_ctx_->GetGenCtxListSize() - 1; i >= 0; i--) {
    if (runtime_ctx_->GetGenCtx(i)->request->request_id == request_id) {
      request_idx = i;
      break;
    }
  }
  if (request_idx == -1) {
    DLOG(ERROR) << "not find running request id:" << request_id
                << ",maybe already stop." << std::endl;
    return AsStatus::ALLSPARK_SUCCESS;
  }
  std::shared_ptr<GenerateContext> gen_ctx =
      runtime_ctx_->GetGenCtx(request_idx);
  std::shared_ptr<Request> request = gen_ctx->request;

  for (int i = 0; i < gen_ctx->k_cache_list.size(); i++) {
    gen_ctx->k_cache_list[i]->Free();
  }
  for (int i = 0; i < gen_ctx->v_cache_list.size(); i++) {
    gen_ctx->v_cache_list[i]->Free();
  }
  DLOG(INFO) << "AsModel::StopRequest: [" << request_id << "] cache released";

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (prefix_cache_manager_ != nullptr) {
      AS_CHECK_STATUS(ExtraEmbeddingUtils::CreateTensorForHash(
          request, request->interim, request->interim, "generated_ids"));
      prefix_cache_manager_->Insert(
          request->interim.at("generated_ids_for_hash"),
          gen_ctx->input_len / ctx_->GetCacheSpanSize() *
              ctx_->GetCacheSpanSize(),
          request->start_ts, gen_ctx->virtual_k_cache->GetLayerCache(),
          gen_ctx->virtual_v_cache->GetLayerCache(),
          gen_ctx->prefix_cache_node_list);

      prefix_cache_manager_->UnRef(gen_ctx->prefix_cache_node_list);

      std::stringstream ss;
      ss << "UnRef request_id: " << request->request_id << ", "
         << "rank: " << ctx_->GetRank();
      prefix_cache_manager_->PrintPrefixCacheInfo(ss.str());
    }

    gen_ctx->virtual_k_cache.reset();
    gen_ctx->virtual_v_cache.reset();

    if (prefix_cache_manager_ != nullptr &&
        isWarmupRequest(request->request_id)) {
      prefix_cache_manager_->EvictAllUnrefered();
    }
  }
#endif
  request->extra_embedding.clear();
  request->interim.clear();

  ctx_->Synchronize();
  runtime_ctx_->FinishRequest(request_idx);
  current_unfinished_request_--;

  if (ctx_->GetRank() == 0) {
    LOG(INFO) << "Stop request with request id: " << request_id;
  }
  if (runtime_ctx_->GetGenCtxListSize() > 0) {
    for (AsOperator* op : topo_ops_) {
      AsStatus status = op->CallReshape(runtime_ctx_.get());
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "reshape failed in topo_ops" << std::endl;
        return ErrorProcess(status);
      }
    }
  }
  using namespace std::chrono;
  request->generate_ts = std::chrono::steady_clock::now();
  auto gen_duration = request->generate_ts - request->context_ts;
  auto ctx_duration = request->context_ts - request->start_ts;
  auto gen_duration_in_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(gen_duration)
          .count();
  float gen_tps =
      (request->generated_len * 1000.0f) / (gen_duration_in_ms + 0.1f);

  auto ctx_dur_ms = duration_cast<milliseconds>(ctx_duration).count();
  float ctx_tps = (request->input_len * 1000.0f) / (ctx_dur_ms + 0.1f);

  if (ctx_->GetRank() == 0) {
    LOG(INFO) << "[" << __FUNCTION__ << "] "
              << "Request ID: " << request->request_id << ", "
              << "Context time(ms): " << ctx_dur_ms << ", "
              << "Generate time(ms): " << gen_duration_in_ms << ", "
              << "Context Length: " << request->input_len << ", "
              << "Generated Length: " << request->generated_len << ", "
              << "Context TPS: " << ctx_tps << ", "
              << "Generate TPS: " << gen_tps << ", "
              << "Prefix Cache Len: " << request->prefix_len;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::ReleaseRequest(const std::string& request_id) {
  std::unique_lock<std::mutex> lock(request_map_lock_);
  if (all_request_map_.find(request_id) != all_request_map_.end()) {
    all_request_map_.erase(request_id);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

Request* AsModel::GetRequestById(const std::string& request_id) {
  std::unique_lock<std::mutex> lock(request_map_lock_);
  if (all_request_map_.find(request_id) == all_request_map_.end()) {
    return nullptr;
  }
  return all_request_map_.at(request_id).get();
}

}  // namespace allspark
