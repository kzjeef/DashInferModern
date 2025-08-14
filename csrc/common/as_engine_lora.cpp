/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_lora.cpp
 */

#include "as_engine_impl.h"

#include "runtime/weight/weight_manager_lora.h"

#include <cassert>
#include <cstring>
#include <future>

namespace allspark {

AsStatus AsEngineImpl::LoadLoraByName(const char* model_name,
                                      const char* lora_name) {
  std::lock_guard<std::mutex> lora_guard(lora_lock_);
  if (!lora_name || strlen(lora_name) == 0) {
    LOG(ERROR) << "[" << model_name << "] LoadLoraByName: Invalid lora_name";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  DLOG(INFO) << "before load_lora " << lora_name
             << ", free space=" << workers_[0]->GetAvailableMemoryBytes();
  if (workers_[0]->GetModel()->GetLoraManager()->IsLoraExists(lora_name)) {
    LOG(ERROR) << "LoRA " << lora_name << " already loaded, unload it first!";
    return AsStatus::ALLSPARK_LORA_ALREADY_LOADED;
  }
  {
    std::lock_guard<std::mutex> usage_guard(lora_usage_lock_);
    if (loras_in_use_.count(model_name) &&
        loras_in_use_[model_name].count(lora_name)) {
      LOG(ERROR) << "LoRA " << lora_name << " in use, cannot load!";
      return AsStatus::ALLSPARK_LORA_IN_USE;
    }
  }

  assert(model_state_map_[model_name].get() != nullptr);
  if (model_state_map_[model_name]->model_stopping.load()) {
    LOG(INFO) << "model is stopping, access denied";
    return AsStatus::ALLSPARK_REQUEST_DENIED;
  }

  std::vector<std::future<AsStatus>> results(nranks_);
  for (int i = 0; i < nranks_; ++i) {
    results[i] = threadpool_->enqueue(i, [this, i, lora_name]() {
      return workers_[i]->LoadLoraByName(lora_name);
    });
  }
  AsStatus status = AsStatus::ALLSPARK_SUCCESS;
  for (auto& result : results) {
    status = result.get();
    AS_CHECK_STATUS(status);
  }
  LOG(INFO) << "after load_lora " << lora_name
            << ", free space=" << workers_[0]->GetAvailableMemoryBytes();
  workers_[0]->GetModel()->GetLoraManager()->PrintLoras();
  return status;
}

AsStatus AsEngineImpl::UnloadLoraByName(const char* model_name,
                                        const char* lora_name) {
  std::lock_guard<std::mutex> lora_guard(lora_lock_);
  if (!lora_name || strlen(lora_name) == 0) {
    LOG(ERROR) << "[" << model_name << "] UnloadLoraByName: Invalid lora_name";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  LOG(INFO) << "before unload_lora " << lora_name
            << ", free space=" << workers_[0]->GetAvailableMemoryBytes();
  workers_[0]->GetModel()->GetLoraManager()->PrintLoras();
  if (!workers_[0]->GetModel()->GetLoraManager()->IsLoraExists(lora_name)) {
    LOG(ERROR) << "LoRA " << lora_name << " not found, cannot unload!";
    return AsStatus::ALLSPARK_LORA_NOT_FOUND;
  }
  {
    std::lock_guard<std::mutex> usage_guard(lora_usage_lock_);
    if (loras_in_use_.count(model_name) &&
        loras_in_use_[model_name].count(lora_name)) {
      LOG(ERROR) << "LoRA " << lora_name << " in use, cannot unload!";
      return AsStatus::ALLSPARK_LORA_IN_USE;
    }
  }

  assert(model_state_map_[model_name].get() != nullptr);
  if (model_state_map_[model_name]->model_stopping.load()) {
    LOG(INFO) << "model is stopping, access denied";
    return AsStatus::ALLSPARK_REQUEST_DENIED;
  }

  ExpandRankThreadPool();
  std::vector<std::future<AsStatus>> results(nranks_);
  for (int i = 0; i < nranks_; ++i) {
    results[i] = threadpool_->enqueue(i, [this, i, lora_name]() {
      return workers_[i]->UnloadLoraByName(lora_name);
    });
  }
  AsStatus status = AsStatus::ALLSPARK_SUCCESS;
  for (auto& result : results) {
    status = result.get();
    AS_CHECK_STATUS(status);
  }

  workers_[0]->GetModel()->GetLoraManager()->PrintLoras();
  LOG(INFO) << "after unload_lora " << lora_name
            << ", free space=" << workers_[0]->GetAvailableMemoryBytes();
  return status;
}

}  // namespace allspark
