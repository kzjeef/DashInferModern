/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_lifecycle.cpp
 */

#include "as_engine_impl.h"

#include <common/allocator.h>
#include <cpu/cpu_context.h>
#include <git_version.h>
#include <utility/allspark_logging.h>

#include <cstdio>
#include <random>

namespace allspark {

AsEngineImpl::AsEngineImpl()
    : device_ctx_(std::make_unique<CPUContext>()),
      is_multi_nodes_(false),
      threadpool_size_(1),
      lora_use_count_(0) {
  util::as_init_log();
  threadpool_ = std::make_unique<ThreadPoolWithID>(threadpool_size_);
  device_ctx_->Init();

  weight_manager_ = WeightManager::Create();
  weight_manager_->RegisterWeightEventListener(
      [&](const std::shared_ptr<ModelWeightHandler>& handler,
          WeightEvent event) {
        if (event == WeightEvent::WeightOnLoad) {
          std::unique_lock<std::mutex> locker(engine_lock_);
          auto model_name = handler->GetModelConfig().model_name;
          if (model_state_map_.count(model_name) > 0) {
            model_state_map_[model_name]->weight_handler_ = handler;
          }
        }
      });

  std::random_device rand_dev;
  random_engine.seed(rand_dev());
  LOG(INFO) << "AllSpark Init with Version: " << GetVersionFull();
}

AsEngineImpl::~AsEngineImpl() {
  LOG(INFO) << "~AsEngine called";
  std::vector<std::string> pending_stop_model;

  {
    std::lock_guard<std::mutex> guard(engine_lock_);
    LOG(INFO) << "model_state_map_ size:" << model_state_map_.size();
    for (auto& model_state : model_state_map_) {
      if (!model_state.second->model_stopped) {
        LOG(INFO) << "Stopping model " << model_state.first;
        pending_stop_model.push_back(model_state.first);
      }
    }
  }

  for (auto& name : pending_stop_model) {
    StopModel(name.c_str());
    ReleaseModel(name.c_str());
  }

  bool destroy_allocator = weight_manager_->GetNumModels() > 0;
  workers_.clear();
  models_.clear();
  weight_manager_.reset();
  if (destroy_allocator) {
    DestroyBFCAllocator();
  }
  LOG(INFO) << "~AsEngineImpl finished.";
}

std::string AsEngineImpl::GetVersionFull() {
  char buffer[256];
  snprintf(buffer, sizeof(buffer), "%s.%s.%s/(GitSha1:%s)",
           ALLSPARK_VERSION_MAJOR, ALLSPARK_VERSION_MINOR,
           ALLSPARK_VERSION_PATCH, kGitHash);
  return std::string(buffer);
}

}  // namespace allspark
