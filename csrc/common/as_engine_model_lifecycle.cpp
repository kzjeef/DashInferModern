/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_model_lifecycle.cpp
 */

#include "as_engine_impl.h"
#include "engine_control_message.h"

#include <utility/allspark_logging.h>

#include <cassert>
#include <future>

namespace allspark {

AsStatus AsEngineImpl::StartModel(const char* model_name) {
  util::as_init_log();
  DLOG(INFO) << "[" << model_name << "] AsEngineImpl::StartModel";

  auto name = std::string(model_name);
  as_stat_ = std::make_unique<AsEngineStat>(name);
  model_state_map_[name] = std::make_shared<ModelControlState>(name);
  model_state_map_[name]->StartLoop(&AsEngineImpl::ModelRunningThread, this,
                                    name, model_state_map_[name]);

#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA &&
      device_ctx_->GetCacheSpanSize() == 0) {
    LOG(INFO) << "StartModel: span cache is disabled, skip warm-up";
    return AsStatus::ALLSPARK_SUCCESS;
  }
#endif

  ExpandRankThreadPool();
  return WarmupModel(model_name);
}

void AsEngineImpl::ExpandRankThreadPool() {
  if (nranks_ > threadpool_size_) {
    threadpool_size_ = nranks_ * 2;
    threadpool_ = std::make_unique<ThreadPoolWithID>(threadpool_size_);
  }
}

AsStatus AsEngineImpl::StopModel(const char* model_name) {
  LOG(INFO) << "[" << model_name << "] StopModel";
  auto reply_promise = std::make_shared<std::promise<AsStatus>>();
  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];

  if (model_state->model_stopping.load()) {
    LOG(INFO) << "model is stopping, access denied";
    return AsStatus::ALLSPARK_REQUEST_DENIED;
  }

  model_state->model_stopping = true;
  auto message = EngineControlMessage(EngineControlMessageId::GracefulStopModel,
                                      reply_promise);
  LOG(INFO) << "AsEngineImpl:: send model stop message.";
  if (!model_state->msg_queue.enqueue(std::move(message))) {
    LOG(ERROR) << "push message queue failed.";
    model_state->model_stopping = false;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  LOG(INFO) << "AsEngineImpl:: wait stop model return";
  AsStatus status = reply_promise->get_future().get();
  LOG(INFO) << "AsEngineImpl:: stop model got return.";
  if (status != AsStatus::ALLSPARK_SUCCESS) {
    LOG(ERROR) << "[" << model_name << "] StopModel failed with error "
               << static_cast<int>(status);
    return status;
  }

  LOG(INFO) << "[" << model_name << "] waiting to join loop thread";
  model_state->StopLoop();
  LOG(INFO) << "[" << model_name << "] loop thread joined";
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsEngineImpl::ReleaseModel(const char* model_name) {
  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  if (!model_state->model_stopped) {
    LOG(INFO) << "Model release without calling model stop, "
                 "please call model stop first!!!";
  }
  LOG(INFO) << "Release Model, intent to start a new model "
               "so release all bfc memory";
  workers_.clear();

  {
    std::unique_lock<std::mutex> locker(engine_lock_);
    weight_manager_->FreeWeight(model_state->weight_handler_);
    if (!model_state->model_stopped) {
      LOG(INFO) << "Model thread not stopped, stop thread may hang.";
    }
    model_state->StopLoop();
    model_state_map_.erase(model_name);
  }
  DestroyDeviceContext();
  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark
