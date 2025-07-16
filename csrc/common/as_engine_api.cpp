/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_api.cpp
 */

#include "as_engine_impl.h"

namespace allspark {

AsEngine::AsEngine() : as_engine_impl_(std::make_unique<AsEngineImpl>()) {}
AsEngine::~AsEngine() = default;

AsStatus AsEngine::BuildModelFromConfigStruct(AsModelConfig& model_config) {
  return as_engine_impl_->BuildModelFromConfigStruct(model_config);
}

AsStatus AsEngine::UnloadModelFromDeviceMemory(const char* model_name) {
  return as_engine_impl_->UnloadModelFromDeviceMemory(model_name);
}

AsStatus AsEngine::ReloadModelToDeviceMemory(const char* model_name) {
  return as_engine_impl_->ReloadModelFromDeviceMemory(model_name);
}

AsStatus AsEngine::GetModelInformation(const char* model_name,
                                       std::string* model_info) {
  return as_engine_impl_->GetModelInformation(model_name, model_info);
}

AsStatus AsEngine::StartModel(const char* model_name) {
  return as_engine_impl_->StartModel(model_name);
}

AsStatus AsEngine::StopModel(const char* model_name) {
  return as_engine_impl_->StopModel(model_name);
}

AsStatus AsEngine::ReleaseModel(const char* model_name) {
  return as_engine_impl_->ReleaseModel(model_name);
}

AsStatus AsEngine::StartRequest(const char* model_name,
                                std::shared_ptr<RequestContent> request_info,
                                RequestHandle** request_handle,
                                ResultQueue** queue) {
  return as_engine_impl_->StartRequest(model_name, request_info, request_handle,
                                       queue);
}

AsStatus AsEngine::StopRequest(const char* model_name,
                               RequestHandle* request_handle) {
  return as_engine_impl_->StopRequest(model_name, request_handle);
}

AsStatus AsEngine::ReleaseRequest(const char* model_name,
                                  RequestHandle* request_handle) {
  return as_engine_impl_->ReleaseRequest(model_name, request_handle);
}

AsStatus AsEngine::SyncRequest(const char* model_name,
                               RequestHandle* request_handle) {
  return as_engine_impl_->SyncRequest(model_name, request_handle);
}

AsStatus AsEngine::LoadLoraByName(const char* model_name,
                                  const char* lora_name) {
  return as_engine_impl_->LoadLoraByName(model_name, lora_name);
}

AsStatus AsEngine::UnloadLoraByName(const char* model_name,
                                    const char* lora_name) {
  return as_engine_impl_->UnloadLoraByName(model_name, lora_name);
}

AsEngineStat AsEngine::GetAsEngineStat(const char* model_name) {
  return as_engine_impl_->GetAsEngineStat(model_name);
}

AsFileInfo AsEngine::GetFileInformation(const char* as_model_path,
                                        const char* as_param_path) {
  return as_engine_impl_->GetFileInformation(as_model_path, as_param_path);
}

std::string AsEngine::GetVersionFull() {
  return as_engine_impl_->GetVersionFull();
}

std::string AsEngine::GetOpProfilingInfo(const char* model_name) {
  return as_engine_impl_->GetOpProfilingInfo(model_name);
}

int AsEngine::GetRankId() { return as_engine_impl_->GetRankId(); }

int AsEngine::GetRankNums() { return as_engine_impl_->GetRankNums(); }

bool AsEngine::IsAllSparkWorkAsService() { return false; }

}  // namespace allspark
