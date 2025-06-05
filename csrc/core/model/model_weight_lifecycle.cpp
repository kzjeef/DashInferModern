/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_weight_lifecycle.cpp
 */

#include "model.h"  // NOLINT

#include <common/device_context.h>
#include <weight/weight_manager.h>

namespace allspark {

AsStatus AsModel::SaveWeights(std::string* out_allsparkz) {
  DLOG(INFO) << "AsModel::SaveWeights()" << std::endl;

  try {
    weight_manager_->SaveWeights(weight_handler_, out_allsparkz);
  } catch (AsException& e) {
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::ReloadModelToDeviceMemory() {
  DLOG(INFO) << "AsModel::LoadWeightsFromBuffer()" << std::endl;
  weight_manager_->SwapInWeight(weight_handler_, this->GetRankInfo());
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::UnloadModelFromDeviceMemory() {
  DLOG(INFO) << "AsModel::UnloadModelFromDeviceMemory()" << std::endl;
  ctx_->Synchronize();

  graph_ops_.clear();
  tensors_.clear();
  embedding_.clear();
  input_names_.clear();
  output_names_.clear();
  topo_ops_.clear();

  weight_manager_->SwapOutWeight(weight_handler_, this->GetRankInfo());
  const_cast<DeviceContext*>(ctx_)->ResetBlockPools();

  DLOG(INFO) << "AsModel::UnloadModelFromDeviceMemory() END" << std::endl;
  return AsStatus::ALLSPARK_SUCCESS;
}

void AsModel::PrintWeights() {}

}  // namespace allspark
