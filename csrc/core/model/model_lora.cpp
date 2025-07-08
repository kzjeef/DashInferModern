/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_lora.cpp
 */

#include "model.h"  // NOLINT

#include <utility/file_util.h>
#include <weight/weight_manager.h>

#include <cassert>

#include "runtime/weight/weight_manager_lora.h"

namespace allspark {

AsStatus AsModel::LoadLoraByName(const std::string& lora_name_or_path) {
  DLOG(INFO) << "AsModel::LoadLoraByName() " << lora_name_or_path << std::endl;
  AsStatus status = AsStatus::ALLSPARK_SUCCESS;
  assert(lora_manager_ != nullptr);
  AsModelConfig lora_cfg = weight_handler_->GetModelConfig();

  if (lora_manager_->GetNumLoras() >= lora_cfg.lora_max_num) {
    LOG(ERROR) << "lora number exceeds limit: " << lora_cfg.lora_max_num;
    return AsStatus::ALLSPARK_LORA_NUM_EXCEED_LIMIT_ERROR;
  }
  auto lora_path_obj = util::Path(lora_cfg.weights_path);
  auto lora_dir = lora_path_obj.parent_path();
  auto lora_name = lora_name_or_path;
  auto lora_path = lora_dir + '/' + lora_name + ".aslora";
  if (lora_name_or_path.front() == '/') {
    lora_path_obj = util::Path(lora_name_or_path);
    lora_path = lora_name_or_path;
    assert(lora_path_obj.extension() == ".aslora");
    lora_name = lora_path_obj.filename().substr(
        0, lora_path_obj.filename().find(".aslora"));
  }

  if (lora_manager_->IsLoraExists(lora_name)) {
    LOG(WARNING) << "lora " << lora_name << " already exists!";
    return status;
  }

  lora_cfg.model_name = lora_name;
  lora_cfg.weights_path = lora_path;
  lora_cfg.model_path = "";
  lora_cfg.is_lora_cfg = true;
  lora_cfg.lora_names.clear();
  auto& lora_weight_handle = lora_manager_->RegisterLora(lora_cfg);
  WeightSwapConfig swap_config;
  swap_config.enable = false;
  lora_manager_->SetSwapConfig(lora_weight_handle, swap_config);
  RankInfo rank_info = GetRankInfo();
  status =
      lora_manager_->LoadWeightForModel(*ctx_, lora_weight_handle, rank_info);
  if (status != AsStatus::ALLSPARK_SUCCESS) {
    lora_manager_->UnRegisterLora(lora_name);
  }
  return status;
}

AsStatus AsModel::UnloadLoraByName(const std::string& lora_name) {
  DLOG(INFO) << "AsModel::UnloadLoraByName()" << std::endl;
  AsStatus status = AsStatus::ALLSPARK_SUCCESS;
  assert(lora_manager_ != nullptr);
  if (!lora_manager_->IsLoraExists(lora_name)) {
    LOG(WARNING) << "lora " << lora_name << " not exists!";
    return status;
  }

  lora_manager_->UnRegisterLora(lora_name);
  for (auto& op : topo_ops_) {
    op->AddTaintedStatus(lora_name);
  }

  return status;
}

}  // namespace allspark
