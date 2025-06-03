/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_lifecycle.cpp
 */

#include "model.h"  // NOLINT

#include <common/engine_runtime.h>

#include <sstream>

namespace allspark {

AsModel::AsModel(const std::string& model_type)
    : model_type_(model_type), ctx_(nullptr), current_unfinished_request_(0) {
  gen_ctx_model_ = std::make_unique<GenerateContext>();
  runtime_ctx_ = std::make_unique<RuntimeContext>();

  // Pre-allocate enough request slots for the common serving case.
  all_request_map_.reserve(1000);
}

AsTensor AsModel::GetOutputTensor(std::string tensor_name) {
  DLOG(INFO) << "AsModel::GetOutputTensor()" << std::endl;
  return *tensors_[tensor_name];
}

void AsModel::GetInformation(std::string* model_info) {
  DLOG(INFO) << "AsModel::GetInformation()" << std::endl;
  std::stringstream ss;
  ss << "Model Type : " << model_type_ << std::endl;
  ss << "Model Inputs : " << std::endl;
  for (const std::string& t_name : input_names_) {
    ss << "    " << tensors_[t_name]->ToString() << std::endl;
  }
  ss << "Model Outputs:" << std::endl;
  for (const std::string& t_name : output_names_) {
    ss << "    " << tensors_[t_name]->ToString() << std::endl;
  }
  *model_info = ss.str();
}

AsModel::~AsModel() = default;

}  // namespace allspark
