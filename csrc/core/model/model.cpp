/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    model.cpp
 */

#include "model.h"  // NOLINT

namespace allspark {

ModelFactory& ModelFactory::getInstance() {
  static ModelFactory model_factory;
  return model_factory;
}

ModelConstructor ModelFactory::GetModel(const std::string& model_type_str) {
  if (model_set_.find(model_type_str) == model_set_.end()) {
    LOG(ERROR) << "Unsupported model type : " << model_type_str << std::endl;
    throw AsException("Unsupported model type");
  }
  return model_set_[model_type_str];
}

void ModelFactory::Register(const std::string& model_type_str,
                            ModelConstructor model_constructor) {
  model_set_[model_type_str] = model_constructor;
}

}  // namespace allspark
