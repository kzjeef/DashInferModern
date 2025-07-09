/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    model.cpp
 */

#include "model.h"  // NOLINT
#include "model_internal.h"  // NOLINT

#include <common/engine_runtime.h>
#include <common/env_config.h>
#include <common/memory_reuser.h>
#include <core/operator/generate_opt/postprocess_id/postprocess_id_op.h>
#include <core/operator/generate_opt/span_attn/span_attn_op.h>
#include <core/operator/generate_opt/mla_attn/mla_attn_op.h>
#include <utility/arbiter.h>
#include <utility/file_util.h>
#include <utility/mem_registry.h>
#include <utility/timer.h>
#include <weight/weight_manager.h>

#include <common/extra_embedding.hpp>
#include <exception>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

#ifdef ENABLE_CUDA
#include <core/kernel/cuda/sample.h>
#include <cuda/cuda_context.h>
#include <curand_kernel.h>
#include <device/cuda/cuda_cache_allocator.h>
#endif
#include <device/memory_func.h>

#include <random>

#include "runtime/weight/weight_manager_lora.h"

namespace allspark {
using std::string;
using std::vector;

/*
#define CHECK_CUDA_ERROR(op) do { \
                ctx_->Synchronize(); \
                cudaError_t r = cudaGetLastError(); \
                if (cudaSuccess != r) { \
                    LOG(ERROR) << "OP ERROR! " << cudaGetErrorString(r) <<
std::endl; \
                    //op->PrintInformation(); \
                } \
            } while (false);
*/

// --------------------------------------------------------------------------
// //

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
