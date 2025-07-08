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

void AsModel::UpdateAsEngineStat(AsEngineStat* as_stat) {
#if ENABLE_SPAN_ATTENTION
  if (cache_span_manager_ && cache_frame_manager_) {
    as_stat->span_size = cache_span_manager_->GetSpanSize();
    as_stat->total_span = cache_frame_manager_->CountFrame();
    as_stat->free_span = cache_frame_manager_->CountFreeFrame();
    as_stat->total_token = as_stat->total_span / (2 * ctx_->GetDecoderLayer()) *
                           ctx_->GetCacheSpanSize();
    as_stat->free_token = as_stat->free_span / (2 * ctx_->GetDecoderLayer()) *
                          ctx_->GetCacheSpanSize();
    as_stat->used_span = as_stat->total_span - as_stat->free_span;
    as_stat->token_usage_percentage =
        static_cast<float>(((as_stat->total_token - as_stat->free_token))) /
        (float)as_stat->total_token;
    if (prefix_cache_manager_ != nullptr) {
      prefix_cache_manager_->UpdateEngineStat(as_stat);
    }
  } else
#endif
  {
    as_stat->total_token = 0;
    as_stat->free_token = 0;
  }
  as_stat->pendding_request = (int)pending_request_queue_.size();
  as_stat->running_request = (int)runtime_ctx_->GetGenCtxListSize();
}

std::string AsModel::GetOpProfilingInfo() {
  if (model_profiler_ == nullptr) {
    LOG(WARNING) << "AS_PROFILE env variable should be set to do profile, "
                    "export AS_PROFILE=ON";
    return {""};
  }
  std::stringstream ss;
  constexpr const char* tags[] = {"forward", "reshape", "alloc"};
  for (auto& tag : tags) {
    ss << "*** " << tag << " ***" << std::endl;
    auto res_stat = model_profiler_->ReportOpStat(tag);
    DLOG(INFO) << "res_stat size: " << res_stat.size() << std::endl;
    ss << std::setfill('-') << std::setw(95) << "-" << std::endl;
    ss << std::setfill(' ') << std::left << std::setw(10) << "rank" << std::left
       << std::setw(20) << "opname" << std::left << std::setw(10) << "count"
       << std::left << std::setw(10) << "min_ms" << std::left << std::setw(10)
       << "max_ms" << std::left << std::setw(10) << "ave_ms" << std::left
       << std::setw(15) << "total_ms" << std::left << std::setw(10)
       << "percentage" << std::endl;
    ss << std::setfill('-') << std::setw(95) << "-" << std::endl;
    for (auto& stat : res_stat) {
      DLOG(INFO) << std::fixed << " rank id: " << rank_
                 << " op name: " << stat.first
                 << " count: " << (long)stat.second[3] << std::setprecision(2)
                 << " min_ms: " << stat.second[0]
                 << " max_ms: " << stat.second[1]
                 << " ave_ms: " << stat.second[2]
                 << " total_ms: " << stat.second[4]
                 << " percentage(%): " << stat.second[5] << std::endl;
      ss << std::setfill(' ') << std::fixed << std::setprecision(2) << std::left
         << std::setw(10) << rank_ << std::left << std::setw(20) << stat.first
         << std::left << std::setw(10) << (long)stat.second[3] << std::left
         << std::setw(10) << stat.second[0] << std::left << std::setw(10)
         << stat.second[1] << std::left << std::setw(10) << stat.second[2]
         << std::left << std::setw(15) << stat.second[4] << std::left
         << std::setw(10) << stat.second[5] << std::endl;
    }
    ss << std::setfill('-') << std::setw(95) << "-" << std::endl;
    ss << std::endl;
  }
  return ss.str();
}
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
