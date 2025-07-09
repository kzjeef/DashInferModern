/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_stats.cpp
 */

#include "model.h"  // NOLINT

#include <common/device_context.h>
#include <common/engine_runtime.h>

#include <iomanip>
#include <sstream>

namespace allspark {

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
        static_cast<float>(as_stat->total_token - as_stat->free_token) /
        static_cast<float>(as_stat->total_token);
    if (prefix_cache_manager_ != nullptr) {
      prefix_cache_manager_->UpdateEngineStat(as_stat);
    }
  } else
#endif
  {
    as_stat->total_token = 0;
    as_stat->free_token = 0;
  }
  as_stat->pendding_request = static_cast<int>(pending_request_queue_.size());
  as_stat->running_request = runtime_ctx_->GetGenCtxListSize();
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
                 << " count: " << static_cast<long>(stat.second[3])
                 << std::setprecision(2) << " min_ms: " << stat.second[0]
                 << " max_ms: " << stat.second[1]
                 << " ave_ms: " << stat.second[2]
                 << " total_ms: " << stat.second[4]
                 << " percentage(%): " << stat.second[5] << std::endl;
      ss << std::setfill(' ') << std::fixed << std::setprecision(2) << std::left
         << std::setw(10) << rank_ << std::left << std::setw(20) << stat.first
         << std::left << std::setw(10) << static_cast<long>(stat.second[3])
         << std::left << std::setw(10) << stat.second[0] << std::left
         << std::setw(10) << stat.second[1] << std::left << std::setw(10)
         << stat.second[2] << std::left << std::setw(15) << stat.second[4]
         << std::left << std::setw(10) << stat.second[5] << std::endl;
    }
    ss << std::setfill('-') << std::setw(95) << "-" << std::endl;
    ss << std::endl;
  }
  return ss.str();
}

}  // namespace allspark
