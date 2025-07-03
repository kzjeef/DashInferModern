/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_memory.cpp
 */

#include "model.h"  // NOLINT

#include <common/device_context.h>
#include <common/engine_runtime.h>

#include <algorithm>
#include <cmath>

namespace allspark {

AsStatus AsModel::AllocDecoderMemory() {
  std::unique_lock<std::mutex> lock(gen_ctx_lock_);
  runtime_ctx_->is_context = false;
  runtime_ctx_->current_batch = 0;
#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA && cache_frame_manager_) {
    int model_layer = ctx_->GetDecoderLayer();
    int span_size = ctx_->GetCacheSpanSize();
    int new_span_batch = 0;
    for (int i = 0; i < runtime_ctx_->GetGenCtxListSize(); i++) {
      std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx_->GetGenCtx(i);
      size_t length_now = gen_ctx->virtual_k_cache->GetSeqLength(0);
      if (length_now % span_size == 0) {
        new_span_batch += 1;
      }
    }
    int decoder_frame = model_layer * 2 * new_span_batch;
    if (decoder_frame > cache_frame_manager_->CountFreeFrame() &&
        prefix_cache_manager_ != nullptr) {
      LOG(INFO) << "Not enough frame for decoder, "
                << "need frame vs free frame: " << decoder_frame << " / "
                << cache_frame_manager_->CountFreeFrame()
                << ", swap unrefered prefix cache to cpu memory";
      prefix_cache_manager_->EvictUnrefered(decoder_frame);
    }
    if (decoder_frame > cache_frame_manager_->CountFreeFrame()) {
      LOG(ERROR) << "free span frame not enough for decoder: "
                 << decoder_frame << " vs "
                 << cache_frame_manager_->CountFreeFrame();
      throw AsException("ALLSPARK_MEMORY_ERROR");
    }
  }
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::Warmup(int64_t bytes_available, int64_t bytes_runtime) {
  DLOG(INFO) << "AsModel::Warmup()";
  if (bytes_available < 0) {
    LOG(ERROR) << "AsModel::Warmup: bytes_available must be non-negative, got "
               << bytes_available;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  if (bytes_runtime < 0) {
    LOG(ERROR) << "AsModel::Warmup: bytes_runtime must be non-negative, got "
               << bytes_runtime;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  float runtime_mem_ratio = 1.1;
  // LoRA load/unload can temporarily fluctuate while BFC reclaims memory.
  if (ctx_->GetLoraEnabled()) {
    runtime_mem_ratio = 1.5;
  }
  LOG(INFO) << "warm-up: runtime memory reservation ratio: "
            << runtime_mem_ratio;

  const int64_t bytes_cache = std::max(
      0L, bytes_available - static_cast<int64_t>(
                                std::ceil(bytes_runtime * runtime_mem_ratio)));

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    size_t num_to_grow = bytes_cache / cache_frame_manager_->GetFrameSize();
    LOG(INFO) << "warm-up: trying to grow " << num_to_grow
              << " frames, current count of frames: "
              << cache_frame_manager_->CountFrame();
    if (cache_frame_manager_->GrowBy(num_to_grow)) {
      if (prefix_cache_manager_ != nullptr) {
        prefix_cache_manager_->UpdateCapacity();
      }
      LOG(INFO)
          << "warm-up: grow successfully, total number of claimed span frames: "
          << cache_frame_manager_->CountFrame();
    } else {
      LOG(ERROR) << "AsModel::Warmup: failed to grow all " << num_to_grow
                 << " frames, total number of claimed span frames: "
                 << cache_frame_manager_->CountFrame();
      return AsStatus::ALLSPARK_MEMORY_ERROR;
    }
  }
#endif

  return AsStatus::ALLSPARK_SUCCESS;
}

int64_t AsModel::GetAvailableMemoryBytes() {
  int64_t bytes{0};
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#if ENABLE_SPAN_ATTENTION
    bytes = cache_allocator_->GetDeviceFreeMemory();
    LOG(INFO) << "AsModel: device available memory (MB): " << (bytes >> 20);
#else
    LOG(WARNING)
        << "AsModel::GetAvailableMemoryBytes: span attention disabled, "
           "this function will always return 0";
#endif
  }
  return bytes;
}

int64_t AsModel::GetOccupiedMemoryBytes() {
  int64_t bytes{0};
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#if ENABLE_SPAN_ATTENTION
    bytes = cache_allocator_->GetDeviceUsedMemory();
    LOG(INFO) << "AsModel: device occupied memory (MB): " << (bytes >> 20);
#else
    LOG(WARNING) << "AsModel::GetOccupiedMemoryBytes: span attention disabled, "
                    "this function will always return 0";
#endif
  }
  return bytes;
}

int64_t AsModel::GetTotalMemoryBytes() {
  int64_t bytes{0};
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#if ENABLE_SPAN_ATTENTION
    bytes = cache_allocator_->GetDeviceTotalMemory();
    LOG(INFO) << "AsModel: device total memory (MB): " << (bytes >> 20);
#else
    LOG(WARNING) << "AsModel::GetTotalMemoryBytes: span attention disabled, "
                    "this function will always return 0";
#endif
  }
  return bytes;
}

#if ENABLE_SPAN_ATTENTION
int64_t AsModel::GetFreeFrame() {
  return cache_frame_manager_->CountFreeFrame();
}
#endif

}  // namespace allspark
