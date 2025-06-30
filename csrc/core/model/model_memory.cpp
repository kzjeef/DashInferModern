/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_memory.cpp
 */

#include "model.h"  // NOLINT

#include <common/device_context.h>
#include <common/engine_runtime.h>

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

}  // namespace allspark
