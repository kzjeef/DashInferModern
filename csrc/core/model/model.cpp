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

AsStatus AsModel::runDecoderContext() {
  util::Timer t_begin;

#if PROFILE_CONTEXT_TIME_GPU
  // should skip the warm up request.
  {
    if (runtime_ctx_->GetGenCtxListSize() >= 10) {
      auto cuda_ctx = dynamic_cast<const CUDAContext*>(ctx_);
      if (cuda_ctx) {
        cuda_ctx->Synchronize();
        LOG(INFO) << "NSys Profiler start.";
        cuda_ctx->NsysProfilerStart();
      }
    }
  }
#endif

  runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_step");
  runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_inv_freq");

  std::shared_ptr<GenerateContext> gen_ctx =
      (runtime_ctx_->GetGenCtx(runtime_ctx_->current_batch));
  GenerateConfig gen_cfg = gen_ctx->gen_cfg;

  DLOG(INFO) << "start run context ,uuid = " << gen_ctx->request->request_id
             << " lora_name=" << gen_cfg.lora_name << std::endl;
  int batch_size = 1;
  size_t in_length =
      gen_ctx->request->interim.at("new_input_ids")->GetShape()[1];
  gen_ctx->batch_size = batch_size;
  if (gen_cfg.do_sample && gen_cfg.num_beams == 1) {
    gen_ctx->generate_method = 0;  // sample
  } else {
    gen_ctx->generate_method = 1;  // beam_search
  }
  gen_ctx->num_beams = gen_cfg.num_beams;
  if (in_length > gen_cfg.max_length) {
    LOG(ERROR) << "Error: input length: " << in_length
               << " exceeds generation config's max length: "
               << gen_cfg.max_length << std::endl;
    return ErrorProcess(AsStatus::ALLSPARK_PARAM_ERROR);
  }
  for (auto& graph : graph_ops_) {
    for (auto& op : graph_ops_[graph.first]) {
      op->SetGenerateContext(gen_ctx);
    }
  }
  gen_ctx->only_decoder = true;
  gen_ctx->num_beams = 1;
  gen_ctx->step = gen_ctx->prefix_len;

  // finish pre-graph first to avoid possibly troublesome set shape
  for (auto& op : graph_ops_["pre_graph"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in pre_graph" << std::endl;
      return ErrorProcess(status);
    }
  }

  for (auto& op : graph_ops_["pre_graph"]) {
    AsStatus status = op->CallForward(runtime_ctx_.get());

#if DEBUG_GEN_LAYER_SYNC
    op->Synchronize();
#endif
#if DEBUG_GEN_LAYER
    if (debugCurrentRequest(runtime_ctx_->GetGenCtx(0)->request->request_id)) {
      op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
      DO_ARBITRATE(rank_, nranks_, 0, op);
#endif
    }
#endif

    CHECK_CUDA_ERROR(op)
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in pre_graph" << std::endl;
      return ErrorProcess(status);
    }
  }

  util::Timer t_pre_graph;
  // other op
  for (auto& op : graph_ops_["decoder"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in decoder" << std::endl;
      return ErrorProcess(status);
    }
  }

  util::Timer t_alloc;
  {
    TracerLog trace(ctx_->GetDeviceType(), "ContextAlloc", 1);
#if ENABLE_SPAN_ATTENTION
    if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#ifdef CONFIG_CONCURRENT_SPAN
      auto num_dec_layers = ctx_->GetDecoderLayer();
      std::vector<std::future<AsStatus>> result(num_dec_layers);
      int layer_idx = 0;
      for (auto& op : graph_ops_["decoder"]) {
        if (dynamic_cast<SpanAttnOp*>(op.get()) != nullptr ||
                dynamic_cast<MLAAttnOp*>(op.get()) != nullptr) {
          result[layer_idx++] = layer_threadpool_->enqueue(
              [this, &op]() { return op->CallAlloc(runtime_ctx_.get()); });
        }
      }
      // sanity check
      if (layer_idx != num_dec_layers) {
        LOG(ERROR) << "ContextAlloc: decoder layer number mismatch";
        return ErrorProcess(AsStatus::ALLSPARK_RUNTIME_ERROR);
      }

      for (int i = 0; i < num_dec_layers; ++i) {
        auto status = result[i].get();
        if (status != AsStatus::ALLSPARK_SUCCESS) {
          LOG(ERROR) << "ContextAlloc: alloc failed in loop::decoder";
          return ErrorProcess(status);
        }
      }
#else
      for (auto& op : graph_ops_["decoder"]) {
        if (dynamic_cast<SpanAttnOp*>(op.get()) != nullptr ||
                dynamic_cast<MLAAttnOp*>(op.get()) != nullptr) {
          AsStatus status = op->CallAlloc(runtime_ctx_.get());
          CHECK_CUDA_ERROR(op)
          if (status != AsStatus::ALLSPARK_SUCCESS) {
            LOG(ERROR) << "ContextAlloc: alloc failed in loop::decoder";
            return ErrorProcess(status);
          }
        }
      }
#endif  // CONFIG_CONCURRENT_SPAN
    } else
#endif  // ENABLE_SPAN_ATTENTION
    {
      for (auto& op : graph_ops_["decoder"]) {
        AsStatus status = op->CallAlloc(runtime_ctx_.get());
        CHECK_CUDA_ERROR(op)
        if (status != AsStatus::ALLSPARK_SUCCESS) {
          LOG(ERROR) << "ContextAlloc: alloc failed in loop::decoder";
          return ErrorProcess(status);
        }
      }
    }
  }

  util::Timer t_forward;
  // pre_forward
  // first decoder for input_ids
  for (auto& op : graph_ops_["decoder"]) {
    AsStatus status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER_SYNC
    op->Synchronize();
#endif
#if DEBUG_GEN_LAYER
    if (debugCurrentRequest(runtime_ctx_->GetGenCtx(0)->request->request_id)) {
      op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
      DO_ARBITRATE(rank_, nranks_, 0, op);
#endif
    }
#endif
    CHECK_CUDA_ERROR(op)
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in decoder" << std::endl;
      return ErrorProcess(status);
    }
  }

  util::Timer t_reshape;
  gen_ctx->in_length_bias =
      gen_ctx->prefix_len == 0 ? gen_ctx->input_len : in_length;

  gen_ctx->num_beams = gen_cfg.num_beams;
  for (auto& op : graph_ops_["gen_graph"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in gen_graph" << std::endl;
      return ErrorProcess(status);
    }
  }

  util::Timer t_gen_graph;
  for (auto& op : graph_ops_["gen_graph"]) {
    AsStatus status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER_SYNC
    op->Synchronize();
#endif
#if DEBUG_GEN_LAYER
    if (debugCurrentRequest(runtime_ctx_->GetGenCtx(0)->request->request_id)) {
      op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
      DO_ARBITRATE(rank_, nranks_, 0, op);
#endif
    }
#endif
    CHECK_CUDA_ERROR(op)
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in gen_graph" << std::endl;
      return ErrorProcess(status);
    }
  }

  util::Timer t_post_graph;
  for (auto& op : graph_ops_["post_graph"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in post" << std::endl;
      return ErrorProcess(status);
    }
    status = op->CallForward(runtime_ctx_.get());
    CHECK_CUDA_ERROR(op)
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in post" << std::endl;
      return ErrorProcess(status);
    }
  }
  gen_ctx->in_length_bias = 0;
  gen_ctx->step = in_length + gen_ctx->prefix_len;

  DLOG(INFO) << "end run context ,uuid = " << gen_ctx->request->request_id
             << std::endl;
  util::Timer t_end;

  auto do_time_profile = EnvVarConfig::GetInt("ALLSPARK_TIME_LOG", 0);
  if (do_time_profile) {
    using util::Timer;
    auto pre_graph_time = Timer::duration_ms(t_begin, t_pre_graph);
    auto pre_reshape_time = Timer::duration_ms(t_pre_graph, t_alloc);
    auto alloc_time = Timer::duration_ms(t_alloc, t_forward);
    auto forward_time = Timer::duration_ms(t_forward, t_reshape);
    auto gen_reshape_time = Timer::duration_ms(t_reshape, t_gen_graph);
    auto gen_time = Timer::duration_ms(t_gen_graph, t_post_graph);
    auto post_graph_time = Timer::duration_ms(t_post_graph, t_end);

    auto context_time_ms = t_begin.elapsed();

    LOG(INFO) << "Context Time [TTFT](ms) " << context_time_ms
              << " pre_graph: " << pre_graph_time
              << " pre_reshape: " << pre_reshape_time
              << " allocate: " << alloc_time << " forward: " << forward_time
              << " gen_reshape: " << gen_reshape_time
              << " gen_forward: " << gen_time
              << " post_graph: " << post_graph_time;
  }

#if PROFILE_CONTEXT_TIME_GPU
  {
    if (runtime_ctx_->GetGenCtxListSize() >= 10) {
      auto cuda_ctx = dynamic_cast<const CUDAContext*>(ctx_);
      if (cuda_ctx) {
        cuda_ctx->Synchronize();
        LOG(INFO) << "NSys Profiler Stop.";
        cuda_ctx->NsysProfilerStop();
      }
    }
  }
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::GenerateContinueDecoder() {
  DLOG(INFO) << "AsModel::GenerateContinueDecoder()" << std::endl;
  // maybe use if,each turn only run one context phase
  util::Timer t0;
  std::unique_lock<std::mutex> lock(gen_ctx_lock_);

  util::Timer t1;
  // DLOG(INFO) << "pthread_self: " << (unsigned long)pthread_self();

  DLOG(INFO) << "Decoder: gen ctx list " << runtime_ctx_->GetGenCtxListSize()
             << " pending init " << pending_request_queue_.size()
             << " t1(ms): " << t0.elapsed();
#if PROFILE_GENERATION_TIME_GPU
  if (runtime_ctx_->GetGenCtxListSize() >= PROFILE_GENERATION_TIME_BS) {
    auto cuda_ctx = dynamic_cast<const CUDAContext*>(ctx_);
    if (cuda_ctx) {
      cuda_ctx->Synchronize();
      cuda_ctx->NsysProfilerStart();
    }
  }
#endif

  // while (pending_request_queue_.size() > 0 &&
  //        runtime_ctx_->GetGenCtxListSize() < ctx_->GetModelMaxBatch()) {
  //     StartRequest(pending_request_queue_.front());
  //     pending_request_queue_.pop();
  // }

  current_unfinished_request_.store(pending_request_queue_.size() +
                                    runtime_ctx_->GetGenCtxListSize());
  const int async_token_num = 1;  // TODO gen_cfg.async_token_num
  for (int now_step = 0; now_step < async_token_num; now_step++) {
    int batch_size = runtime_ctx_->GetGenCtxListSize();
    if (batch_size == 0) {
      // LOG(INFO) << "Continue: empty batch size";
      return AsStatus::ALLSPARK_EMPTY_REQUEST;
    }

    util::Timer t2;
    gen_ctx_model_->step++;  // 利用这个废弃的字段，给校对工具使用
    runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_step");
    runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_inv_freq");
    {
      TracerLog trace(ctx_->GetDeviceType(), "DecoderAlloc", 0);
      // do NOT run this concurrently, it reduces performance
#if ENABLE_SPAN_ATTENTION
      if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#ifdef CONFIG_CONCURRENT_SPAN
        auto num_dec_layers = ctx_->GetDecoderLayer();
        std::vector<std::future<AsStatus>> result(num_dec_layers);
        int layer_idx = 0;
        for (auto& op : graph_ops_["decoder"]) {
          if (dynamic_cast<SpanAttnOp*>(op.get()) != nullptr ||
                dynamic_cast<MLAAttnOp*>(op.get()) != nullptr) {
            result[layer_idx++] = layer_threadpool_->enqueue(
                [this, &op]() { return op->CallAlloc(runtime_ctx_.get()); });
          }
        }
        // sanity check
        if (layer_idx != num_dec_layers) {
          LOG(ERROR) << "DecoderAlloc: decoder layer number mismatch";
          return ErrorProcess(AsStatus::ALLSPARK_RUNTIME_ERROR);
        }

        for (int i = 0; i < num_dec_layers; ++i) {
          auto status = result[i].get();
          if (status != AsStatus::ALLSPARK_SUCCESS) {
            LOG(ERROR) << "DecoderAlloc: alloc failed in loop::decoder";
            return ErrorProcess(status);
          }
        }
#else
        for (auto& op : graph_ops_["decoder"]) {
          if (dynamic_cast<SpanAttnOp*>(op.get()) != nullptr ||
                dynamic_cast<MLAAttnOp*>(op.get()) != nullptr) {
            AsStatus status = op->CallAlloc(runtime_ctx_.get());
            CHECK_CUDA_ERROR(op)
            if (status != AsStatus::ALLSPARK_SUCCESS) {
              LOG(ERROR) << "DecoderAlloc: alloc failed in loop::decoder";
              return ErrorProcess(status);
            }
          }
        }
#endif  // CONFIG_CONCURRENT_SPAN
      } else
#endif  // ENABLE_SPAN_ATTENTION
      {
        for (auto& op : graph_ops_["decoder"]) {
          AsStatus status = op->CallAlloc(runtime_ctx_.get());
          CHECK_CUDA_ERROR(op)
          if (status != AsStatus::ALLSPARK_SUCCESS) {
            LOG(ERROR) << "DecoderAlloc: alloc failed in loop::decoder";
            return ErrorProcess(status);
          }
        }
      }
    }

    util::Timer t3;
    {
      TracerLog trace(ctx_->GetDeviceType(), "DecoderForward", 1);
      for (auto& op : graph_ops_["decoder"]) {
        AsStatus status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER_SYNC
        op->Synchronize();
#endif
#if DEBUG_GEN_LAYER
        if (debugCurrentRequest(
                runtime_ctx_->GetGenCtx(0)->request->request_id)) {
          op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
          DO_ARBITRATE(rank_, nranks_, gen_ctx_model_->step, op);
#endif
        }
#endif
        CHECK_CUDA_ERROR(op)
        if (status != AsStatus::ALLSPARK_SUCCESS) {
          LOG(ERROR) << "forward failed in loop::decoder" << std::endl;
          return ErrorProcess(status);
        }
      }
    }

    util::Timer t4;
    for (int i = 0; i < batch_size; i++) {
      runtime_ctx_->GetGenCtx(i)->step += 1;
    }

    DLOG(INFO) << " decoder(ms): " << t0.elapsed();
    for (auto& op : graph_ops_["gen_graph"]) {
      AsStatus status = op->CallReshape(runtime_ctx_.get());
      CHECK_CUDA_ERROR(op)
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in loop::decoder" << std::endl;
        return ErrorProcess(status);
      }
    }

    util::Timer t5;
    for (auto& op : graph_ops_["gen_graph"]) {
      AsStatus status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER_SYNC
      op->Synchronize();
#endif
#if DEBUG_GEN_LAYER
      if (debugCurrentRequest(
              runtime_ctx_->GetGenCtx(0)->request->request_id)) {
        op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
        DO_ARBITRATE(rank_, nranks_, gen_ctx_model_->step, op);
#endif
      }
#endif
      CHECK_CUDA_ERROR(op)
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in loop::gen_graph" << std::endl;
        return ErrorProcess(status);
      }
    }

    util::Timer t6;
    for (auto& op : graph_ops_["post_graph"]) {
      AsStatus status = op->CallReshape(runtime_ctx_.get());
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "reshape failed in post" << std::endl;
        return ErrorProcess(status);
      }
      status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER_SYNC
      op->Synchronize();
#endif
#if DEBUG_GEN_LAYER
      if (debugCurrentRequest(
              runtime_ctx_->GetGenCtx(0)->request->request_id)) {
        op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
        DO_ARBITRATE(rank_, nranks_, gen_ctx_model_->step, op);
#endif
      }
#endif
      CHECK_CUDA_ERROR(op)
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in post" << std::endl;
        return ErrorProcess(status);
      }
    }

    // clean up the finished request
    for (int i = runtime_ctx_->GetGenCtxListSize() - 1; i >= 0; i--) {
      if (runtime_ctx_->GetGenCtx(i)->finish) {
        auto ret = StopRequest(runtime_ctx_->GetGenCtx(i)->request->request_id);
        if (ret != AsStatus::ALLSPARK_SUCCESS) return ret;
      }
    }
    util::Timer t7;

    long tpot_ms = t0.elapsed();
    DLOG(INFO) << " Decoder Time(TPOT) (ms): " << tpot_ms;
    auto do_time_profile = EnvVarConfig::GetInt("ALLSPARK_TIME_LOG", 0);
    if (do_time_profile) {
      using util::Timer;
      auto lock_time = Timer::duration_ms(t0, t1);
      auto alloc_time = Timer::duration_ms(t1, t3);
      auto forward_time = Timer::duration_ms(t3, t4);
      auto reshape_time = Timer::duration_ms(t4, t5);
      auto gen_forward_time = Timer::duration_ms(t5, t6);
      auto post_forward_time = Timer::duration_ms(t6, t7);

      LOG(INFO) << "Decoder Loop Time [TPOT] (ms): " << tpot_ms
                << " running: " << runtime_ctx_->GetGenCtxListSize()
                << " lock time(ms):" << lock_time << " alloc: " << alloc_time
                << " forward_time: " << forward_time
                << " reshape: " << reshape_time
                << " gen_frd: " << gen_forward_time
                << " post_gen: " << post_forward_time;
    }

#if PROFILE_GENERATION_TIME_GPU
    if (runtime_ctx_->GetGenCtxListSize() >= PROFILE_GENERATION_TIME_BS) {
      auto cuda_ctx = dynamic_cast<const CUDAContext*>(ctx_);
      if (cuda_ctx) {
        cuda_ctx->Synchronize();
        cuda_ctx->NsysProfilerStop();
      }
    }
#endif
  }

  return AsStatus::ALLSPARK_STREAMING;
}

// XXX: This function will be called by *all* worker thread
// the handle and gen cfg passed from main loop thread,
// so *any* write operation (none const operation is not allowed!)
AsStatus AsModel::StartRequestImpl(
    const std::shared_ptr<RequestHandle> request_handle,
    const std::string request_id, TensorMap* outputs,
    const GenerateConfig& gen_cfg) {
  DLOG(INFO) << "AsModel::StartRequestImpl()" << std::endl;
  std::shared_ptr<Request> request_ptr = std::make_shared<Request>(
      request_id, *request_handle->inputs_internal, *outputs, gen_cfg);
  request_ptr->input_len = request_ptr->inputs.at("input_ids")->GetShape()[1];
  request_ptr->origin_len = request_ptr->input_len;
  request_ptr->extra_embedding = request_handle->mm_embedding_internal;
  request_ptr->enqueue_ts = request_handle->create_ts;
#ifdef ENABLE_JSON_MODE
  if (gen_cfg.response_format.count("type")) {
    try {
      if (gen_cfg.response_format.at("type") == "json_object") {
        request_ptr->format_enforcer = request_handle->format_enforcer;
      }
    } catch (const std::out_of_range& ex) {
      // not found response format, ignore.
    }
  }
#endif
  DLOG(INFO) << "AsModel::StartRequestImpl(): input length:"
             << request_ptr->input_len;

  std::unique_lock<std::mutex> lock(request_map_lock_);
  pending_request_queue_.push(request_ptr);

  all_request_map_[request_id] = request_ptr;
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AsModel::GenerateContinue() {
  AsStatus ret = GenerateContinueDecoder();
  AS_CHECK_STATUS(ret);
  return ret;
  // return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AsModel::AllocDecoderMemory() {
  std::unique_lock<std::mutex> lock(gen_ctx_lock_);
  const int async_token_num = 1;  // TODO gen_cfg.async_token_num
  runtime_ctx_->is_context = false;
  runtime_ctx_->current_batch = 0;
#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (cache_frame_manager_) {
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
      if (decoder_frame > cache_frame_manager_->CountFreeFrame()) {
        if (cache_frame_manager_ && prefix_cache_manager_ != nullptr) {
          LOG(INFO) << "Not enough frame for decoder, "
                    << "need frame vs free frame: " << decoder_frame << " / "
                    << cache_frame_manager_->CountFreeFrame()
                    << ", swap unrefered prefix cache to cpu memory";
          prefix_cache_manager_->EvictUnrefered(decoder_frame);
        }
      }
      // after release all preifx cache still not enough frame
      if (decoder_frame > cache_frame_manager_->CountFreeFrame()) {
        LOG(ERROR) << "free span frame not enough for decoder: "
                   << decoder_frame << " vs "
                   << cache_frame_manager_->CountFreeFrame();
        throw AsException("ALLSPARK_MEMORY_ERROR");
      }
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
  // sgmv op在load、unload时仍有cuda mem小幅波动，多留些余量以免波动出OOM
  // 波动的原因与BFC释放和重新回收mem有关
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

// recommended batch size is never used in practice, so just disable it
#if 0
  // compute best batch size
  int maxlen = ctx_->GetModelMaxLength();
  const char* env_outlen = std::getenv("ALLSPARK_EXPECT_OUTLEN");
  if (env_outlen != nullptr) {
    int outlen = std::atoi(env_outlen);
    if (outlen > ctx_->GetModelMaxLength() || outlen < 0) {
      LOG(ERROR) << "AsModel::Warmup: invalid ALLSPARK_EXPECT_OUTLEN=" << outlen
                 << ", should be a non-negative integer no larger than "
                    "model max length "
                 << ctx_->GetModelMaxLength();
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }

    // assert: 0 <= outlen <= maxlen
    if (outlen > 0) {
      LOG(INFO) << "warm-up: using ALLSPARK_EXPECT_OUTLEN=" << outlen;
      maxlen = outlen;
    } else {
      LOG(INFO) << "warm-up: ALLSPARK_EXPECT_OUTLEN=0, use model max length "
                << ctx_->GetModelMaxLength();
    }
  } else {
    LOG(INFO) << "warm-up: envariable ALLSPARK_EXPECT_OUTLEN not found, "
                 "use model max length "
              << ctx_->GetModelMaxLength();
  }

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
  const size_t best_batch_size_z =
      cache_frame_manager_->CountFrame() /
      (size_t(2) * ctx_->GetDecoderLayer() *
       (maxlen + ctx_->GetCacheSpanSize() - 1) / ctx_->GetCacheSpanSize());
  if (best_batch_size_z > std::numeric_limits<int>::max()) {
    LOG(ERROR) << "AsModel::Warmup: best_batch_size exceeds int max, got "
               << best_batch_size_z;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  const int best_batch_size = static_cast<int>(best_batch_size_z);
  LOG(INFO) << "warm-up: recommended batch size is " << best_batch_size
            << ", current model max batch is " << ctx_->GetModelMaxBatch();
  }
#endif  // ENABLE_SPAN_ATTENTION
#endif

  return AsStatus::ALLSPARK_SUCCESS;
}

int64_t AsModel::GetAvailableMemoryBytes() {
  int64_t ret{0};
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#if ENABLE_SPAN_ATTENTION
    ret = cache_allocator_->GetDeviceFreeMemory();
    LOG(INFO) << "AsModel: device available memory (MB): " << (ret >> 20);
#else
    LOG(WARNING)
        << "AsModel::GetAvailableMemoryBytes: span attention disabled, "
           "this function will always return 0";
#endif  // ENABLE_SPAN_ATTENTION
  }
  return ret;
}

int64_t AsModel::GetOccupiedMemoryBytes() {
  int64_t ret{0};
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#if ENABLE_SPAN_ATTENTION
    ret = cache_allocator_->GetDeviceUsedMemory();
    LOG(INFO) << "AsModel: device occupied memory (MB): " << (ret >> 20);
#else
    LOG(WARNING) << "AsModel::GetOccupiedMemoryBytes: span attention disabled, "
                    "this function will always return 0";
#endif  // ENABLE_SPAN_ATTENTION
  }
  return ret;
}

int64_t AsModel::GetTotalMemoryBytes() {
  int64_t ret{0};
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#if ENABLE_SPAN_ATTENTION
    ret = cache_allocator_->GetDeviceTotalMemory();
    LOG(INFO) << "AsModel: device total memory (MB): " << (ret >> 20);
#else
    LOG(WARNING) << "AsModel::GetTotalMemoryBytes: span attention disabled, "
                    "this function will always return 0";
#endif  // ENABLE_SPAN_ATTENTION
  }
  return ret;
}

#if ENABLE_SPAN_ATTENTION
int64_t AsModel::GetFreeFrame() {
  return cache_frame_manager_->CountFreeFrame();
}
#endif

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

AsStatus AsModel::LoadLoraByName(const std::string& lora_name_or_path) {
  DLOG(INFO) << "AsModel::LoadLoraByName() " << lora_name_or_path << std::endl;
  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  // check if lora already exists
  assert(lora_manager_ !=
         nullptr);  // 必须AsModel::Init()之后才能调用LoadLoraByName
  AsModelConfig lora_cfg =
      weight_handler_->GetModelConfig();  // copy cfg from base-model

  if (lora_manager_->GetNumLoras() >= lora_cfg.lora_max_num) {
    LOG(ERROR) << "lora number exceeds limit: " << lora_cfg.lora_max_num;
    return AsStatus::ALLSPARK_LORA_NUM_EXCEED_LIMIT_ERROR;
  }
  auto lora_path_obj = util::Path(lora_cfg.weights_path);
  auto lora_dir = lora_path_obj.parent_path();
  auto lora_name = lora_name_or_path;
  auto lora_path = lora_dir + '/' + lora_name + ".aslora";
  if (lora_name_or_path.front() == '/') {  // only absolute path or name is
                                           // allowed, relative-path not allowed
    lora_path_obj = util::Path(lora_name_or_path);
    lora_path = lora_name_or_path;
    assert(lora_path_obj.extension() == ".aslora");
    lora_name = lora_path_obj.filename().substr(
        0, lora_path_obj.filename().find(".aslora"));
  }
  // lora_path, lora_name OK

  if (lora_manager_->IsLoraExists(lora_name)) {
    LOG(WARNING) << "lora " << lora_name << " already exists!";
    return ret;
  }
  // load lora
  lora_cfg.model_name = lora_name;
  lora_cfg.weights_path = lora_path;
  lora_cfg.model_path = "";  // lora不使用该字段, (no graph for lora)
  lora_cfg.is_lora_cfg = true;
  lora_cfg.lora_names.clear();  // lora should NOT have any sub-loras...
  auto& lora_weight_handle = lora_manager_->RegisterLora(lora_cfg);
  WeightSwapConfig swap_config;
  swap_config.enable =
      false;  // 由调用方来显式load_lora/unload_lora，所以对于lora禁用swap，来提升加载速度
  lora_manager_->SetSwapConfig(lora_weight_handle, swap_config);
  RankInfo rank_info = GetRankInfo();
  ret = lora_manager_->LoadWeightForModel(*ctx_, lora_weight_handle, rank_info);
  if (ret != AsStatus::ALLSPARK_SUCCESS)
    lora_manager_->UnRegisterLora(lora_name);  // rollback
  return ret;
}

AsStatus AsModel::UnloadLoraByName(const std::string& lora_name) {
  DLOG(INFO) << "AsModel::UnloadLoraByName()" << std::endl;
  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  // check if lora already exists
  assert(lora_manager_ !=
         nullptr);  // 必须AsModel::Init()之后才能调用LoadLoraByName
  if (!lora_manager_->IsLoraExists(lora_name)) {
    LOG(WARNING) << "lora " << lora_name << " not exists!";
    return ret;
  }

  lora_manager_->UnRegisterLora(lora_name);
  // set lora tainted, currently only useful for GemmLora op
  for (auto& op : topo_ops_) {
    op->AddTaintedStatus(lora_name);
  }

  return ret;
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
