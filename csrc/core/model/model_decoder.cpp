/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_decoder.cpp
 */

#include "model.h"  // NOLINT
#include "model_internal.h"  // NOLINT

#include <common/device_context.h>
#include <common/engine_runtime.h>
#include <common/env_config.h>
#include <core/operator/generate_opt/mla_attn/mla_attn_op.h>
#include <core/operator/generate_opt/span_attn/span_attn_op.h>
#include <utility/arbiter.h>
#include <utility/timer.h>

#include <future>
#include <vector>

#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif

namespace allspark {

AsStatus AsModel::runDecoderContext() {
  util::Timer t_begin;

#if PROFILE_CONTEXT_TIME_GPU
  if (runtime_ctx_->GetGenCtxListSize() >= 10) {
    auto cuda_ctx = dynamic_cast<const CUDAContext*>(ctx_);
    if (cuda_ctx) {
      cuda_ctx->Synchronize();
      LOG(INFO) << "NSys Profiler start.";
      cuda_ctx->NsysProfilerStart();
    }
  }
#endif

  runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_step");
  runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_inv_freq");

  std::shared_ptr<GenerateContext> gen_ctx =
      runtime_ctx_->GetGenCtx(runtime_ctx_->current_batch);
  GenerateConfig gen_cfg = gen_ctx->gen_cfg;

  DLOG(INFO) << "start run context ,uuid = " << gen_ctx->request->request_id
             << " lora_name=" << gen_cfg.lora_name << std::endl;
  int batch_size = 1;
  size_t in_length =
      gen_ctx->request->interim.at("new_input_ids")->GetShape()[1];
  gen_ctx->batch_size = batch_size;
  if (gen_cfg.do_sample && gen_cfg.num_beams == 1) {
    gen_ctx->generate_method = 0;
  } else {
    gen_ctx->generate_method = 1;
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
#endif
    } else
#endif
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
  if (runtime_ctx_->GetGenCtxListSize() >= 10) {
    auto cuda_ctx = dynamic_cast<const CUDAContext*>(ctx_);
    if (cuda_ctx) {
      cuda_ctx->Synchronize();
      LOG(INFO) << "NSys Profiler Stop.";
      cuda_ctx->NsysProfilerStop();
    }
  }
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::GenerateContinueDecoder() {
  DLOG(INFO) << "AsModel::GenerateContinueDecoder()" << std::endl;
  util::Timer t0;
  std::unique_lock<std::mutex> lock(gen_ctx_lock_);

  util::Timer t1;
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

  current_unfinished_request_.store(pending_request_queue_.size() +
                                    runtime_ctx_->GetGenCtxListSize());
  const int async_token_num = 1;
  for (int now_step = 0; now_step < async_token_num; now_step++) {
    int batch_size = runtime_ctx_->GetGenCtxListSize();
    if (batch_size == 0) {
      return AsStatus::ALLSPARK_EMPTY_REQUEST;
    }

    util::Timer t2;
    gen_ctx_model_->step++;
    runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_step");
    runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_inv_freq");
    {
      TracerLog trace(ctx_->GetDeviceType(), "DecoderAlloc", 0);
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
#endif
      } else
#endif
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

    for (int i = runtime_ctx_->GetGenCtxListSize() - 1; i >= 0; i--) {
      if (runtime_ctx_->GetGenCtx(i)->finish) {
        auto ret = StopRequest(runtime_ctx_->GetGenCtx(i)->request->request_id);
        if (ret != AsStatus::ALLSPARK_SUCCESS) {
          return ret;
        }
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

}  // namespace allspark
