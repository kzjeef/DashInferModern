/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_init.cpp
 */

#include "model.h"  // NOLINT

#include <common/engine_runtime.h>
#include <common/env_config.h>
#include <common/memory_reuser.h>
#include <weight/weight_manager.h>

#include <algorithm>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <device/cuda/cuda_cache_allocator.h>
#endif

#include "runtime/weight/weight_manager_lora.h"

namespace allspark {

void AsModel::ChangeGemmOpType(OpRegistType& op_type) {
  if (op_type.op_type_str == "GemmA8W8" &&
      ctx_->GetDeviceType() == DeviceType::CUDA) {
    op_type.op_type_str = "GemmSparseA8W8";
  }
}

AsStatus AsModel::Init(const TransformerProto& model_proto,
                       const DeviceContext& ctx) {
  DLOG(INFO) << "AsModel::Init()" << std::endl;

  std::unique_lock<std::mutex> lock(gen_ctx_lock_);

  ctx_ = &ctx;
  DeviceType device_type = ctx.GetDeviceType();
  auto do_profile = EnvVarConfig::GetString("AS_PROFILE", "OFF");
  if (do_profile == "ON") {
    model_profiler_ = std::make_shared<ModelProfiler>(this);
  } else {
    model_profiler_ = nullptr;
  }

  auto rankInfo = this->GetRankInfo();
  AS_CHECK_STATUS(
      weight_manager_->LoadWeightForModel(ctx, weight_handler_, rankInfo));

  auto& model_cfg = weight_handler_->GetModelConfig();
  lora_manager_ = LoraManager::Create(model_cfg.lora_max_num, rankInfo);
  if (model_cfg.lora_names.size() > 0) {
    LOG(WARNING) << "Config item 'lora_names' is not any longer supported "
                    "and ignored!";
  }

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#ifdef CONFIG_CONCURRENT_SPAN
    int num_pool_threads =
        std::max(std::min(std::thread::hardware_concurrency() / ctx.GetNranks(),
                          static_cast<uint32_t>(ctx.GetDecoderLayer())),
                 1U);
    LOG(INFO) << "Model init: size of context thread pool: "
              << num_pool_threads;
    layer_threadpool_ = std::make_unique<ThreadPool>(num_pool_threads);
#endif

    tokens_per_cache_span_ = ctx.GetCacheSpanSize();
    if (tokens_per_cache_span_ > 0) {
      int num_cache_heads = ctx.GetNumberGroups() > 0 ? ctx.GetNumberGroups()
                                                      : ctx.GetNumberHeads();

#ifdef ENABLE_CUDA
      if (device_type == DeviceType::CUDA) {
        cache_allocator_ = std::make_shared<CudaCacheAllocator>(ctx_);
      }
#endif

#ifdef CONFIG_CONCURRENT_SPAN
      if (ctx.GetCacheSpanNumGrow() != 0) {
        LOG(WARNING) << "WARNING: using ConcurrentCacheFrameManager, "
                        "cache_span_num_grow is ignored";
      }
      cache_frame_manager_ = std::make_shared<ConcurrentCacheFrameManager>(
          device_type, ctx.GetCacheSpanNumInit());
      cache_span_manager_ =
          std::make_shared<ConcurrentCacheSpanManager>(cache_frame_manager_);
#else
      cache_frame_manager_ = std::make_shared<DefaultCacheFrameManager>(
          device_type, ctx.GetCacheSpanNumInit(), ctx.GetCacheSpanNumGrow());
      cache_span_manager_ =
          std::make_shared<DefaultCacheSpanManager>(cache_frame_manager_);
#endif

      if (num_cache_heads % rankInfo.rank_size != 0) {
        LOG(ERROR) << "AsModel::Init: head number should be a multiple of "
                   << "nranks, head number: " << num_cache_heads
                   << ", nranks: " << rankInfo.rank_size;
        return AsStatus::ALLSPARK_PARAM_ERROR;
      }

      size_t span_bytes_z = CacheUtils::GetSpanSizeInBytes(
          *(ctx.GetCacheConfig()), ctx.GetDtype(),
          num_cache_heads / rankInfo.rank_size, ctx.GetSizePerHead());
      if (span_bytes_z > std::numeric_limits<int64_t>::max()) {
        LOG(ERROR) << "AsModel::Init: span size in bytes exceeds int64_t, got "
                   << span_bytes_z;
        return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
      }

      cache_span_manager_->Init(static_cast<int64_t>(span_bytes_z));
      LOG(INFO) << "AsModel: tokens per cache span: " << tokens_per_cache_span_
                << ", init spans: " << ctx.GetCacheSpanNumInit()
                << ", grow spans: " << ctx.GetCacheSpanNumGrow();
    }
  }
#endif

  for (auto& t : model_proto.inputs()) {
    tensors_.insert(
        std::make_pair(t.name(), std::make_unique<AsTensor>(t, device_type)));
    input_names_.emplace_back(t.name());
  }

  for (auto& t : model_proto.outputs()) {
    tensors_.insert(
        std::make_pair(t.name(), std::make_unique<AsTensor>(t, device_type)));
    output_names_.emplace_back(t.name());
  }

  tensors_.insert(std::make_pair(
      "workspace", std::make_unique<AsTensor>("workspace", ctx.GetDeviceType(),
                                              DataType::INT8)));
  gen_ctx_model_ = std::make_unique<GenerateContext>();
  runtime_ctx_ = std::make_unique<RuntimeContext>();

  std::unique_ptr<AsTensor> rotary_step = std::make_unique<AsTensor>(
      "rotary_step", ctx_->GetDeviceType(), DataType::INT32);
  rotary_step->SetShape(Shape{ctx_->GetModelMaxBatch()});
  std::unique_ptr<AsTensor> rotary_inv_freq = std::make_unique<AsTensor>(
      "rotary_inv_freq", ctx_->GetDeviceType(), DataType::FLOAT32);
  rotary_inv_freq->SetShape(
      Shape{ctx_->GetModelMaxBatch() * ctx_->GetSizePerHead() / 2});

  runtime_ctx_->CreateLayerCacheManager();
  runtime_ctx_->GetLayerCacheManager()->CreateCache("rotary_step",
                                                    std::move(rotary_step));
  runtime_ctx_->GetLayerCacheManager()->CreateCache("rotary_inv_freq",
                                                    std::move(rotary_inv_freq));

  auto& graph = model_proto.graphs();
  int nodes = 0;

  DLOG(INFO) << "Start process model graph.";
  for (auto& g_name : model_proto.graph_names()) {
    std::vector<std::unique_ptr<AsOperator>> ops;

    for (auto& op_proto : graph.at(g_name).ops()) {
      OpRegistType op_type(op_proto.op_type(), ctx.GetDeviceType());
      if (ctx_->GetSparsityMatmulMode()) {
        ChangeGemmOpType(op_type);
      }

      std::unique_ptr<AsOperator> op =
          OpFactory::getInstance().GetOperator(op_type)();

      AS_CHECK_STATUS(op->CallInit(op_proto, ctx, weight_manager_,
                                   weight_handler_, lora_manager_, rankInfo,
                                   &tensors_, model_profiler_.get()));

      op->SetEmbeddingMap(&embedding_);
      ops.emplace_back(std::move(op));
      nodes += 1;
    }
    graph_ops_.insert(std::make_pair(g_name, std::move(ops)));
  }

  std::vector<std::vector<AsTensor*>> topo_tensors;
  topo_tensors.resize(nodes + 2);
  int topo_size = 0;
  for (auto& g_name : model_proto.graph_names()) {
    for (auto& op : graph_ops_[g_name]) {
      for (const std::string& name : op->GetInNames()) {
        topo_tensors[topo_size].emplace_back(tensors_[name].get());
      }
      for (const std::string& name : op->GetOutNames()) {
        topo_tensors[topo_size].emplace_back(tensors_[name].get());
      }
      topo_size += 1;
    }
  }
  MemoryReuser memory_reuser;
  memory_reuser.binding_with_algo_0(topo_tensors,
                                    const_cast<DeviceContext*>(ctx_));
  tensors_["attention_mask"]->SetShape(Shape{1, ctx_->GetModelMaxLength()});
  tensors_.insert(std::make_pair(
      "context_k_workspace",
      std::make_unique<AsTensor>("context_k_workspace", ctx.GetDeviceType(),
                                 DataType::INT8)));
  tensors_.insert(std::make_pair(
      "context_v_workspace",
      std::make_unique<AsTensor>("context_v_workspace", ctx.GetDeviceType(),
                                 DataType::INT8)));

#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    tensors_.insert(std::make_pair(
        "cublas_workspace",
        std::make_unique<AsTensor>("cublas_workspace", ctx.GetDeviceType(),
                                   DataType::INT8)));
  }
#endif

  const size_t kv_ws_bytes = ctx.GetModelMaxLength() * ctx.GetNumberHeads() *
                             ctx.GetSizePerHead() * SizeofType(ctx.GetDtype()) /
                             rankInfo.rank_size;
  if (kv_ws_bytes > std::numeric_limits<dim_t>::max()) {
    LOG(ERROR) << "AsModel::Init: context KV workspace size exceeds dim_t";
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  tensors_["context_k_workspace"]->SetShape(
      Shape{static_cast<dim_t>(kv_ws_bytes)});
  tensors_["context_v_workspace"]->SetShape(
      Shape{static_cast<dim_t>(kv_ws_bytes)});

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    model_cfg = weight_handler_->GetModelConfig();
    if (model_cfg.enable_prefix_cache) {
      prefix_cache_manager_ = std::make_shared<PrefixCacheManager>(
          cache_span_manager_, cache_frame_manager_, prefix_cache_coordinator_,
          &tensors_, ctx_, model_cfg.prefix_cache_ttl);
    }
  }
#endif

#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, device_id);
    size_t cu_ws_bytes = 8 * 1024 * 1024;
    if (dprop.major >= 9) {
      cu_ws_bytes = 64 * 1024 * 1024;
    }
    tensors_["cublas_workspace"]->SetShape(
        Shape{static_cast<dim_t>(cu_ws_bytes)});
  }
#endif

  DLOG(INFO) << "load model success.";
  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark
