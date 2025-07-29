/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_model.cpp
 */

#include "as_engine_device.h"
#include "as_engine_impl.h"
#include "thread_utils.h"

#include <common/env_config.h>
#include <cpu/cpu_info.h>
#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <unistd.h>
#include <utility/file_util.h>
#include <utility/mem_registry.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <future>
#include <iomanip>
#include <stdexcept>
#include <tuple>
#include <thread>

#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif

namespace allspark {
namespace {

constexpr int kWarmupInput = 5;

bool ReadProtoFromTextFile(const char* filename,
                           google::protobuf::Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  auto* input = new google::protobuf::io::FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void CheckAndOverridePrefillMode(AsModelConfig& model_config) {
  try {
    DeviceType device_type = DeviceType::CUDA;
    std::vector<int> device_ids;
    std::tie(device_type, device_ids) =
        engine_internal::ParseDeviceType(model_config.compute_unit);

    if (device_type == DeviceType::CPU) {
      if (CPUInfo::SupportAVX512F()) {
        LOG(INFO) << "Detect avx512f supported, switch Prefill mode to flash";
        model_config.prefill_mode = AsMHAPrefill::AsPrefillFlashV2;
      } else if (model_config.prefill_mode != AsMHAPrefill::AsPrefillDefault) {
        LOG(INFO) << "Warn: CPU only support Prefill model default";
        model_config.prefill_mode = AsMHAPrefill::AsPrefillDefault;
      }
    } else if (device_type == DeviceType::CUDA) {
#ifdef ENABLE_CUDA
      int sm_version = CUDAContext::GetStreamProcessorVersion(
          device_ids.empty() ? 0 : device_ids[0]);
      LOG(INFO) << "Auto Prefill selection, CUDA Detected, SM: " << std::hex
                << sm_version;
      if (sm_version >= CUDASMDef::SM_Ampere && CUDA_VERSION >= 11080) {
        model_config.prefill_mode = AsMHAPrefill::AsPrefillFlashV2;
        LOG(INFO) << "Prefill Auto Select: Ampere GPU detected, choose "
                     "flashv2 as prefill flash.";
      } else if (sm_version >= CUDASMDef::SM_Volta) {
        model_config.prefill_mode = AsMHAPrefill::AsPrefillXformer;
        LOG(INFO) << "Prefill Auto Select: Volta GPU detected, choose "
                     "Xformer as prefill flash.";
      }
#endif
    }
  } catch (const std::invalid_argument& error) {
    LOG(INFO) << "Prefill Auto Select got exception, ignore this auto set. "
              << error.what();
  }
}

}  // namespace

AsStatus AsEngineImpl::BuildModelFromConfigStruct(AsModelConfig& model_config) {
  EnvVarConfig env_config;
  CheckAndOverridePrefillMode(model_config);

  DLOG(INFO) << "AsEngineImpl::BuildModelFromConfigStruct()" << std::endl;
  LOG(INFO) << "Build model use following config:\n"
            << model_config.ToString() << std::endl;
  LOG(INFO) << "Memory Info:  BFC_ALLOCATOR: "
            << env_config.GetString("BFC_ALLOCATOR", "default:ON")
            << " BFC_MEM_RATIO: "
            << env_config.GetString("BFC_MEM_RATIO", "default:0.975");

  std::string model_path = model_config.model_path;
  LOG(INFO) << "Load model from : " << model_path << std::endl;
  if (model_path.empty() || !util::IsExists(model_path)) {
    LOG(ERROR) << "No such file or directory : " << model_path << std::endl;
    return AsStatus::ALLSPARK_IO_ERROR;
  }

  AS_CHECK_STATUS(CreateDeviceContext(model_config.compute_unit));
#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA) {
    AS_CHECK_STATUS(setSpanCacheConfig(
        model_config.cache_mode, model_config.cache_span_size,
        model_config.cache_span_num_init, model_config.cache_span_num_grow));
  }
#endif
  device_ctx_->SetPrefillMode(model_config.prefill_mode);
  device_ctx_->SetEvictionStrategy(model_config.eviction_strategy);
  device_ctx_->SetSparsityMatmulMode(model_config.enable_sparsity_matmul);
  device_ctx_->SetSchedulingStrategy(model_config.scheduling_strategy);
  if (model_config.num_threads != 0) {
    AS_CHECK_STATUS(SetNumThreads(model_config.num_threads));
  }
  AS_CHECK_STATUS(SetMatmulPrecision(model_config.matmul_precision));

  engine_max_length_ = model_config.engine_max_length;
  engine_max_batch_ = model_config.engine_max_batch;
  engine_max_prefill_length_ = model_config.engine_max_prefill_length;
  if (engine_max_length_ <= 2) {
    LOG(ERROR) << "Illegal egnine_max_length = " << engine_max_length_;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (engine_max_batch_ <= 0) {
    LOG(ERROR) << "Illegal egnine_max_batch = " << engine_max_batch_;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (engine_max_prefill_length_ < 0 ||
      engine_max_prefill_length_ > engine_max_length_) {
    LOG(ERROR) << "Illegal engine_max_prefill_length = "
               << engine_max_prefill_length_;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA &&
      engine_max_prefill_length_ % device_ctx_->GetCacheSpanSize() != 0) {
    LOG(ERROR) << "Illegal engine_max_prefill_length = "
               << engine_max_prefill_length_
               << "need be a multiple of span_size";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
#endif
  if (engine_max_prefill_length_ > 0 && !model_config.enable_prefix_cache) {
    LOG(ERROR) << "Chunk Prefill only support in enable_prefix_cache=true";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (engine_max_prefill_length_ == 0) {
    LOG(INFO) << "engine_max_prefill_length_ = 0 ,disabled chunk_prefill";
    engine_max_prefill_length_ = engine_max_length_;
  }

  std::shared_ptr<TransformerProto> model_ir =
      std::make_shared<TransformerProto>();
  if (model_config.text_graph) {
    if (!ReadProtoFromTextFile(model_path.c_str(), model_ir.get())) {
      LOG(ERROR) << "Invalid text model format. model_path:" << model_path
                 << std::endl;
      return AsStatus::ALLSPARK_IO_ERROR;
    }
  } else {
    std::ifstream input(model_path);
    if (!model_ir->ParseFromIstream(&input)) {
      LOG(ERROR) << "Invalid binary model format. model_path:" << model_path
                 << std::endl;
      return AsStatus::ALLSPARK_IO_ERROR;
    }
  }

  if (model_config.weights_path.empty()) {
    LOG(ERROR) << "weights path not set , please check input param: "
                  "AsModelConfig: weights_path";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  DLOG(INFO) << "Load weights from : " << model_config.weights_path
             << std::endl;

  std::shared_ptr<ModelWeightHandler> model_weight_handler;
  try {
    model_weight_handler = weight_manager_->RegisterModel(model_config, model_ir);
    weight_manager_->CheckModelConsistency(model_weight_handler);
  } catch (AsModelException&) {
    LOG(ERROR) << "Failed to register model file. " << strerror(errno);
    return AsStatus::ALLSPARK_IO_ERROR;
  }

  std::string model_proto;
  model_ir->SerializeToString(&model_proto);
  std::string& model_name = model_config.model_name;
  if (model_name.empty()) {
    LOG(ERROR) << "model name mot set, please check input param: "
                  "AsModelConfig: model_name";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  model_config.swap_threshold =
      std::min(static_cast<int64_t>(0), model_config.swap_threshold);
  util::SetSwapThreshold(model_config.swap_threshold);

  WeightSwapConfig swap_config;
  swap_config.enable = model_config.swap_threshold >= 0;
  weight_manager_->SetSwapConfig(model_weight_handler, swap_config);
  AS_CHECK_STATUS(
      BuildModel(model_name.c_str(), model_proto, model_weight_handler));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsEngineImpl::BuildModel(
    const char* model_name, const std::string& model_proto,
    std::shared_ptr<ModelWeightHandler> weight_handler,
    const std::map<std::string, int>& model_limits) {
  DLOG(INFO) << "AsEngineImpl::BuildModel()" << std::endl;
  AsModelConfig model_config = weight_handler->GetModelConfig();
  std::unique_ptr<TransformerProto> model_ir =
      std::make_unique<TransformerProto>();
  model_ir->ParseFromString(model_proto);

  auto& graph = model_ir->graphs();
  device_ctx_->SetLoraEnabled(false);
  for (auto& graph_name : model_ir->graph_names()) {
    for (auto& op_proto : graph.at(graph_name).ops()) {
      if (op_proto.op_type() == "GemmLoraCapsule") {
        device_ctx_->SetLoraEnabled(true);
        break;
      }
    }
  }
  if (device_ctx_->GetLoraEnabled()) {
    LOG(INFO) << "lora enabled";
  }

  device_ctx_->SetNumberHeads(model_ir->model_conf().num_heads());
  device_ctx_->SetNumberGroups(model_ir->model_conf().multi_query_group_num());
  device_ctx_->SetSizePerHead(model_ir->model_conf().size_per_head());
  device_ctx_->SetIntermediateSize(model_ir->model_conf().intermediate_size());
  device_ctx_->SetDecoderLayer(model_ir->model_conf().dec_layer());
  device_ctx_->SetDtype(model_ir->model_conf().dtype());
  device_ctx_->SetLoraMaxNum(model_config.lora_max_num);
  device_ctx_->SetLoraMaxRank(model_config.lora_max_rank);
  for (const auto& item : model_limits) {
    if (item.second < 0) {
      LOG(ERROR) << "invalid engine limit param, should >= 0" << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    if (item.first == "engine_max_length") {
      engine_max_length_ = item.second;
    }
    if (item.first == "engine_max_batch") {
      engine_max_batch_ = item.second;
    }
    if (item.first == "swap_threshold") {
      util::SetSwapThreshold(item.second);
    }
  }

  const char* cache_size = std::getenv("ALLSPARK_KVCACHE_ALLOC_SIZE");
  if (cache_size == nullptr) {
    device_ctx_->SetKVcacheSize(engine_max_length_);
  } else {
    int kv_size = std::atoi(cache_size);
    if (kv_size > engine_max_length_) {
      LOG(ERROR) << "invalid ALLSPARK_KVCACHE_ALLOC_SIZE = " << kv_size
                 << ", should <= engine_max_length" << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    device_ctx_->SetKVcacheSize(kv_size == -1 ? engine_max_length_ : kv_size);
  }

  const char* torch_sample = std::getenv("ALLSPARK_USE_TORCH_SAMPLE");
  device_ctx_->SetUseTorchSample(torch_sample != nullptr &&
                                 std::atoi(torch_sample) != 0);
  device_ctx_->SetModelMaxLength(engine_max_length_);
  device_ctx_->SetModelMaxBatch(engine_max_batch_);
  device_ctx_->SetModelMaxPrefillLength(engine_max_prefill_length_);

#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA &&
      device_ctx_->GetCacheSpanSize() != 0 &&
      device_ctx_->GetCacheSpanNumInit() == 0 &&
      device_ctx_->GetCacheSpanNumGrow() == 0) {
    LOG(INFO) << "BuildModel: using adaptive cache span settings";
    constexpr int kv_cache_count = 2;
    int warmup_single_batch_spans =
        (device_ctx_->GetModelMaxLength() +
         device_ctx_->GetCacheSpanSize() - 1) /
            device_ctx_->GetCacheSpanSize() +
        1;
    int multi_batch_tokens =
        kWarmupInput +
        (device_ctx_->GetModelMaxBatch() /
             (device_ctx_->GetModelMaxPrefillLength() / kWarmupInput) +
         5);
    if (device_ctx_->GetSchedulingStrategy() !=
        AsSchedulingStrategy::ContextPriority) {
      multi_batch_tokens += device_ctx_->GetModelMaxBatch();
    }

    int warmup_multi_batch_spans =
        (((multi_batch_tokens + 32) + device_ctx_->GetCacheSpanSize() - 1) /
             device_ctx_->GetCacheSpanSize() +
         1) *
        device_ctx_->GetModelMaxBatch();
    int num_spans_per_seq =
        std::max(warmup_single_batch_spans, warmup_multi_batch_spans);
    int num_spans = kv_cache_count * device_ctx_->GetDecoderLayer() *
                    (num_spans_per_seq + 1);
    AS_CHECK_STATUS(setSpanCacheConfig(device_ctx_->GetCacheMode(),
                                       device_ctx_->GetCacheSpanSize(),
                                       num_spans, 0));
    use_adaptive_cache_ = true;
  }
#endif

  LOG(INFO) << "Start BuildModel";
  ExpandRankThreadPool();
  std::vector<std::thread> worker_threads(nranks_);
  std::vector<std::promise<AsStatus>> results(nranks_);
#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA &&
      model_config.enable_prefix_cache) {
    prefix_cache_coordinator_ =
        std::make_shared<PrefixCacheCoordinator>(nranks_);
  }
#endif

  for (int i = 0; i < nranks_; ++i) {
    worker_threads[i] = std::thread([&, i]() {
      setThreadName(i, "ModelBuildThread");
      try {
        LOG(INFO) << "Start Build model for rank: " << i;
        AsStatus status = workers_[i]->BuildModel(
            *model_ir, weight_manager_, weight_handler, device_ctx_.get(),
            prefix_cache_coordinator_);
        LOG(INFO) << "Finish Build model for rank: " << i;
        results[i].set_value(status);
      } catch (const std::exception&) {
        results[i].set_exception(std::current_exception());
      }
    });
  }

  AsStatus build_status = AsStatus::ALLSPARK_SUCCESS;
  for (int i = 0; i < nranks_; ++i) {
    try {
      AsStatus status = results[i].get_future().get();
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        build_status = status;
      }
    } catch (const std::exception& error) {
      LOG(ERROR) << "Build model failed with exception: " << error.what()
                 << " rank " << i;
      throw;
    }
  }

  for (auto& worker_thread : worker_threads) {
    worker_thread.join();
  }
  model_irs_[model_name] = std::move(model_ir);
  return build_status;
}

}  // namespace allspark
