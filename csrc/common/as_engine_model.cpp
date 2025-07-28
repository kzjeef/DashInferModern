/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_model.cpp
 */

#include "as_engine_device.h"
#include "as_engine_impl.h"

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
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <tuple>

#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif

namespace allspark {
namespace {

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

}  // namespace allspark
