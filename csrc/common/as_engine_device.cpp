/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_device.cpp
 */

#include "as_engine_device.h"
#include "as_engine_impl.h"

#include <common/allocator.h>
#include <common/device_context.h>
#include <cpu/cpu_context.h>

#include <future>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>

#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#include <nccl.h>
#endif

namespace allspark {
namespace engine_internal {

namespace {

DeviceType GetDeviceTypeFromString(const std::string& device_type) {
  const std::unordered_map<std::string, DeviceType> device_map(
      {{"CPU", DeviceType::CPU}, {"CUDA", DeviceType::CUDA}});
  auto device = device_map.find(device_type);
  if (device == device_map.end()) {
    return DeviceType::DEVICETYPE_UNDEFINED;
  }
  return device->second;
}

}  // namespace

std::pair<DeviceType, std::vector<int>> ParseDeviceType(
    const std::string& compute_unit) {
  size_t pos = compute_unit.find(':');
  if (pos == std::string::npos) {
    LOG(ERROR) << "Not Support ComputeUnit: " << compute_unit;
    throw std::invalid_argument("not support compute unit");
  }

  DeviceType device_type = GetDeviceTypeFromString(compute_unit.substr(0, pos));
  std::vector<int> device_ids;
  std::stringstream ss(compute_unit.substr(pos + 1));
  std::string item;
  while (std::getline(ss, item, ',')) {
    device_ids.push_back(std::stoi(item));
  }
  return std::make_pair(device_type, device_ids);
}

}  // namespace engine_internal

AsStatus AsEngineImpl::SetNumThreads(int num_threads) {
  DLOG(INFO) << "AsEngineImpl::SetNumThreads()" << std::endl;
  device_ctx_->SetNumThreads(num_threads);

  std::vector<std::future<AsStatus>> results(nranks_);
  for (int i = 0; i < workers_.size(); ++i) {
    results[i] = threadpool_->enqueue(i, [this, i, num_threads]() {
      return workers_[i]->SetNumThreads(num_threads);
    });
  }
  for (int i = 0; i < workers_.size(); ++i) {
    AS_CHECK_STATUS(results[i].get());
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

std::unordered_map<std::string, int> AsEngineImpl::precision_map_({
    {"highest", PrecisionLevel::HIGHEST},
    {"high", PrecisionLevel::HIGH},
    {"medium", PrecisionLevel::MEDIUM_BF16},
    {"medium_bf16", PrecisionLevel::MEDIUM_BF16},
    {"medium_fp16", PrecisionLevel::MEDIUM_FP16},
});

AsStatus AsEngineImpl::SetMatmulPrecision(const std::string& precision) {
  DLOG(INFO) << "AsEngineImpl::SetMatmulPrecision()" << std::endl;
  if (precision_map_.find(precision) == precision_map_.end()) {
    LOG(ERROR) << "Invalid precision_type:" << precision << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  device_ctx_->SetMatmulPrecision(precision_map_[precision]);
  for (int i = 0; i < nranks_; ++i) {
    workers_[i]->GetDeviceContext()->SetMatmulPrecision(
        precision_map_[precision]);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

#if ENABLE_SPAN_ATTENTION
AsStatus AsEngineImpl::setSpanCacheConfig(AsCacheMode mode, int span_size,
                                          int span_num_init,
                                          int span_num_grow) {
  SpanCacheConfig::Ptr cache_config =
      SpanCacheConfig::Create(mode, span_size, span_num_init, span_num_grow);
  if (!cache_config) {
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  device_ctx_->SetCacheConfig(cache_config);
  return AsStatus::ALLSPARK_SUCCESS;
}
#endif

AsStatus AsEngineImpl::SetDeviceIds(const std::vector<int>& device_ids) {
  DLOG(INFO) << "AsEngineImpl::SetDeviceIds()" << std::endl;
  if (is_device_id_set_) {
    LOG(WARNING) << "WARNING: device_ids already set, ignored!" << std::endl;
    return AsStatus::ALLSPARK_SUCCESS;
  }
  if (device_ctx_ == nullptr) {
    LOG(WARNING) << "device type should be set first" << std::endl;
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }

  DeviceType backend = device_ctx_->GetDeviceType();
  if (backend == DeviceType::CUDA) {
    InitBFCAllocator(backend, device_ids);
  }

  nranks_ = device_ids.size();
  LOG(INFO) << "SetDeviceIds: DeviceIDs.size() " << device_ids.size();
  for (int device_id : device_ids) {
    DLOG(INFO) << device_id;
  }
  workers_.resize(nranks_);
#ifdef ENABLE_CUDA
  ncclUniqueId id;
  if (backend == DeviceType::CUDA) {
    ncclGetUniqueId(&id);
  }
#endif
  std::vector<std::thread> worker_threads(nranks_);
  LOG(INFO) << "Start create " << nranks_ << " Device: " << backend
            << " workers.";
  for (int i = 0; i < nranks_; ++i) {
    worker_threads[i] = std::thread([&, i]() {
      switch (backend) {
#ifdef ENABLE_CUDA
        case DeviceType::CUDA:
          workers_[i] =
              std::make_unique<CudaWorker>(i, nranks_, id, device_ids[i]);
          break;
#endif
        case DeviceType::CPU:
          workers_[i] = std::make_unique<CpuWorker>(i, nranks_, device_ids[i]);
          break;
        default:
          LOG(ERROR) << "Unsupported device type: " << int(backend);
          break;
      }
      workers_[i]->Init();
      workers_[i]->InitCCL(i, nranks_);
      workers_[i]->SetWeightManager(weight_manager_);
    });
  }
  for (auto& worker_thread : worker_threads) {
    worker_thread.join();
  }

  is_device_id_set_ = true;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsEngineImpl::CreateDeviceContext(const std::string& compute_unit) {
  DLOG(INFO) << "AsEngineImpl::CreateDeviceContext()" << compute_unit
             << std::endl;
  DeviceType device_type = DeviceType::CUDA;
  std::vector<int> device_ids;

  try {
    std::tie(device_type, device_ids) =
        engine_internal::ParseDeviceType(compute_unit);
  } catch (const std::invalid_argument&) {
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  switch (device_type) {
    case DeviceType::CPU:
      device_ctx_ = std::make_unique<CPUContext>();
      AS_CHECK_STATUS(SetDeviceIds({0}));
      break;
#ifdef ENABLE_CUDA
    case DeviceType::CUDA:
      device_ctx_ = std::make_unique<CUDAContext>();
      AS_CHECK_STATUS(SetDeviceIds(device_ids));
      break;
#endif
    default:
      LOG(ERROR) << "Not Support ComputeUnit: " << compute_unit;
      return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

void AsEngineImpl::DestroyDeviceContext() {
  is_device_id_set_ = false;
  DestroyBFCAllocator();
}

}  // namespace allspark
