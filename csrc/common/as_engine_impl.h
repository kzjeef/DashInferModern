/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_impl.h
 */

#pragma once

#include "engine_runtime.h"
#include "engine_worker.h"
#include "thread_pool_with_id.h"

#include <cache/prefix_cache_manager.h>
#include <core/model/model.h>
#include <interface/allspark.h>
#include <pd/pd_role.h>
#include <weight/weight_manager.h>

#include <atomic>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace allspark {

class AsEngineImpl final {
 public:
  AsEngineImpl();
  ~AsEngineImpl();

  AsStatus BuildModelFromConfigStruct(AsModelConfig& model_config);
  AsStatus UnloadModelFromDeviceMemory(const char* model_name);
  AsStatus ReloadModelFromDeviceMemory(const char* model_name);

  AsStatus LoadLoraByName(const char* model_name, const char* lora_name);
  AsStatus UnloadLoraByName(const char* model_name, const char* lora_name);
  std::vector<std::string> LoadFakeLoras(const char* model_name);
  void UnloadFakeLoras(const char* model_name,
                       const std::vector<std::string>& fake_lora_names);

  AsStatus GetModelInformation(const char* model_name, std::string* model_info);
  AsFileInfo GetFileInformation(const char* as_model_path,
                                const char* as_param_path);

  AsStatus StartModel(const char* model_name);
  AsStatus TunePrefixCache(const char* model_name);
  AsStatus StopModel(const char* model_name);
  AsStatus ReleaseModel(const char* model_name);

  AsStatus StartRequest(const char* model_name,
                        std::shared_ptr<AsEngine::RequestContent> request_info,
                        RequestHandle** request_handle,
                        AsEngine::ResultQueue** queue,
                        const std::string customized_uuid = "");
  AsStatus StopRequest(const char* model_name, RequestHandle* request_handle);
  AsStatus ReleaseRequest(const char* model_name,
                          RequestHandle* request_handle);
  AsStatus SyncRequest(const char* model_name, RequestHandle* request_handle);

#if ENABLE_SPAN_ATTENTION
  int64_t GetFreeFrame(const char* model_name);
#endif

  AsEngineStat GetAsEngineStat(const char* model_name);
  std::string GetVersionFull();
  std::string GetOpProfilingInfo(const char* model_name);
  int GetRankId();
  int GetRankNums();

 private:
  AsStatus BuildModel(const char* model_name, const std::string& model_proto,
                      std::shared_ptr<ModelWeightHandler> weight_handler,
                      const std::map<std::string, int>& model_limits = {});
  AsStatus WarmupModelInternal_(const char* model_name,
                                int64_t min_bytes_available,
                                std::vector<std::string>& fake_lora_names);
  AsStatus WarmupModel(const char* model_name);

  AsStatus SetNumThreads(int num_threads);
  AsStatus SetDeviceIds(const std::vector<int>& device_ids);
  AsStatus CreateDeviceContext(const std::string& compute_unit);
  void DestroyDeviceContext();
  AsStatus SetMatmulPrecision(const std::string& precision);
#if ENABLE_SPAN_ATTENTION
  AsStatus setSpanCacheConfig(AsCacheMode mode, int span_size,
                              int span_num_init, int span_num_grow);
#endif

  AsStatus RunTextGenerationContinue(const char* model_name);
  AsStatus RunTextGenerationContext(const char* model_name,
                                    bool is_new_context);
  AsStatus StartRequestImpl(const char* model_name,
                            std::shared_ptr<RequestHandle> request_handle,
                            DLTensorMap* outputs, GenerateConfig& gen_cfg);
  AsStatus StopRequestByRequestID(const char* model_name,
                                  std::string request_id);
  AsStatus ReleaseRequestByRequestID(const char* model_name,
                                     const std::string& request_id);

  template <typename T>
  int64_t GetInputBatch(const T& inputs);

  void UpdateAsEngineStat();
  void ModelRunningThread(std::string model_name,
                          std::shared_ptr<ModelControlState> model_state);
  void UpdateResult(std::string model_name,
                    std::shared_ptr<ModelControlState> model_state,
                    bool& synchronizing,
                    std::unordered_set<std::string>& sync_pending_set);
  AsStatus InputParamsVerify(
      const char* model_name,
      std::shared_ptr<AsEngine::RequestContent>& request_info);
  AsStatus RichInputVerify(
      TensorListMap& extra_embedding,
      std::shared_ptr<AsEngine::RequestContent>& request_info);
  void ExpandRankThreadPool();

  std::string ChooseVictimRequest(const std::vector<std::string>& request_ids,
                                  const std::vector<int>& request_lens, int n);
  AsStatus RunEngineContext(std::string model_name);

  pd::PdRuntimeConfig pd_config_;
  bool is_device_id_set_ = false;
  bool is_multi_nodes_;
  int nranks_ = 1;
  std::unique_ptr<AsEngineStat> as_stat_;
  std::vector<std::unique_ptr<Worker>> workers_;
  std::unique_ptr<DeviceContext> device_ctx_;
  std::unordered_map<std::string, std::unique_ptr<AsModel>> models_;
  std::unordered_map<std::string, std::unique_ptr<TransformerProto>> model_irs_;
  static std::unordered_map<std::string, int> precision_map_;

  std::unordered_map<std::string, std::shared_ptr<ModelControlState>>
      model_state_map_;

  std::mutex engine_lock_;
  std::mutex lora_lock_;
  std::mutex lora_usage_lock_;
  std::filesystem::path fake_lora_temp_dir_;
  int engine_max_length_ = 0;
  int engine_max_batch_ = 0;
  int engine_max_prefill_length_ = 0;
  const int engine_max_top_logprobs_ = 10;
  int engine_gpu_swap_threshold_ = -1;
  std::unique_ptr<ThreadPoolWithID> threadpool_;
  int threadpool_size_{1};
  bool use_adaptive_cache_{false};
  std::mt19937 random_engine;

  std::shared_ptr<WeightManager> weight_manager_;
  std::unordered_map<std::string, std::multiset<std::string>> loras_in_use_;
  std::atomic<int> lora_use_count_;

  PrefixCacheCoordinator::Ptr prefix_cache_coordinator_;
};

}  // namespace allspark
