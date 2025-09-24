/*!
 * Copyright (c) 2026 Segno System.
 * @file    allspark_c.cpp
 */

#include "allspark_c.h"

#include "allspark.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <new>

struct as_engine {
  allspark::AsEngine engine;
};

struct as_request {
  as_engine_t* owner = nullptr;
  std::string model_name;
  allspark::RequestHandle_t handle = nullptr;
  allspark::AsEngine::ResultQueue_t queue = nullptr;
  std::shared_ptr<allspark::AsEngine::GeneratedElements> pending_result;
};

namespace {

as_status_t ToCStatus(allspark::AsStatus status) {
  return static_cast<as_status_t>(status);
}

template <typename Function>
as_status_t GuardCAbi(Function&& function) noexcept {
  try {
    return function();
  } catch (const std::bad_alloc&) {
    return AS_STATUS_MEMORY_ERROR;
  } catch (const std::exception&) {
    return AS_STATUS_RUNTIME_ERROR;
  } catch (...) {
    return AS_STATUS_UNKNOWN_ERROR;
  }
}

}  // namespace

extern "C" {

void as_model_config_init(as_model_config_t* config) {
  if (config == nullptr) {
    return;
  }
  std::memset(config, 0, sizeof(*config));
  config->struct_size = sizeof(*config);
  config->api_version = AS_C_API_VERSION;
  config->compute_unit = "CUDA:0";
  config->matmul_precision = "highest";
  config->engine_max_length = 2048;
  config->engine_max_batch = 32;
  config->engine_max_prefill_length = 0;
  config->swap_threshold = -1;
  config->cache_span_size = 128;
  config->prefix_cache_ttl = 300;
  config->enable_prefix_cache = 1;
}

void as_generate_config_init(as_generate_config_t* config) {
  if (config == nullptr) {
    return;
  }
  std::memset(config, 0, sizeof(*config));
  config->struct_size = sizeof(*config);
  config->api_version = AS_C_API_VERSION;
  config->max_length = 20;
  config->eos_token_id = 102;
  config->top_k = 50;
  config->top_p = 1.0f;
  config->temperature = 1.0f;
  config->repetition_penalty = 1.0f;
  config->do_sample = 1;
  config->early_stopping = 1;
}

as_status_t as_engine_create(as_engine_t** engine) {
  if (engine == nullptr) {
    return AS_STATUS_PARAM_ERROR;
  }
  *engine = nullptr;
  return GuardCAbi([engine]() {
    *engine = new as_engine();
    return AS_STATUS_SUCCESS;
  });
}

void as_engine_destroy(as_engine_t* engine) {
  try {
    delete engine;
  } catch (...) {
    // C callers cannot receive exceptions from a destructor.
  }
}

as_status_t as_engine_get_version(as_engine_t* engine, char* buffer,
                                  size_t* buffer_size) {
  if (engine == nullptr || buffer_size == nullptr) {
    return AS_STATUS_PARAM_ERROR;
  }
  return GuardCAbi([engine, buffer, buffer_size]() {
    const std::string version = engine->engine.GetVersionFull();
    const size_t required = version.size() + 1;
    const size_t available = *buffer_size;
    *buffer_size = required;
    if (buffer == nullptr) {
      return AS_STATUS_SUCCESS;
    }
    if (available < required) {
      return AS_STATUS_EXCEED_LIMIT;
    }
    std::memcpy(buffer, version.c_str(), required);
    return AS_STATUS_SUCCESS;
  });
}

const char* as_status_string(as_status_t status) {
  switch (status) {
    case AS_STATUS_SUCCESS:
      return "AS_STATUS_SUCCESS";
    case AS_STATUS_UNKNOWN_ERROR:
      return "AS_STATUS_UNKNOWN_ERROR";
    case AS_STATUS_PARAM_ERROR:
      return "AS_STATUS_PARAM_ERROR";
    case AS_STATUS_IO_ERROR:
      return "AS_STATUS_IO_ERROR";
    case AS_STATUS_MEMORY_ERROR:
      return "AS_STATUS_MEMORY_ERROR";
    case AS_STATUS_RUNTIME_ERROR:
      return "AS_STATUS_RUNTIME_ERROR";
    case AS_STATUS_EXCEED_LIMIT:
      return "AS_STATUS_EXCEED_LIMIT";
    case AS_STATUS_INVALID_CALL:
      return "AS_STATUS_INVALID_CALL";
    case AS_STATUS_EMPTY_REQUEST:
      return "AS_STATUS_EMPTY_REQUEST";
    case AS_STATUS_ILLEGAL_REQUEST_ID:
      return "AS_STATUS_ILLEGAL_REQUEST_ID";
    case AS_STATUS_CACHE_MEMORY_OUT:
      return "AS_STATUS_CACHE_MEMORY_OUT";
    case AS_STATUS_REQUEST_DENIED:
      return "AS_STATUS_REQUEST_DENIED";
    case AS_STATUS_CHUNK_PREFILL:
      return "AS_STATUS_CHUNK_PREFILL";
    case AS_STATUS_DEPRECATED:
      return "AS_STATUS_DEPRECATED";
    case AS_STATUS_LORA_NUM_EXCEED_LIMIT:
      return "AS_STATUS_LORA_NUM_EXCEED_LIMIT";
    case AS_STATUS_LORA_RANK_EXCEED_LIMIT:
      return "AS_STATUS_LORA_RANK_EXCEED_LIMIT";
    case AS_STATUS_LORA_NOT_FOUND:
      return "AS_STATUS_LORA_NOT_FOUND";
    case AS_STATUS_LORA_ALREADY_LOADED:
      return "AS_STATUS_LORA_ALREADY_LOADED";
    case AS_STATUS_LORA_IN_USE:
      return "AS_STATUS_LORA_IN_USE";
    case AS_STATUS_STREAMING:
      return "AS_STATUS_STREAMING";
    default:
      return "AS_STATUS_UNDEFINED";
  }
}

as_status_t as_engine_build_model(as_engine_t* engine,
                                  const as_model_config_t* config) {
  if (engine == nullptr || config == nullptr ||
      config->struct_size < sizeof(as_model_config_t) ||
      config->api_version != AS_C_API_VERSION ||
      config->model_name == nullptr || config->model_name[0] == '\0' ||
      config->model_path == nullptr || config->model_path[0] == '\0' ||
      config->weights_path == nullptr || config->weights_path[0] == '\0') {
    return AS_STATUS_PARAM_ERROR;
  }
  if (config->cache_mode < 0 || config->cache_mode > 2 ||
      (config->prefill_mode != 0 && config->prefill_mode != 10 &&
       config->prefill_mode != 11) ||
      config->eviction_strategy < 0 || config->eviction_strategy > 1 ||
      config->scheduling_strategy < 0 ||
      config->scheduling_strategy > 1) {
    return AS_STATUS_PARAM_ERROR;
  }

  return GuardCAbi([engine, config]() {
    allspark::AsModelConfig cpp_config;
    cpp_config.model_name = config->model_name;
    cpp_config.model_path = config->model_path;
    cpp_config.weights_path = config->weights_path;
    cpp_config.compute_unit =
        config->compute_unit == nullptr ? "CUDA:0" : config->compute_unit;
    cpp_config.matmul_precision = config->matmul_precision == nullptr
                                      ? "highest"
                                      : config->matmul_precision;
    cpp_config.engine_max_length = config->engine_max_length;
    cpp_config.engine_max_batch = config->engine_max_batch;
    cpp_config.engine_max_prefill_length =
        config->engine_max_prefill_length;
    cpp_config.swap_threshold = config->swap_threshold;
    cpp_config.num_threads = config->num_threads;
    cpp_config.cache_span_size = config->cache_span_size;
    cpp_config.cache_span_num_init = config->cache_span_num_init;
    cpp_config.cache_span_num_grow = config->cache_span_num_grow;
    cpp_config.prefix_cache_ttl = config->prefix_cache_ttl;
    cpp_config.cache_mode =
        static_cast<allspark::AsCacheMode>(config->cache_mode);
    cpp_config.prefill_mode =
        static_cast<allspark::AsMHAPrefill>(config->prefill_mode);
    cpp_config.eviction_strategy =
        static_cast<allspark::AsEvictionStrategy>(config->eviction_strategy);
    cpp_config.scheduling_strategy = static_cast<allspark::AsSchedulingStrategy>(
        config->scheduling_strategy);
    cpp_config.text_graph = config->text_graph != 0;
    cpp_config.enable_prefix_cache = config->enable_prefix_cache != 0;
    cpp_config.enable_sparsity_matmul =
        config->enable_sparsity_matmul != 0;
    return ToCStatus(engine->engine.BuildModelFromConfigStruct(cpp_config));
  });
}

as_status_t as_engine_start_model(as_engine_t* engine,
                                  const char* model_name) {
  if (engine == nullptr || model_name == nullptr || model_name[0] == '\0') {
    return AS_STATUS_PARAM_ERROR;
  }
  return GuardCAbi([engine, model_name]() {
    return ToCStatus(engine->engine.StartModel(model_name));
  });
}

as_status_t as_engine_stop_model(as_engine_t* engine,
                                 const char* model_name) {
  if (engine == nullptr || model_name == nullptr || model_name[0] == '\0') {
    return AS_STATUS_PARAM_ERROR;
  }
  return GuardCAbi([engine, model_name]() {
    return ToCStatus(engine->engine.StopModel(model_name));
  });
}

as_status_t as_engine_release_model(as_engine_t* engine,
                                    const char* model_name) {
  if (engine == nullptr || model_name == nullptr || model_name[0] == '\0') {
    return AS_STATUS_PARAM_ERROR;
  }
  return GuardCAbi([engine, model_name]() {
    return ToCStatus(engine->engine.ReleaseModel(model_name));
  });
}

as_status_t as_engine_load_lora(as_engine_t* engine, const char* model_name,
                                const char* lora_name_or_path) {
  if (engine == nullptr || model_name == nullptr || model_name[0] == '\0' ||
      lora_name_or_path == nullptr || lora_name_or_path[0] == '\0') {
    return AS_STATUS_PARAM_ERROR;
  }
  return GuardCAbi([engine, model_name, lora_name_or_path]() {
    return ToCStatus(
        engine->engine.LoadLoraByName(model_name, lora_name_or_path));
  });
}

as_status_t as_engine_unload_lora(as_engine_t* engine, const char* model_name,
                                  const char* lora_name) {
  if (engine == nullptr || model_name == nullptr || model_name[0] == '\0' ||
      lora_name == nullptr || lora_name[0] == '\0') {
    return AS_STATUS_PARAM_ERROR;
  }
  return GuardCAbi([engine, model_name, lora_name]() {
    return ToCStatus(engine->engine.UnloadLoraByName(model_name, lora_name));
  });
}

as_status_t as_engine_start_request(as_engine_t* engine,
                                    const char* model_name,
                                    const as_named_tensor_t* inputs,
                                    size_t input_count,
                                    const as_generate_config_t* config,
                                    as_request_t** request) {
  if (request != nullptr) {
    *request = nullptr;
  }
  if (engine == nullptr || model_name == nullptr || model_name[0] == '\0' ||
      inputs == nullptr || input_count == 0 || config == nullptr ||
      request == nullptr ||
      config->struct_size < sizeof(as_generate_config_t) ||
      config->api_version != AS_C_API_VERSION) {
    return AS_STATUS_PARAM_ERROR;
  }

  return GuardCAbi([=]() -> as_status_t {
    auto input_map = std::make_shared<allspark::DLTensorMap>();
    for (size_t i = 0; i < input_count; ++i) {
      if (inputs[i].name == nullptr || inputs[i].name[0] == '\0' ||
          inputs[i].tensor == nullptr) {
        return AS_STATUS_PARAM_ERROR;
      }
      if (!input_map->emplace(inputs[i].name, inputs[i].tensor).second) {
        return AS_STATUS_PARAM_ERROR;
      }
    }
    if (input_map->count("input_ids") == 0) {
      return AS_STATUS_PARAM_ERROR;
    }

    auto request_content =
        std::make_shared<allspark::AsEngine::RequestContent>();
    request_content->infer_type =
        allspark::AsEngine::RequestInferType::Generate;
    request_content->mm_type = allspark::AsEngine::RequestMMType::TextInput;
    request_content->inputs = std::move(input_map);
    request_content->config.max_length = config->max_length;
    request_content->config.min_length = config->min_length;
    request_content->config.eos_token_id = config->eos_token_id;
    request_content->config.top_k = config->top_k;
    request_content->config.top_p = config->top_p;
    request_content->config.temperature = config->temperature;
    request_content->config.repetition_penalty = config->repetition_penalty;
    request_content->config.presence_penalty = config->presence_penalty;
    request_content->config.frequency_penalty = config->frequency_penalty;
    request_content->config.seed = config->seed;
    request_content->config.do_sample = config->do_sample != 0;
    request_content->config.early_stopping = config->early_stopping != 0;
    request_content->config.lora_name =
        config->lora_name == nullptr ? "" : config->lora_name;

    auto wrapper = std::make_unique<as_request>();
    wrapper->owner = engine;
    wrapper->model_name = model_name;
    const as_status_t status = ToCStatus(engine->engine.StartRequest(
        model_name, std::move(request_content), &wrapper->handle,
        &wrapper->queue));
    if (status != AS_STATUS_SUCCESS) {
      return status;
    }
    *request = wrapper.release();
    return AS_STATUS_SUCCESS;
  });
}

as_status_t as_engine_stop_request(as_engine_t* engine,
                                   as_request_t* request) {
  if (engine == nullptr || request == nullptr || request->owner != engine ||
      request->handle == nullptr) {
    return AS_STATUS_PARAM_ERROR;
  }
  return GuardCAbi([engine, request]() {
    return ToCStatus(engine->engine.StopRequest(request->model_name.c_str(),
                                                request->handle));
  });
}

as_status_t as_engine_sync_request(as_engine_t* engine,
                                   as_request_t* request) {
  if (engine == nullptr || request == nullptr || request->owner != engine ||
      request->handle == nullptr) {
    return AS_STATUS_PARAM_ERROR;
  }
  return GuardCAbi([engine, request]() {
    return ToCStatus(engine->engine.SyncRequest(request->model_name.c_str(),
                                                request->handle));
  });
}

as_status_t as_engine_release_request(as_engine_t* engine,
                                      as_request_t* request) {
  if (engine == nullptr || request == nullptr || request->owner != engine ||
      request->handle == nullptr) {
    return AS_STATUS_PARAM_ERROR;
  }
  return GuardCAbi([engine, request]() {
    const as_status_t status = ToCStatus(engine->engine.ReleaseRequest(
        request->model_name.c_str(), request->handle));
    if (status == AS_STATUS_SUCCESS) {
      delete request;
    }
    return status;
  });
}

as_status_t as_request_get_status(const as_request_t* request,
                                  int32_t* status) {
  if (request == nullptr || request->queue == nullptr || status == nullptr) {
    return AS_STATUS_PARAM_ERROR;
  }
  return GuardCAbi([request, status]() {
    *status = static_cast<int32_t>(request->queue->GenerateStatus());
    return AS_STATUS_SUCCESS;
  });
}

as_status_t as_request_generated_length(const as_request_t* request,
                                        size_t* length) {
  if (request == nullptr || request->queue == nullptr || length == nullptr) {
    return AS_STATUS_PARAM_ERROR;
  }
  return GuardCAbi([request, length]() {
    *length = request->queue->GeneratedLength();
    return AS_STATUS_SUCCESS;
  });
}

as_status_t as_request_fetch_tokens(as_request_t* request, int32_t timeout_ms,
                                    int64_t* tokens, size_t* token_count) {
  if (request == nullptr || request->queue == nullptr ||
      token_count == nullptr) {
    return AS_STATUS_PARAM_ERROR;
  }
  return GuardCAbi([=]() -> as_status_t {
    if (request->pending_result == nullptr) {
      if (timeout_ms < 0) {
        request->pending_result = request->queue->Get();
      } else if (timeout_ms == 0) {
        request->pending_result = request->queue->GetNoWait();
      } else {
        request->pending_result = request->queue->Get(timeout_ms);
      }
    }
    if (request->pending_result == nullptr) {
      *token_count = 0;
      return AS_STATUS_EMPTY_REQUEST;
    }

    const auto& ids = request->pending_result->ids_from_generate;
    const size_t required = ids.size();
    const size_t available = *token_count;
    *token_count = required;
    if (required == 0) {
      request->pending_result.reset();
      return AS_STATUS_SUCCESS;
    }
    if (tokens == nullptr) {
      return AS_STATUS_SUCCESS;
    }
    if (available < required) {
      return AS_STATUS_EXCEED_LIMIT;
    }
    std::copy(ids.begin(), ids.end(), tokens);
    request->pending_result.reset();
    return AS_STATUS_SUCCESS;
  });
}

}  // extern "C"
