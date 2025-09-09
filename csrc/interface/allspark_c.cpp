/*!
 * Copyright (c) 2026 Segno System.
 * @file    allspark_c.cpp
 */

#include "allspark_c.h"

#include "allspark.h"

#include <cstring>
#include <exception>
#include <new>

struct as_engine {
  allspark::AsEngine engine;
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

}  // extern "C"
