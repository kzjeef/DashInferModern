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

}  // extern "C"
