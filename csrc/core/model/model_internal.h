/*!
 * Copyright (c) 2026 Segno System.
 * @file    model_internal.h
 */

#pragma once

#include <string>

#define DEBUG_GEN_LAYER 0
#define DEBUG_GEN_LAYER_SAVE_NPY 0
#define DEBUG_GEN_LAYER_SYNC 0

#ifdef ENABLE_CUDA
#define PROFILE_CONTEXT_TIME_GPU 0
#define PROFILE_GENERATION_TIME_GPU 0
#define PROFILE_GENERATION_TIME_BS 100
#endif

inline bool isWarmupRequest(const std::string& request_id) {
  const std::string prefix = "warmup_request_";
  return request_id.find(prefix) == 0;
}

#ifdef DEBUG_GEN_LAYER
inline bool debugCurrentRequest(const std::string& request_id) {
  if (isWarmupRequest(request_id)) {
    return false;
  }

  const std::string target_request;
  return target_request.empty() || target_request == request_id;
}
#endif

#define CHECK_CUDA_ERROR(op)
