/*!
 * Copyright (c) 2026 Segno System.
 * @file    engine_runtime.cpp
 */

#include "engine_runtime.h"

namespace allspark {

ModelControlState::ModelControlState(const std::string& name)
    : model_name(name), msg_queue(1000) {
  request_handle_map.reserve(1000);
  result_queue_map.reserve(1000);
  msg_queue_size.store(0);
}

}  // namespace allspark
