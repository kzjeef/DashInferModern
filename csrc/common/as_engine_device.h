/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_device.h
 */

#pragma once

#include <interface/allspark.h>

#include <string>
#include <utility>
#include <vector>

namespace allspark::engine_internal {

std::pair<DeviceType, std::vector<int>> ParseDeviceType(
    const std::string& compute_unit);

}  // namespace allspark::engine_internal
