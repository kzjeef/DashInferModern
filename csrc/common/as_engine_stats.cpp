/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_stats.cpp
 */

#include <interface/allspark.h>

#include <map>
#include <string>

namespace allspark {

std::string AsEngineStat::ToString() const {
  std::string result = "Members of AsEngineStat\n";
  result += "total_span = " + std::to_string(total_span) + "\n";
  result += "used_span = " + std::to_string(used_span) + "\n";
  result += "free_span = " + std::to_string(free_span) + "\n";
  result += "span_size = " + std::to_string(span_size) + "\n";
  result += "free_token = " + std::to_string(free_token) + "\n";
  result += "pendding_request = " + std::to_string(pendding_request) + "\n";
  result += "running_request = " + std::to_string(running_request) + "\n";
  result += "total_device_memory_pool_size = " +
            std::to_string(total_device_memory_pool_size) + "\n";
  result += "used_device_memory_pool_size = " +
            std::to_string(used_device_memory_pool_size) + "\n";
  result +=
      "total_generated_token = " + std::to_string(total_generated_token) + "\n";
  result +=
      "total_prefill_token = " + std::to_string(total_prefill_token) + "\n";
  result +=
      "prefix_cache_hit_rate = " + std::to_string(prefix_cache_hit_rate) + "\n";
  result += "prefix_cache_miss_rate = " +
            std::to_string(prefix_cache_miss_rate) + "\n";
  return result;
}

std::map<std::string, std::string> AsEngineStat::ToMap() const {
  std::map<std::string, std::string> engine_stat_map;
  engine_stat_map["total_span"] = std::to_string(total_span);
  engine_stat_map["used_span"] = std::to_string(used_span);
  engine_stat_map["free_span"] = std::to_string(free_span);
  engine_stat_map["free_token"] = std::to_string(free_token);
  engine_stat_map["total_token"] = std::to_string(total_token);
  engine_stat_map["span_size"] = std::to_string(span_size);
  engine_stat_map["pendding_request"] = std::to_string(pendding_request);
  engine_stat_map["running_request"] = std::to_string(running_request);
  engine_stat_map["total_device_memory_pool_size"] =
      std::to_string(total_device_memory_pool_size);
  engine_stat_map["used_device_memory_pool_size"] =
      std::to_string(used_device_memory_pool_size);
  engine_stat_map["total_generated_token"] =
      std::to_string(total_generated_token);
  engine_stat_map["total_prefill_token"] = std::to_string(total_prefill_token);
  engine_stat_map["generate_token_persec"] =
      std::to_string(generate_token_persec);
  engine_stat_map["process_token_persec"] =
      std::to_string(process_token_persec);
  engine_stat_map["token_usage_percentage"] =
      std::to_string(token_usage_percentage);
  engine_stat_map["prefix_cache_hit_token"] =
      std::to_string(prefix_cache_hit_token);
  engine_stat_map["prefix_cache_miss_token"] =
      std::to_string(prefix_cache_miss_token);
  engine_stat_map["prefix_cache_hit_rate"] =
      std::to_string(prefix_cache_hit_rate);
  engine_stat_map["prefix_cache_miss_rate"] =
      std::to_string(prefix_cache_miss_rate);
  return engine_stat_map;
}

}  // namespace allspark
