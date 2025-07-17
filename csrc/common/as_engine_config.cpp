/*!
 * Copyright (c) 2026 Segno System.
 * @file    as_engine_config.cpp
 */

#include "engine_runtime.h"

#include <interface/allspark.h>

#include <string>
#include <utility>

namespace allspark {

AsModelConfig::AsModelConfig() {
#ifdef ENABLE_CUDA
  prefill_mode = AsMHAPrefill::AsPrefillXformer;
#else
  prefill_mode = AsMHAPrefill::AsPrefillDefault;
#endif
  cache_span_size = default_span_size;
}

AsModelConfig::AsModelConfig(
    std::string in_model_name, std::string in_model_path,
    std::string in_weights_path, std::string in_compute_unit,
    int in_engine_max_length, int in_engine_max_batch,
    int in_engine_max_prefill_length, int64_t in_swap_threshold,
    bool in_text_graph, int in_num_threads, std::string in_matmul_precision,
    std::vector<std::string> lora_names, int in_cache_span_size,
    int in_cache_span_num_init, int in_cache_span_num_grow,
    bool enable_prefix_cache, int prefix_cache_ttl,
    AsMHAPrefill in_prefill_mode, AsCacheMode in_cache_mode,
    AsEvictionStrategy in_eviction_strategy,
    AsSchedulingStrategy in_scheduling_strategy, bool enable_sparsity_matmul,
    int lora_max_rank, int lora_max_num)
    : model_name(std::move(in_model_name)),
      model_path(std::move(in_model_path)),
      weights_path(std::move(in_weights_path)),
      compute_unit(std::move(in_compute_unit)),
      num_threads(in_num_threads),
      matmul_precision(in_matmul_precision),
      swap_threshold(in_swap_threshold),
      engine_max_length(in_engine_max_length),
      engine_max_batch(in_engine_max_batch),
      engine_max_prefill_length(in_engine_max_prefill_length),
      lora_names(lora_names),
      cache_span_size(in_cache_span_size),
      cache_span_num_init(in_cache_span_num_init),
      cache_span_num_grow(in_cache_span_num_grow),
      cache_mode(in_cache_mode),
      enable_prefix_cache(enable_prefix_cache),
      prefix_cache_ttl(prefix_cache_ttl),
      prefill_mode(in_prefill_mode),
      eviction_strategy(in_eviction_strategy),
      text_graph(in_text_graph),
      scheduling_strategy(in_scheduling_strategy),
      enable_sparsity_matmul(enable_sparsity_matmul),
      lora_max_rank(lora_max_rank),
      lora_max_num(lora_max_num) {
  if (in_prefill_mode == AsMHAPrefill::AsPrefillDefault) {
#ifdef ENABLE_CUDA
    prefill_mode = AsMHAPrefill::AsPrefillXformer;
#else
    prefill_mode = AsMHAPrefill::AsPrefillDefault;
#endif
  }
}

std::string AsModelConfig::ToString() const {
  std::string prefill_string;
  switch (prefill_mode) {
    case AsMHAPrefill::AsPrefillDefault:
      prefill_string = "AsPrefillDefault";
      break;
    case AsMHAPrefill::AsPrefillFlashV2:
      prefill_string = "AsPrefillFlashV2";
      break;
    case AsMHAPrefill::AsPrefillXformer:
      prefill_string = "AsPrefillXformer";
      break;
    default:
      prefill_string = "AsPrefillUnknown";
  }
  std::string cache_mode_string = SpanCacheConfig::CacheMode2String(cache_mode);
  std::string eviction_strategy_string;
  switch (eviction_strategy) {
    case AsEvictionStrategy::MaxLength:
      eviction_strategy_string = "MaxLength";
      break;
    case AsEvictionStrategy::Random:
      eviction_strategy_string = "Random";
      break;
    default:
      eviction_strategy_string = "Unknown";
  }

  std::string lora_name_string = "init_loaded_loras: [";
  for (const auto& name : lora_names) {
    lora_name_string += name + ",\t";
  }
  lora_name_string += "]";

  std::string result = "AsModelConfig :\n";
  result += "\tmodel_name: " + model_name + "\n";
  result += "\tmodel_path: " + model_path + "\n";
  result += "\tweights_path: " + weights_path + "\n";
  result += "\tcompute_unit: " + compute_unit + "\n";
  result += "\tnum_threads: " + std::to_string(num_threads) + "\n";
  result += "\tmatmul_precision: " + matmul_precision + "\n";
  result += "\tprefill_mode: " + prefill_string + "\n";
  result += "\tcache_mode: " + cache_mode_string + "\n";
  result += "\teviction_strategy: " + eviction_strategy_string + "\n";
  result += "\tengine_max_length = " + std::to_string(engine_max_length) +
            "\n";
  result += "\tengine_max_batch = " + std::to_string(engine_max_batch) + "\n";
  result += "\tengine_max_prefill_length = " +
            std::to_string(engine_max_prefill_length) + "\n";
#ifdef FIXED_SPAN_SIZE
  result += "\tcache_span_size (fixed) = " +
            std::to_string(FIXED_SPAN_SIZE) + "\n";
#else
  result += "\tcache_span_size = " + std::to_string(cache_span_size) + "\n";
#endif
  result += "\tcache_span_num_init = " +
            std::to_string(cache_span_num_init) + "\n";
  result += "\tcache_span_num_grow = " +
            std::to_string(cache_span_num_grow) + "\n";
  result += "\tenable_prefix_cache = " +
            std::to_string(enable_prefix_cache) + "\n";
  result += "\tprefix_cache_ttl = " + std::to_string(prefix_cache_ttl) + "\n";
  result += "\t" + lora_name_string + "\n";
  result += "\tswap_threshold = " + std::to_string(swap_threshold) + "\n";
  result += "\tenable_sparsity_matmul = " +
            std::to_string(enable_sparsity_matmul) + "\n";
  result += "\tlora_max_rank= " + std::to_string(lora_max_rank) + "\n";
  result += "\tlora_max_num= " + std::to_string(lora_max_num) + "\n";
  return result;
}

}  // namespace allspark
