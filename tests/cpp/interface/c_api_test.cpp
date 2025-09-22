/*!
 * Copyright (c) 2026 Segno System.
 * @file    c_api_test.cpp
 */

#include <gtest/gtest.h>

#include <allspark_c.h>

#include <string>
#include <vector>

TEST(CApiTest, ValidatesEngineOutputPointer) {
  EXPECT_EQ(as_engine_create(nullptr), AS_STATUS_PARAM_ERROR);
  as_engine_destroy(nullptr);
}

TEST(CApiTest, CreatesEngineAndReadsVersion) {
  as_engine_t* engine = nullptr;
  ASSERT_EQ(as_engine_create(&engine), AS_STATUS_SUCCESS);
  ASSERT_NE(engine, nullptr);

  size_t required = 0;
  EXPECT_EQ(as_engine_get_version(engine, nullptr, &required),
            AS_STATUS_SUCCESS);
  ASSERT_GT(required, 1u);

  std::vector<char> short_buffer(required - 1);
  size_t short_size = short_buffer.size();
  EXPECT_EQ(as_engine_get_version(engine, short_buffer.data(), &short_size),
            AS_STATUS_EXCEED_LIMIT);
  EXPECT_EQ(short_size, required);

  std::vector<char> buffer(required);
  size_t buffer_size = buffer.size();
  EXPECT_EQ(as_engine_get_version(engine, buffer.data(), &buffer_size),
            AS_STATUS_SUCCESS);
  EXPECT_EQ(buffer_size, required);
  EXPECT_FALSE(std::string(buffer.data()).empty());

  as_engine_destroy(engine);
}

TEST(CApiTest, ReturnsStableStatusNames) {
  EXPECT_STREQ(as_status_string(AS_STATUS_SUCCESS), "AS_STATUS_SUCCESS");
  EXPECT_STREQ(as_status_string(AS_STATUS_RUNTIME_ERROR),
               "AS_STATUS_RUNTIME_ERROR");
  EXPECT_STREQ(as_status_string(-1), "AS_STATUS_UNDEFINED");
}

TEST(CApiTest, InitializesVersionedModelConfig) {
  as_model_config_t config;
  as_model_config_init(&config);

  EXPECT_EQ(config.struct_size, sizeof(config));
  EXPECT_EQ(config.api_version, AS_C_API_VERSION);
  EXPECT_EQ(config.engine_max_length, 2048);
  EXPECT_EQ(config.engine_max_batch, 32);
  EXPECT_EQ(config.swap_threshold, -1);
  EXPECT_EQ(config.cache_span_size, 128);
  EXPECT_EQ(config.enable_prefix_cache, 1);
  EXPECT_STREQ(config.compute_unit, "CUDA:0");
  EXPECT_STREQ(config.matmul_precision, "highest");
}

TEST(CApiTest, RejectsIncompleteModelConfig) {
  as_engine_t* engine = nullptr;
  ASSERT_EQ(as_engine_create(&engine), AS_STATUS_SUCCESS);

  as_model_config_t config;
  as_model_config_init(&config);
  EXPECT_EQ(as_engine_build_model(engine, &config), AS_STATUS_PARAM_ERROR);

  config.model_name = "mini";
  config.model_path = "mini.asgraph";
  config.weights_path = "mini.asparam";
  config.cache_mode = 99;
  EXPECT_EQ(as_engine_build_model(engine, &config), AS_STATUS_PARAM_ERROR);

  EXPECT_EQ(as_engine_start_model(nullptr, "mini"), AS_STATUS_PARAM_ERROR);
  EXPECT_EQ(as_engine_stop_model(engine, nullptr), AS_STATUS_PARAM_ERROR);
  EXPECT_EQ(as_engine_release_model(engine, ""), AS_STATUS_PARAM_ERROR);

  as_engine_destroy(engine);
}

TEST(CApiTest, InitializesGenerationConfig) {
  as_generate_config_t config;
  as_generate_config_init(&config);

  EXPECT_EQ(config.struct_size, sizeof(config));
  EXPECT_EQ(config.api_version, AS_C_API_VERSION);
  EXPECT_EQ(config.max_length, 20);
  EXPECT_EQ(config.eos_token_id, 102);
  EXPECT_EQ(config.top_k, 50);
  EXPECT_FLOAT_EQ(config.top_p, 1.0f);
  EXPECT_FLOAT_EQ(config.temperature, 1.0f);
  EXPECT_FLOAT_EQ(config.repetition_penalty, 1.0f);
  EXPECT_EQ(config.do_sample, 1);
  EXPECT_EQ(config.early_stopping, 1);
}

TEST(CApiTest, RejectsInvalidRequestArguments) {
  as_engine_t* engine = nullptr;
  ASSERT_EQ(as_engine_create(&engine), AS_STATUS_SUCCESS);

  as_generate_config_t config;
  as_generate_config_init(&config);
  as_request_t* request = reinterpret_cast<as_request_t*>(1);
  EXPECT_EQ(as_engine_start_request(engine, "mini", nullptr, 0, &config,
                                    &request),
            AS_STATUS_PARAM_ERROR);
  EXPECT_EQ(request, nullptr);

  as_named_tensor_t invalid_input{"input_ids", nullptr};
  EXPECT_EQ(as_engine_start_request(engine, "mini", &invalid_input, 1,
                                    &config, &request),
            AS_STATUS_PARAM_ERROR);
  EXPECT_EQ(request, nullptr);

  int32_t status = -1;
  size_t length = 0;
  EXPECT_EQ(as_request_get_status(nullptr, &status), AS_STATUS_PARAM_ERROR);
  EXPECT_EQ(as_request_generated_length(nullptr, &length),
            AS_STATUS_PARAM_ERROR);
  EXPECT_EQ(as_request_fetch_tokens(nullptr, 0, nullptr, &length),
            AS_STATUS_PARAM_ERROR);

  as_engine_destroy(engine);
}
