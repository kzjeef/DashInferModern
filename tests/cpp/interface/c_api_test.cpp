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
