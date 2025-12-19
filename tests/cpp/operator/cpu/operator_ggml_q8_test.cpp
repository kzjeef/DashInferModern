// Copyright (c) 2026 Segno System.

#ifdef ENABLE_GGML_GEMM

#include <ggml-cpu.h>
#include <ggml.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace allspark {
namespace {

TEST(GGMLQ8Test, QuantizedMatmulMatchesFloatReference) {
  constexpr int64_t k = 32;
  constexpr int64_t n = 4;
  constexpr int64_t m = 3;
  std::vector<float> weights(n * k);
  std::vector<float> input(m * k);
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = std::sin(i * 0.17f);
  }
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = std::cos(i * 0.11f);
  }

  const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, k);
  std::vector<uint8_t> packed(row_bytes * n);
  ASSERT_EQ(packed.size(),
            ggml_quantize_chunk(GGML_TYPE_Q8_0, weights.data(),
                                packed.data(), 0, n, k, nullptr));

  const size_t context_size =
      ggml_tensor_overhead() * 4 + ggml_graph_overhead();
  struct ggml_init_params params = {context_size, nullptr, true};
  struct ggml_context* context = ggml_init(params);
  ASSERT_NE(nullptr, context);

  struct ggml_tensor* weight =
      ggml_new_tensor_2d(context, GGML_TYPE_Q8_0, k, n);
  weight->data = packed.data();
  struct ggml_tensor* x =
      ggml_new_tensor_2d(context, GGML_TYPE_F32, k, m);
  x->data = input.data();
  std::vector<float> output(m * n);
  struct ggml_tensor* y = ggml_mul_mat(context, weight, x);
  y->data = output.data();

  struct ggml_cgraph* graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, y);
  struct ggml_cplan plan = ggml_graph_plan(graph, 4, nullptr);
  std::vector<uint8_t> work(plan.work_size);
  plan.work_data = work.empty() ? nullptr : work.data();
  ASSERT_EQ(GGML_STATUS_SUCCESS, ggml_graph_compute(graph, &plan));
  ggml_free(context);

  float max_error = 0.0f;
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t column = 0; column < n; ++column) {
      float expected = 0.0f;
      for (int64_t inner = 0; inner < k; ++inner) {
        expected += input[row * k + inner] *
                    weights[column * k + inner];
      }
      max_error = std::max(
          max_error,
          std::abs(output[row * n + column] - expected));
    }
  }
  EXPECT_LT(max_error, 0.08f);
}

}  // namespace
}  // namespace allspark

#endif  // ENABLE_GGML_GEMM
