/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_op_cpu.cpp
 */

#include "gemm_op_cpu.h"

#include <core/kernel/kernel.h>
#include <cpu/cpu_common.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <unordered_map>

#ifdef ENABLE_GGML_GEMM
#include <ggml-cpu.h>
#include <ggml.h>
#endif

using dnnl::memory;
using tag = memory::format_tag;
namespace allspark {

#ifdef ENABLE_GGML_GEMM
namespace {
std::mutex ggml_q8_cache_mutex;
std::unordered_map<const AsTensor*, std::shared_ptr<std::vector<uint8_t>>>
    ggml_q8_cache;
}  // namespace
#endif

inline bool UseOneDnn(int batch, float alpha) {
  return batch == 1 && alpha == 1.0f;
}

AsStatus GemmOpCPU::Init(const OperatorProto& op_proto,
                         const DeviceContext& ctx, const TensorMap& weights_map,
                         TensorMap* tensor_map) {
  LOG(ERROR) << "GemmOpCPU only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmOpCPU::InitV2(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap& weights_buffer, TensorMap* tensor_map) {
  AS_CHECK_STATUS(GemmOpBase::InitV2(op_proto, ctx, weights_map, weights_buffer,
                                     tensor_map));

#ifdef ENABLE_GGML_GEMM
  const auto& attr_map = op_proto.attr();
  const auto q8_attr = attr_map.find("use_ggml_q8_0");
  use_ggml_q8_0_ =
      q8_attr != attr_map.end() && *reinterpret_cast<const bool*>(
                                       q8_attr->second.data());
  if (use_ggml_q8_0_) {
    if (weights_[0]->GetDataType() != DataType::FLOAT32 || batch_ != 1 ||
        k_ % ggml_blck_size(GGML_TYPE_Q8_0) != 0) {
      LOG(ERROR) << "GGML Q8_0 requires a dense FP32 weight with K divisible "
                    "by "
                 << ggml_blck_size(GGML_TYPE_Q8_0) << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    if (activation_ != UNARYTYPE_UNDEFINED && activation_ != UnaryType::SILU) {
      LOG(ERROR) << "GGML Q8_0 only supports the Qwen SiLU GEMM epilogue"
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    if (binary_type_ != BINARYTYPE_UNDEFINED && binary_type_ != BinaryType::ADD) {
      LOG(ERROR) << "GGML Q8_0 only supports the Qwen residual-add epilogue"
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }

    {
      std::lock_guard<std::mutex> lock(ggml_q8_cache_mutex);
      const auto cached = ggml_q8_cache.find(weights_[0]);
      if (cached != ggml_q8_cache.end()) {
        ggml_weight_ = cached->second;
      }
    }
    if (!ggml_weight_) {
      const float* source =
          static_cast<const float*>(weights_[0]->GetDataPtr());
      std::vector<float> transposed;
      if (!transB_) {
        transposed.resize(static_cast<size_t>(n_) * k_);
        cpu::parallel_for(n_, [&](int64_t column) {
          for (int64_t row = 0; row < k_; ++row) {
            transposed[static_cast<size_t>(column) * k_ + row] =
                source[static_cast<size_t>(row) * ldb_ + column];
          }
        });
        source = transposed.data();
      }

      const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, k_);
      auto quantized = std::make_shared<std::vector<uint8_t>>(row_bytes * n_);
      const size_t written = ggml_quantize_chunk(
          GGML_TYPE_Q8_0, source, quantized->data(), 0, n_, k_, nullptr);
      if (written != quantized->size()) {
        LOG(ERROR) << "GGML Q8_0 weight packing produced " << written
                   << " bytes, expected " << quantized->size() << std::endl;
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }
      {
        std::lock_guard<std::mutex> lock(ggml_q8_cache_mutex);
        auto inserted = ggml_q8_cache.emplace(weights_[0], quantized);
        ggml_weight_ = inserted.first->second;
      }
    }

    const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
    ggml_n_threads_ = std::max(1, std::min(12, cpu_ctx->GetNumThread()));
    return AsStatus::ALLSPARK_SUCCESS;
  }
#endif

  auto eng = DNNLEngine::GetInstance().GetEngine();
  dnnl_op_ctx_ = std::make_unique<DNNLOpContext>();
  dnnl_op_ctx_->pr_fwd_.resize(1);
  // onednn gemm may have extra inputs since introduced binary_op
  dnnl_op_ctx_->ins_.resize(weights_.size() + 2);
  dnnl_op_ctx_->outs_.resize(1);

  // original weight, bias are always fp32
  memory::data_type dt = memory::data_type::f32;
  memory::dims w_stride =
      transB_ ? memory::dims{1, ldb_} : memory::dims{ldb_, 1};
  memory::desc w_desc({k_, n_}, dt, w_stride);
  dnnl_op_ctx_->ins_[1] =
      std::make_unique<memory>(w_desc, eng, weights_[0]->GetDataPtr());

  dnnl_op_ctx_->attr_ = std::make_unique<dnnl::primitive_attr>();
  dnnl::post_ops po;
  if (weights_.size() == 2) {
    memory::desc b_desc({1, n_}, dt, {n_, 1});
    dnnl_op_ctx_->ins_[2] =
        std::make_unique<memory>(b_desc, eng, weights_[1]->GetDataPtr());
  }

  if (activation_ != UNARYTYPE_UNDEFINED) {
    auto& algo_map = DNNLOpContext::unary_algo_map_;
    if (algo_map.find(activation_) == algo_map.end()) {
      LOG(ERROR) << "Unsupported unary type:" << UnaryType_Name(activation_)
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    float alpha = 0.f;
    float beta = 0.f;
    if (activation_ == UnaryType::SILU) {
      alpha = 1.f;
    }
    po.append_eltwise(algo_map[activation_], alpha, beta);
  }
  dnnl_op_ctx_->attr_->set_post_ops(po);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus GemmOpCPU::Reshape() {
  int yn = n_;
  AS_CHECK_STATUS(GemmOpBase::Reshape(yn));

#ifdef ENABLE_GGML_GEMM
  if (use_ggml_q8_0_) {
    if (dtype_ != DataType::FLOAT32) {
      LOG(ERROR) << "GGML Q8_0 requires FP32 Qwen activations" << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    return AsStatus::ALLSPARK_SUCCESS;
  }
#endif

  const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
  auto eng = DNNLEngine::GetInstance().GetEngine();
  auto dt = weight_data_type_ == DataType::BFLOAT16 ? memory::data_type::bf16
                                                    : memory::data_type::f32;
  memory::desc x_desc({m_, k_}, dt, memory::dims{lda_, 1});
  memory::desc y_desc({m_, n_}, memory::data_type::f32, memory::dims{n_, 1});
  memory::desc w_desc({k_, n_}, dt, tag::any);

  if (reshape_cnt >= 2) {
    // Prevent repeated reorder operations
    w_desc = dnnl_op_ctx_->ins_[1]->get_desc();
  } else {
    reshape_cnt++;
  }

  memory::desc b_desc;
  if (weights_.size() == 2) {
    b_desc = dnnl_op_ctx_->ins_[2]->get_desc();
  }
  if (binary_type_ != BINARYTYPE_UNDEFINED) {
    // auto po = dnnl_op_ctx_->attr_->get_post_ops();
    dnnl::post_ops po;
    auto& algo_map = DNNLOpContext::binary_algo_map_;
    if (algo_map.find(binary_type_) == algo_map.end()) {
      LOG(ERROR) << "Unsupported binary type:" << BinaryType_Name(binary_type_)
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    auto binary_md = std::make_unique<dnnl::memory::desc>(
        memory::dims{m_, n_}, memory::data_type::f32, memory::dims{n_, 1});
    po.append_binary(algo_map[binary_type_], *binary_md);
    dnnl_op_ctx_->attr_->set_post_ops(po);
  }
  dnnl::matmul::primitive_desc prim_desc;
  if (activation_ != UNARYTYPE_UNDEFINED ||
      binary_type_ != BINARYTYPE_UNDEFINED) {
    prim_desc = dnnl::matmul::primitive_desc(eng, x_desc, w_desc, b_desc,
                                             y_desc, *(dnnl_op_ctx_->attr_));
  } else {
    prim_desc =
        dnnl::matmul::primitive_desc(eng, x_desc, w_desc, b_desc, y_desc);
  }

  dnnl_op_ctx_->pr_fwd_[0] = std::make_unique<dnnl::matmul>(prim_desc);
  dnnl_op_ctx_->ins_[0] =
      std::make_unique<dnnl::memory>(prim_desc.src_desc(), eng);
  dnnl_op_ctx_->outs_[0] =
      std::make_unique<dnnl::memory>(prim_desc.dst_desc(), eng);
  if (binary_type_ != BINARYTYPE_UNDEFINED) {
    auto bin_idx = dnnl_op_ctx_->ins_.size() - 1;
    dnnl_op_ctx_->ins_[bin_idx] =
        std::make_unique<dnnl::memory>(prim_desc.dst_desc(), eng);
  }
  if (prim_desc.weights_desc() != dnnl_op_ctx_->ins_[1]->get_desc()) {
    memory::desc wei_src_desc = dnnl_op_ctx_->ins_[1]->get_desc();
    memory::desc wei_dst_desc = prim_desc.weights_desc();
#if 1
    Shape weight_shape = weights_[0]->GetShape();

    int64_t num_elem = wei_dst_desc.get_size() / SizeofType(weight_data_type_);
    auto weight_tmp = std::make_unique<AsTensor>(
        weights_[0]->GetName(), weights_[0]->GetDeviceType(), weight_data_type_,
        weights_[0]->GetDataMode(), Shape({num_elem}));
    auto wei_src_mem = memory(wei_src_desc, eng, weights_[0]->GetDataPtr());
    auto wei_dst_mem = memory(wei_dst_desc, eng, weight_tmp->GetDataPtr());

    dnnl::reorder(wei_src_mem, wei_dst_mem)
        .execute(cpu_ctx->GetStream(), wei_src_mem, wei_dst_mem);

    weights_[0]->Free();
    weights_[0]->SetDataType(weight_data_type_);
    weights_[0]->SetShape(Shape({num_elem}));
    TensorUtils::DeepCopyWholeAsync(*weights_[0], *weight_tmp, ctx_);
    if (weights_[0]->GetSizeInByte() == wei_dst_desc.get_size())
      weights_[0]->SetShape(std::move(weight_shape));
    dnnl_op_ctx_->ins_[1] = std::make_unique<dnnl::memory>(
        wei_dst_desc, eng, weights_[0]->GetDataPtr());
#else
    auto weight_tmp = std::make_unique<AsTensor>(
        weights_[0]->GetName() + "_tmp", *weights_[0]);
    auto wei_src_mem = memory(wei_src_desc, eng, weight_tmp->GetDataPtr());
    dnnl_op_ctx_->ins_[1] = std::make_unique<dnnl::memory>(
        wei_dst_desc, eng, weights_[0]->GetDataPtr());
    dnnl::reorder(wei_src_mem, *dnnl_op_ctx_->ins_[1])
        .execute(cpu_ctx->GetStream(), wei_src_mem, *dnnl_op_ctx_->ins_[1]);
#endif
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmOpCPU::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  void* in = in_tensor->GetDataPtr();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  void* bias = (weights_.size() == 2) ? weights_[1]->GetDataPtr() : nullptr;
  void* bin_in = (in_names_.size() == 2)
                     ? tensor_map_->at(in_names_[1])->GetDataPtr()
                     : nullptr;
  if (is_split_k_) {
    in = (char*)in + k_ * rank_id_ * SizeofType(dtype_);
  }

#ifdef ENABLE_GGML_GEMM
  if (use_ggml_q8_0_) {
    const size_t context_size =
        ggml_tensor_overhead() * 4 + ggml_graph_overhead();
    struct ggml_init_params params = {context_size, nullptr, true};
    struct ggml_context* ggml_context = ggml_init(params);
    if (ggml_context == nullptr) {
      return AsStatus::ALLSPARK_MEMORY_ERROR;
    }

    struct ggml_tensor* weight = ggml_new_tensor_2d(
        ggml_context, GGML_TYPE_Q8_0, k_, n_);
    weight->data = ggml_weight_->data();
    struct ggml_tensor* input =
        ggml_new_tensor_2d(ggml_context, GGML_TYPE_F32, k_, m_);
    input->data = in;
    input->nb[1] = static_cast<size_t>(lda_) * sizeof(float);
    input->nb[2] = input->nb[1] * m_;
    input->nb[3] = input->nb[2];
    struct ggml_tensor* output = ggml_mul_mat(ggml_context, weight, input);
    output->data = out;

    struct ggml_cgraph* graph = ggml_new_graph(ggml_context);
    ggml_build_forward_expand(graph, output);
    struct ggml_cplan plan =
        ggml_graph_plan(graph, ggml_n_threads_, nullptr);
    if (plan.work_size > 0) {
      ggml_work_buffer_.resize(plan.work_size);
      plan.work_data = ggml_work_buffer_.data();
    }
    const enum ggml_status status = ggml_graph_compute(graph, &plan);
    ggml_free(ggml_context);
    if (status != GGML_STATUS_SUCCESS) {
      LOG(ERROR) << "GGML Q8_0 GEMM failed with status "
                 << static_cast<int>(status) << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }

    float* output_data = static_cast<float*>(out);
    const int64_t output_count = m_ * n_;
    if (alpha_ != 1.0f) {
      for (int64_t i = 0; i < output_count; ++i) {
        output_data[i] *= alpha_;
      }
    }
    if (bias != nullptr) {
      const float* bias_data = static_cast<const float*>(bias);
      cpu::parallel_for(m_, [&](int64_t row) {
        for (int64_t column = 0; column < n_; ++column) {
          output_data[static_cast<size_t>(row) * n_ + column] +=
              bias_data[column];
        }
      });
    }
    if (bin_in != nullptr) {
      const float* binary_data = static_cast<const float*>(bin_in);
      for (int64_t i = 0; i < output_count; ++i) {
        output_data[i] += binary_data[i];
      }
    }
    if (activation_ == UnaryType::SILU) {
      for (int64_t i = 0; i < output_count; ++i) {
        output_data[i] /= 1.0f + std::exp(-output_data[i]);
      }
    }
    return AsStatus::ALLSPARK_SUCCESS;
  }
#endif

  const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
  auto eng = DNNLEngine::GetInstance().GetEngine();
  dnnl::memory& x_mem = *(dnnl_op_ctx_->ins_[0]);
  dnnl::memory& w_mem = *(dnnl_op_ctx_->ins_[1]);
  dnnl::memory& y_mem = *(dnnl_op_ctx_->outs_[0]);
  memory::desc x_desc({m_, k_}, memory::data_type::f32, memory::dims{lda_, 1});
  if (x_desc != x_mem.get_desc()) {
    auto x_in_mem = memory(x_desc, eng, in);
    dnnl::reorder(x_in_mem, x_mem)
        .execute(cpu_ctx->GetStream(), x_in_mem, x_mem);
  } else {
    x_mem.set_data_handle(in);
  }
  memory::desc y_desc({m_, n_}, memory::data_type::f32, memory::dims{n_, 1});
  if (y_desc == y_mem.get_desc()) {
    y_mem.set_data_handle(out);
  }
  std::unordered_map<int, memory> args{
      {DNNL_ARG_SRC, x_mem}, {DNNL_ARG_WEIGHTS, w_mem}, {DNNL_ARG_DST, y_mem}};
  if (bias) {
    dnnl::memory& b_mem = *(dnnl_op_ctx_->ins_[2]);
    args.insert({DNNL_ARG_BIAS, b_mem});
  }
  if (binary_type_ != BINARYTYPE_UNDEFINED) {
    dnnl::memory& bin_mem =
        *(dnnl_op_ctx_->ins_[dnnl_op_ctx_->ins_.size() - 1]);
    bin_mem.set_data_handle(bin_in);
    args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, bin_mem});
  }
  dnnl_op_ctx_->pr_fwd_[0]->execute(cpu_ctx->GetStream(), args);
  if (y_desc != y_mem.get_desc()) {
    auto y_out_mem = memory(y_desc, eng, out);
    dnnl::reorder(y_mem, y_out_mem)
        .execute(cpu_ctx->GetStream(), y_mem, y_out_mem);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

#ifndef ENABLE_ARM_V84_V9
REGISTER_OP(Gemm, CPU, GemmOpCPU)
#endif

}  // namespace allspark
