#!/usr/bin/env python3
# Copyright (c) 2026 Segno System.

"""Create a dense-only miniature DeepSeek-V3 ModelOpt NVFP4 checkpoint.

The fixture keeps MLA projection weights in BF16 and stores attention output
and dense FFN weights in their native packed FP4 representation. It is sized
for the SM100 kernel alignment contract and intentionally contains no MoE
layer while native FP4 expert GEMM remains out of scope.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _rand_bf16(*shape):
    return torch.randn(*shape, dtype=torch.bfloat16) * 0.01


def _add_nvfp4_linear(tensors, base_name, n_dim, k_dim):
    if n_dim % 128 or k_dim % 64:
        raise ValueError(
            f"NVFP4 dimensions must satisfy N%128=0 and K%64=0: "
            f"{base_name} has N={n_dim}, K={k_dim}")
    tensors[base_name + ".weight"] = torch.randint(
        0, 256, (n_dim, k_dim // 2), dtype=torch.uint8)
    tensors[base_name + ".weight_scale"] = (
        torch.empty(n_dim, k_dim // 16, dtype=torch.float32)
        .uniform_(0.015625, 0.125)
        .to(torch.float8_e4m3fn)
    )
    tensors[base_name + ".weight_scale_2"] = torch.tensor(
        [0.001], dtype=torch.float32)
    tensors[base_name + ".input_scale"] = torch.tensor(
        [0.1], dtype=torch.float32)


def _verify_checkpoint(output_dir, config, tensors, shard_name):
    with open(output_dir / "config.json", encoding="utf-8") as config_file:
        saved_config = json.load(config_file)
    with open(output_dir / "hf_quant_config.json",
              encoding="utf-8") as quant_file:
        quant_config = json.load(quant_file)
    with open(output_dir / "model.safetensors.index.json",
              encoding="utf-8") as index_file:
        saved_index = json.load(index_file)

    if saved_config != config:
        raise RuntimeError("saved config does not match generated config")
    if quant_config["quantization"]["quant_algo"] != "NVFP4":
        raise RuntimeError("checkpoint does not advertise NVFP4")
    if set(saved_index["weight_map"]) != set(tensors):
        raise RuntimeError("safetensors index does not cover every tensor")
    if set(saved_index["weight_map"].values()) != {shard_name}:
        raise RuntimeError("safetensors index points at an unexpected shard")

    packed_count = 0
    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as f:
        if set(f.keys()) != set(tensors):
            raise RuntimeError("saved shard keys do not match generated weights")
        for name, expected in tensors.items():
            actual = f.get_tensor(name)
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise RuntimeError(f"tensor contract mismatch for {name}")
            if name.endswith(".weight") and actual.dtype == torch.uint8:
                packed_count += 1
                n_dim, packed_k = actual.shape
                if n_dim % 128 or (packed_k * 2) % 64:
                    raise RuntimeError(f"unaligned packed tensor {name}")

    if packed_count != config["num_hidden_layers"] * 4:
        raise RuntimeError(
            f"expected four packed linears per layer, got {packed_count}")
    if any(".mlp.experts." in name for name in tensors):
        raise RuntimeError("dense-only mini checkpoint unexpectedly contains MoE")
    print(f"  Verified: {packed_count} packed NVFP4 linear tensors")


def create_tiny_dsv3_nvfp4(output_dir, num_layers=2, seed=20250401):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    hidden_size = 256
    intermediate_size = 512
    num_attention_heads = 4
    kv_lora_rank = 64
    q_lora_rank = 128
    qk_nope_head_dim = 64
    qk_rope_head_dim = 64
    v_head_dim = 64
    vocab_size = 512

    if num_layers < 1:
        raise ValueError("num_layers must be positive")

    config = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_deepseek.DeepseekV3Config",
        },
        "model_type": "deepseek_v3",
        "torch_dtype": "bfloat16",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "moe_intermediate_size": 128,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_attention_heads,
        "kv_lora_rank": kv_lora_rank,
        "q_lora_rank": q_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "vocab_size": vocab_size,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": num_layers,
        "n_shared_experts": 1,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "max_position_embeddings": 128,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "num_nextn_predict_layers": 0,
        "ep_size": 1,
        "moe_layer_freq": 1,
    }

    tensors = {
        "model.embed_tokens.weight": _rand_bf16(vocab_size, hidden_size),
        "model.norm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
        "lm_head.weight": _rand_bf16(vocab_size, hidden_size),
    }

    for layer_index in range(num_layers):
        prefix = f"model.layers.{layer_index}"
        tensors[f"{prefix}.input_layernorm.weight"] = torch.ones(
            hidden_size, dtype=torch.bfloat16)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(
            hidden_size, dtype=torch.bfloat16)

        tensors[f"{prefix}.self_attn.q_a_proj.weight"] = _rand_bf16(
            q_lora_rank, hidden_size)
        tensors[f"{prefix}.self_attn.q_a_layernorm.weight"] = torch.ones(
            q_lora_rank, dtype=torch.bfloat16)
        tensors[f"{prefix}.self_attn.q_b_proj.weight"] = _rand_bf16(
            num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim),
            q_lora_rank)
        tensors[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = \
            _rand_bf16(kv_lora_rank + qk_rope_head_dim, hidden_size)
        tensors[f"{prefix}.self_attn.kv_a_layernorm.weight"] = torch.ones(
            kv_lora_rank, dtype=torch.bfloat16)
        tensors[f"{prefix}.self_attn.kv_b_proj.weight"] = _rand_bf16(
            num_attention_heads * (qk_nope_head_dim + v_head_dim),
            kv_lora_rank)

        _add_nvfp4_linear(
            tensors, f"{prefix}.self_attn.o_proj", hidden_size,
            num_attention_heads * v_head_dim)
        _add_nvfp4_linear(
            tensors, f"{prefix}.mlp.gate_proj", intermediate_size,
            hidden_size)
        _add_nvfp4_linear(
            tensors, f"{prefix}.mlp.up_proj", intermediate_size,
            hidden_size)
        _add_nvfp4_linear(
            tensors, f"{prefix}.mlp.down_proj", hidden_size,
            intermediate_size)

    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, str(output_dir / shard_name))
    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {name: shard_name for name in tensors},
    }
    with open(output_dir / "model.safetensors.index.json", "w",
              encoding="utf-8") as index_file:
        json.dump(index, index_file, indent=2)
    with open(output_dir / "config.json", "w",
              encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
    with open(output_dir / "hf_quant_config.json", "w",
              encoding="utf-8") as quant_file:
        json.dump({
            "producer": {"name": "modelopt", "version": "0.22.0"},
            "quantization": {"quant_algo": "NVFP4", "group_size": 16},
        }, quant_file, indent=2)

    config_module = '''# Copyright (c) 2026 Segno System.
import torch
from transformers import PretrainedConfig


class DeepseekV3Config(PretrainedConfig):
    model_type = "deepseek_v3"

    def __init__(self, **kwargs):
        dtype = kwargs.get("torch_dtype")
        if isinstance(dtype, str):
            kwargs["torch_dtype"] = getattr(torch, dtype)
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
'''
    (output_dir / "configuration_deepseek.py").write_text(
        config_module, encoding="utf-8")
    with open(output_dir / "generation_config.json", "w",
              encoding="utf-8") as generation_file:
        json.dump({
            "do_sample": True,
            "eos_token_id": config["eos_token_id"],
            "max_length": 16,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        }, generation_file, indent=2)

    _verify_checkpoint(output_dir, config, tensors, shard_name)
    print("Created tiny DeepSeek-V3 NVFP4 model:")
    print(f"  Layers: {num_layers} (dense-only)")
    print(f"  Seed: {seed}")
    print(f"  Size on disk: {total_size / 1e6:.1f}MB")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20250401)
    args = parser.parse_args()
    create_tiny_dsv3_nvfp4(args.output, args.num_layers, args.seed)
