#!/usr/bin/env python3
# Copyright (c) 2026 Segno System.

"""Create a self-contained miniature DeepSeek-V3 BF16 checkpoint.

The default fixture is intentionally small enough for a fast load-and-generate
smoke test. It preserves the important DeepSeek-V3 structures (MLA, one dense
FFN layer, one routed-MoE layer, a shared expert, and grouped sigmoid routing)
without preserving the production model's width.

Usage:
    python create_tiny_dsv3.py --output /tmp/tiny-dsv3
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def create_tiny_dsv3(output_dir, num_layers=2, num_experts=4,
                     seed=20250301):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    # qk_nope + qk_rope remains 128 so this uses the flash-attention head
    # dimension already built by DashInfer.
    hidden_size = 512
    intermediate_size = 1024
    moe_intermediate_size = 256
    num_attention_heads = 8
    kv_lora_rank = 64
    q_lora_rank = 128
    qk_nope_head_dim = 64
    qk_rope_head_dim = 64
    v_head_dim = 64
    vocab_size = 512
    first_k_dense_replace = 1
    n_shared_experts = 1

    if num_layers < 2:
        raise ValueError("num_layers must be at least 2 to cover dense and MoE paths")
    if num_experts < 2 or num_experts % 2 != 0:
        raise ValueError("num_experts must be an even integer >= 2")

    config = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_deepseek.DeepseekV3Config"
        },
        "model_type": "deepseek_v3",
        "torch_dtype": "bfloat16",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "moe_intermediate_size": moe_intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_attention_heads,
        "kv_lora_rank": kv_lora_rank,
        "q_lora_rank": q_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "vocab_size": vocab_size,
        "n_routed_experts": num_experts,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": first_k_dense_replace,
        "n_shared_experts": n_shared_experts,
        "n_group": 2,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
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

    def rand_bf16(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16) * 0.01

    tensors = {
        "model.embed_tokens.weight": rand_bf16(vocab_size, hidden_size),
        "model.norm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
        "lm_head.weight": rand_bf16(vocab_size, hidden_size),
    }

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        tensors[f"{prefix}.input_layernorm.weight"] = torch.ones(
            hidden_size, dtype=torch.bfloat16)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(
            hidden_size, dtype=torch.bfloat16)

        tensors[f"{prefix}.self_attn.q_a_proj.weight"] = rand_bf16(
            q_lora_rank, hidden_size)
        tensors[f"{prefix}.self_attn.q_a_layernorm.weight"] = torch.ones(
            q_lora_rank, dtype=torch.bfloat16)
        tensors[f"{prefix}.self_attn.q_b_proj.weight"] = rand_bf16(
            num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim),
            q_lora_rank)
        tensors[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = rand_bf16(
            kv_lora_rank + qk_rope_head_dim, hidden_size)
        tensors[f"{prefix}.self_attn.kv_a_layernorm.weight"] = torch.ones(
            kv_lora_rank, dtype=torch.bfloat16)
        tensors[f"{prefix}.self_attn.kv_b_proj.weight"] = rand_bf16(
            num_attention_heads * (qk_nope_head_dim + v_head_dim),
            kv_lora_rank)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = rand_bf16(
            hidden_size, num_attention_heads * v_head_dim)

        if i < first_k_dense_replace:
            tensors[f"{prefix}.mlp.gate_proj.weight"] = rand_bf16(
                intermediate_size, hidden_size)
            tensors[f"{prefix}.mlp.up_proj.weight"] = rand_bf16(
                intermediate_size, hidden_size)
            tensors[f"{prefix}.mlp.down_proj.weight"] = rand_bf16(
                hidden_size, intermediate_size)
        else:
            tensors[f"{prefix}.mlp.gate.weight"] = rand_bf16(
                num_experts, hidden_size)
            tensors[f"{prefix}.mlp.gate.e_score_correction_bias"] = torch.zeros(
                num_experts, dtype=torch.bfloat16)

            tensors[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = rand_bf16(
                moe_intermediate_size, hidden_size)
            tensors[f"{prefix}.mlp.shared_experts.up_proj.weight"] = rand_bf16(
                moe_intermediate_size, hidden_size)
            tensors[f"{prefix}.mlp.shared_experts.down_proj.weight"] = rand_bf16(
                hidden_size, moe_intermediate_size)

            for j in range(num_experts):
                tensors[f"{prefix}.mlp.experts.{j}.gate_proj.weight"] = rand_bf16(
                    moe_intermediate_size, hidden_size)
                tensors[f"{prefix}.mlp.experts.{j}.up_proj.weight"] = rand_bf16(
                    moe_intermediate_size, hidden_size)
                tensors[f"{prefix}.mlp.experts.{j}.down_proj.weight"] = rand_bf16(
                    hidden_size, moe_intermediate_size)

    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, str(output_dir / shard_name))

    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {key: shard_name for key in tensors},
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # direct_load=True only needs AutoConfig, so the fixture is independent of
    # a locally downloaded production DeepSeek repository.
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
    with open(output_dir / "configuration_deepseek.py", "w") as f:
        f.write(config_module)

    generation_config = {
        "do_sample": True,
        "eos_token_id": config["eos_token_id"],
        "max_length": 16,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
    }
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(generation_config, f, indent=2)

    n_params = sum(t.numel() for t in tensors.values())
    print("Created tiny DeepSeek-V3 model:")
    print(f"  Layers: {num_layers}")
    print(f"  Experts: {num_experts}")
    print(f"  Dense layers: {first_k_dense_replace}")
    print(f"  Seed: {seed}")
    print(f"  Parameters: {n_params / 1e6:.1f}M")
    print(f"  Size on disk: {total_size / 1e6:.1f}MB")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20250301)
    args = parser.parse_args()
    create_tiny_dsv3(
        args.output, args.num_layers, args.num_experts, args.seed)
