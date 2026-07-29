#!/usr/bin/env python3
# Copyright (c) 2026 Segno System.

"""Run a deterministic one-layer Qwen model on Apple Silicon with GGML Q8_0."""

import argparse
import tempfile
import time
from pathlib import Path

import torch

from dashinfer import allspark
from dashinfer.allspark._allspark import AsStatus, GenerateRequestStatus
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark.generation_config import ASGenerationConfigBuilder
from dashinfer.allspark.runtime_config import AsModelRuntimeConfigBuilder


MODEL_NAME = "qwen_mini_ggml"
VOCAB_SIZE = 64
HIDDEN_SIZE = 32
INTERMEDIATE_SIZE = 64
NUM_HEADS = 2
HEAD_SIZE = HIDDEN_SIZE // NUM_HEADS


def make_qwen_weights(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def weight(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.02

    return {
        "model.embed_tokens.weight": weight(VOCAB_SIZE, HIDDEN_SIZE),
        "model.norm.weight": torch.ones(HIDDEN_SIZE, dtype=torch.float32),
        "lm_head.weight": weight(VOCAB_SIZE, HIDDEN_SIZE),
        "model.layers.0.input_layernorm.weight": torch.ones(
            HIDDEN_SIZE, dtype=torch.float32),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(
            HIDDEN_SIZE, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": weight(
            HIDDEN_SIZE, HIDDEN_SIZE),
        "model.layers.0.self_attn.k_proj.weight": weight(
            HIDDEN_SIZE, HIDDEN_SIZE),
        "model.layers.0.self_attn.v_proj.weight": weight(
            HIDDEN_SIZE, HIDDEN_SIZE),
        "model.layers.0.self_attn.q_proj.bias": weight(HIDDEN_SIZE),
        "model.layers.0.self_attn.k_proj.bias": weight(HIDDEN_SIZE),
        "model.layers.0.self_attn.v_proj.bias": weight(HIDDEN_SIZE),
        "model.layers.0.self_attn.o_proj.weight": weight(
            HIDDEN_SIZE, HIDDEN_SIZE),
        "model.layers.0.mlp.gate_proj.weight": weight(
            INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "model.layers.0.mlp.up_proj.weight": weight(
            INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "model.layers.0.mlp.down_proj.weight": weight(
            HIDDEN_SIZE, INTERMEDIATE_SIZE),
    }


def serialize_qwen(engine, output_dir, seed):
    model_config = {
        "model_type": "Qwen_v20",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": 1,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_HEADS,
        "size_per_head": HEAD_SIZE,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rotary_emb_base": 10000.0,
    }
    engine.serialize_model_from_torch(
        model_name=MODEL_NAME,
        model_type="Qwen_v20",
        torch_model=make_qwen_weights(seed),
        model_config=model_config,
        save_dir=str(output_dir),
        data_type="float32",
        multigpu_mode=0,
        do_binary_add_fused=True,
        rotary_base=model_config["rotary_emb_base"],
        use_ggml_q8_0=True,
    )


def run_qwen(output_dir, max_length, threads, seed, timeout_seconds):
    input_ids = [1, 2, 3, 4]
    if max_length <= len(input_ids):
        raise ValueError(
            f"--max-length must be greater than {len(input_ids)}")
    if threads < 0:
        raise ValueError("--threads must be non-negative; use 0 for auto")
    if timeout_seconds <= 0:
        raise ValueError("--timeout must be positive")

    engine = allspark.Engine()
    serialize_qwen(engine, output_dir, seed)
    runtime_max_length = max(16, max_length)

    runtime_config = (
        AsModelRuntimeConfigBuilder()
        .model_name(MODEL_NAME)
        .model_dir(str(output_dir), MODEL_NAME)
        .compute_unit(TargetDevice.CPU, [0], threads)
        .max_batch(1)
        .max_length(runtime_max_length)
        .max_prefill_length(runtime_max_length)
        .build()
    )

    input_tensor = torch.tensor([input_ids], dtype=torch.int64)
    inputs = {
        "input_ids": torch.utils.dlpack.to_dlpack(input_tensor),
    }
    generation = ASGenerationConfigBuilder(seed=seed, eos_token_id=VOCAB_SIZE - 1)
    generation.update({
        "early_stopping": False,
        "max_length": max_length,
        "top_k": 1,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    })

    installed = False
    started = False
    handle = None
    try:
        engine.install_model(runtime_config)
        installed = True
        if engine.start_model(MODEL_NAME) != AsStatus.ALLSPARK_SUCCESS:
            raise RuntimeError("failed to start Qwen mini")
        started = True

        status, handle, queue = engine.start_request(
            MODEL_NAME, inputs, generation.build())
        if status != AsStatus.ALLSPARK_SUCCESS:
            raise RuntimeError(f"failed to start Qwen mini request: {status}")

        generated_ids = []
        deadline = time.monotonic() + timeout_seconds
        while queue.GenerateStatus() in (
                GenerateRequestStatus.Init,
                GenerateRequestStatus.Generating,
                GenerateRequestStatus.ContextFinished):
            if time.monotonic() >= deadline:
                raise TimeoutError("Qwen mini generation timed out")
            element = queue.Get()
            if element is not None:
                generated_ids.extend(element.ids_from_generate)
            else:
                time.sleep(0.001)

        if queue.GenerateStatus() != GenerateRequestStatus.GenerateFinished:
            raise RuntimeError(
                f"Qwen mini generation did not finish: {queue.GenerateStatus()}")
        if not generated_ids:
            raise RuntimeError("Qwen mini produced no tokens")
        if not all(0 <= token < VOCAB_SIZE for token in generated_ids):
            raise RuntimeError(f"Qwen mini produced an invalid token: {generated_ids}")
        return generated_ids
    finally:
        if handle is not None:
            engine.release_request(MODEL_NAME, handle)
        if started:
            engine.stop_model(MODEL_NAME)
        if installed:
            engine.release_model(MODEL_NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        tokens = run_qwen(
            args.output_dir, args.max_length, args.threads, args.seed,
            args.timeout)
    else:
        with tempfile.TemporaryDirectory(prefix="dashinfer-qwen-mini-") as tmp:
            tokens = run_qwen(
                Path(tmp), args.max_length, args.threads, args.seed,
                args.timeout)
    print("PASS: Qwen mini GGML Q8_0 prefill/decode", tokens)


if __name__ == "__main__":
    main()
