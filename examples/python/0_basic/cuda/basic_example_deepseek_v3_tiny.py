#!/usr/bin/env python3
# Copyright (c) 2026 Segno System.

"""Run the generated miniature DeepSeek-V3 checkpoint on one CUDA GPU."""

import argparse

from dashinfer import allspark
from dashinfer.allspark._allspark import AsStatus, GenerateRequestStatus
from dashinfer.allspark.engine import TargetDevice


def main(model_dir, max_length):
    model_name = "tiny_deepseek_v3"
    loader = allspark.HuggingFaceModel(
        model_dir,
        model_name,
        in_memory_serialize=True,
        user_set_data_type="bfloat16",
        trust_remote_code=True,
    )
    engine = allspark.Engine()

    (loader.load_model(direct_load=True)
     .serialize_to_memory(engine, multinode_mode=False)
     .free_model())

    runtime_builder = loader.create_reference_runtime_config_builder(
        model_name, TargetDevice.CUDA, [0], max_batch=1,
        max_context_length=max_length)
    runtime_config = runtime_builder.build()

    if engine.install_model(runtime_config) != AsStatus.ALLSPARK_SUCCESS:
        raise RuntimeError("failed to install tiny DeepSeek-V3")
    if engine.start_model(model_name) != AsStatus.ALLSPARK_SUCCESS:
        raise RuntimeError("failed to start tiny DeepSeek-V3")

    generation = loader.create_reference_generation_config_builder(
        runtime_config)
    generation.update({
        "do_sample": True,
        "top_k": 1,
        "max_length": min(max_length, 12),
        "eos_token_id": 1,
    })
    status, handle, queue = engine.start_request_ids(
        model_name, loader, [2, 3, 4, 5], generation)
    if status != AsStatus.ALLSPARK_SUCCESS:
        raise RuntimeError(f"failed to start request: {status}")

    generated_ids = []
    while queue.GenerateStatus() in (
            GenerateRequestStatus.Init,
            GenerateRequestStatus.Generating,
            GenerateRequestStatus.ContextFinished):
        element = queue.Get()
        if element is not None:
            generated_ids.extend(element.ids_from_generate)

    final_status = queue.GenerateStatus()
    engine.release_request(model_name, handle)
    engine.stop_model(model_name)
    engine.release_model(model_name)
    loader.free_memory_serialize_file()

    if final_status != GenerateRequestStatus.GenerateFinished:
        raise RuntimeError(f"generation did not finish: {final_status}")
    if not generated_ids:
        raise RuntimeError("generation completed without producing a token")
    print("PASS: tiny DeepSeek-V3 prefill/decode", generated_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--max-length", type=int, default=32)
    args = parser.parse_args()
    main(args.model_dir, args.max_length)
