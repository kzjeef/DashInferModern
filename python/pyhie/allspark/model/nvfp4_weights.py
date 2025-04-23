# Copyright (c) 2026 Segno System.
"""Packed-weight helpers for NVIDIA ModelOpt NVFP4 checkpoints."""

import torch


NVFP4_AUX_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
)


def validate_nvfp4_linear(state_dict, weight_name, block_size=16):
    """Validate one packed NVFP4 linear and return its logical ``(N, K)``."""
    if block_size <= 0:
        raise ValueError("NVFP4 block_size must be positive")
    if not weight_name.endswith(".weight"):
        raise ValueError("NVFP4 weight name must end with '.weight'")

    base_name = weight_name[:-len(".weight")]
    required = (weight_name,) + tuple(
        base_name + suffix for suffix in NVFP4_AUX_SUFFIXES)
    missing = [name for name in required if name not in state_dict]
    if missing:
        raise KeyError(
            "NVFP4 linear {} is missing {}".format(weight_name, missing))

    packed = state_dict[weight_name]
    block_scale = state_dict[base_name + ".weight_scale"]
    global_scale = state_dict[base_name + ".weight_scale_2"]
    input_scale = state_dict[base_name + ".input_scale"]

    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError(
            "{} must be a 2D uint8 packed tensor".format(weight_name))
    n_dim, packed_k = packed.shape
    k_dim = packed_k * 2
    if k_dim % block_size != 0:
        raise ValueError(
            "{} logical K={} is not divisible by block_size={}".format(
                weight_name, k_dim, block_size))

    expected_scale_shape = (n_dim, k_dim // block_size)
    if (block_scale.dtype != torch.float8_e4m3fn
            or tuple(block_scale.shape) != expected_scale_shape):
        raise ValueError(
            "{}.weight_scale must be FP8 E4M3 with shape {}".format(
                base_name, expected_scale_shape))

    for suffix, scalar in (
            ("weight_scale_2", global_scale),
            ("input_scale", input_scale)):
        if scalar.dtype != torch.float32 or scalar.numel() != 1:
            raise ValueError(
                "{}.{} must be one FP32 value".format(base_name, suffix))

    return n_dim, k_dim


def merge_nvfp4_linears(state_dict, first_weight, second_weight,
                         merged_weight, block_size=16):
    """Concatenate two compatible NVFP4 linears without unpacking FP4 data."""
    first_shape = validate_nvfp4_linear(
        state_dict, first_weight, block_size)
    second_shape = validate_nvfp4_linear(
        state_dict, second_weight, block_size)
    if first_shape[1] != second_shape[1]:
        raise ValueError(
            "cannot merge NVFP4 linears with different K dimensions")

    first_base = first_weight[:-len(".weight")]
    second_base = second_weight[:-len(".weight")]
    merged_base = merged_weight[:-len(".weight")]

    for suffix in ("weight_scale_2", "input_scale"):
        first_scalar = state_dict[first_base + "." + suffix]
        second_scalar = state_dict[second_base + "." + suffix]
        if not torch.equal(first_scalar, second_scalar):
            raise ValueError(
                "cannot merge NVFP4 linears with different {} values".format(
                    suffix))

    state_dict[merged_weight] = torch.cat(
        (state_dict[first_weight], state_dict[second_weight]), dim=0)
    state_dict[merged_base + ".weight_scale"] = torch.cat(
        (state_dict[first_base + ".weight_scale"],
         state_dict[second_base + ".weight_scale"]), dim=0)
    state_dict[merged_base + ".weight_scale_2"] = \
        state_dict[first_base + ".weight_scale_2"].clone()
    state_dict[merged_base + ".input_scale"] = \
        state_dict[first_base + ".input_scale"].clone()

    return validate_nvfp4_linear(state_dict, merged_weight, block_size)
