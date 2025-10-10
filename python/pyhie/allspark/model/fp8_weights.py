# Copyright (c) 2026 Segno System.
"""Packed-preserving helpers for native block-wise FP8 checkpoints."""

import math

import torch


FP8_SCALE_SUFFIX = ".weight_scale_inv"


def _fp8_e4m3_dtype():
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("this PyTorch build does not expose FP8 E4M3")
    return dtype


def validate_fp8_linear(state_dict, weight_name, block_size=(128, 128)):
    """Validate one native FP8 linear and return its logical ``(N, K)``."""
    if (not isinstance(block_size, (list, tuple)) or len(block_size) != 2 or
            int(block_size[0]) <= 0 or int(block_size[1]) <= 0):
        raise ValueError("FP8 block_size must contain two positive integers")
    if not weight_name.endswith(".weight"):
        raise ValueError("FP8 weight name must end with '.weight'")

    scale_name = weight_name[:-len(".weight")] + FP8_SCALE_SUFFIX
    missing = [name for name in (weight_name, scale_name)
               if name not in state_dict]
    if missing:
        raise KeyError(
            "FP8 linear {} is missing {}".format(weight_name, missing))

    weight = state_dict[weight_name]
    scale = state_dict[scale_name]
    if weight.dtype != _fp8_e4m3_dtype() or weight.ndim != 2:
        raise ValueError(
            "{} must be a 2D FP8 E4M3 tensor".format(weight_name))
    if scale.dtype != torch.float32 or scale.ndim != 2:
        raise ValueError(
            "{} must be a 2D FP32 tensor".format(scale_name))

    n_dim, k_dim = weight.shape
    block_n, block_k = int(block_size[0]), int(block_size[1])
    expected_scale_shape = (
        math.ceil(n_dim / block_n),
        math.ceil(k_dim / block_k),
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "{} must have shape {}, got {}".format(
                scale_name, expected_scale_shape, tuple(scale.shape)))
    return n_dim, k_dim


def merge_fp8_linears(state_dict, first_weight, second_weight, merged_weight,
                       block_size=(128, 128)):
    """Concatenate compatible FP8 linears without changing their dtype."""
    first_shape = validate_fp8_linear(
        state_dict, first_weight, block_size)
    second_shape = validate_fp8_linear(
        state_dict, second_weight, block_size)
    if first_shape[1] != second_shape[1]:
        raise ValueError(
            "cannot merge FP8 linears with different K dimensions")

    state_dict[merged_weight] = torch.cat(
        (state_dict[first_weight], state_dict[second_weight]), dim=0)
    merged_scale = merged_weight[:-len(".weight")] + FP8_SCALE_SUFFIX
    first_scale = first_weight[:-len(".weight")] + FP8_SCALE_SUFFIX
    second_scale = second_weight[:-len(".weight")] + FP8_SCALE_SUFFIX
    state_dict[merged_scale] = torch.cat(
        (state_dict[first_scale], state_dict[second_scale]), dim=0)
    return validate_fp8_linear(state_dict, merged_weight, block_size)
