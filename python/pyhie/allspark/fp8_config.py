# Copyright (c) 2026 Segno System.
"""Metadata-only detection for native FP8 checkpoints."""

import json
import os
from typing import Any, Dict, Tuple


def _parse_block_size(value: Any, source: str) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "FP8 weight_block_size in {} must contain two integers".format(
                source))
    try:
        block_size = (int(value[0]), int(value[1]))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "FP8 weight_block_size in {} must contain two integers".format(
                source)) from error
    if block_size[0] <= 0 or block_size[1] <= 0:
        raise ValueError(
            "FP8 weight_block_size in {} must be positive".format(source))
    return block_size


def detect_native_fp8(model_path: str) -> Dict[str, Any]:
    """Inspect checkpoint metadata without materializing or casting weights."""
    result = {
        "enabled": False,
        "format": None,
        "block_size": (128, 128),
        "source": None,
    }
    if not os.path.isdir(model_path):
        return result

    sidecar_path = os.path.join(model_path, "hf_quant_config.json")
    if os.path.isfile(sidecar_path):
        try:
            with open(sidecar_path, "r", encoding="utf-8") as config_file:
                sidecar = json.load(config_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "failed to read FP8 metadata from {}: {}".format(
                    sidecar_path, error)) from error
        quantization = sidecar.get("quantization", {})
        if isinstance(quantization, dict) and str(
                quantization.get("quant_algo", "")).upper() == "FP8":
            result.update({
                "enabled": True,
                "format": "per_tensor",
                "source": "hf_quant_config.json",
            })
            return result

    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "failed to read FP8 metadata from {}: {}".format(
                    config_path, error)) from error
        quantization = config.get("quantization_config") or {}
        if not quantization:
            text_config = config.get("text_config") or {}
            quantization = text_config.get("quantization_config") or {}
        if (isinstance(quantization, dict) and
                str(quantization.get("quant_method", "")).lower() == "fp8"):
            block_size = _parse_block_size(
                quantization.get("weight_block_size", (128, 128)),
                config_path)
            result.update({
                "enabled": True,
                "format": "blockwise",
                "block_size": block_size,
                "source": "config.json",
            })
            return result

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as index_file:
                index = json.load(index_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "failed to read FP8 tensor index from {}: {}".format(
                    index_path, error)) from error
        weight_map = index.get("weight_map", {})
        if isinstance(weight_map, dict) and any(
                name.endswith(".weight_scale_inv") for name in weight_map):
            result.update({
                "enabled": True,
                "format": "blockwise",
                "source": "model.safetensors.index.json",
            })

    return result
