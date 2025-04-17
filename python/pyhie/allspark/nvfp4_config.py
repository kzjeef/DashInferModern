# Copyright (c) 2026 Segno System.
"""Metadata detection for packed NVIDIA ModelOpt NVFP4 checkpoints."""

import json
import os
from typing import Any, Dict


def detect_modelopt_nvfp4(model_path: str) -> Dict[str, Any]:
    """Return the packed-weight contract advertised by a ModelOpt checkpoint.

    ModelOpt commonly writes the quantization record to
    ``hf_quant_config.json``. Some exported checkpoints instead embed the same
    record in ``config.json``. Detection is deliberately metadata-only:
    weights remain packed and are never converted to a higher-precision type.
    """
    result = {
        "enabled": False,
        "block_size": 16,
        "source": None,
    }
    if not os.path.isdir(model_path):
        return result

    candidates = (
        ("hf_quant_config.json", ("quantization",)),
        ("config.json", ("quantization_config",)),
    )
    for filename, path in candidates:
        config_path = os.path.join(model_path, filename)
        if not os.path.isfile(config_path):
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "failed to read NVFP4 metadata from {}: {}".format(
                    config_path, error)) from error

        quant_config = config
        for key in path:
            quant_config = quant_config.get(key, {})
            if not isinstance(quant_config, dict):
                quant_config = {}
                break

        algorithm = str(quant_config.get("quant_algo", "")).upper()
        method = str(quant_config.get("quant_method", "")).lower()
        if algorithm != "NVFP4" and method != "nvfp4":
            continue

        try:
            block_size = int(quant_config.get("group_size", 16))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "NVFP4 group_size in {} must be an integer".format(
                    config_path)) from error
        if block_size <= 0:
            raise ValueError(
                "NVFP4 group_size in {} must be positive".format(config_path))

        result.update({
            "enabled": True,
            "block_size": block_size,
            "source": filename,
        })
        return result

    return result
