# Copyright (c) 2026 Segno System.
"""Small, dependency-free helpers for supported Qwen vision models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VLModelSpec:
    family: str
    architecture: str
    runtime_name: str


_MODEL_SPECS = {
    "Qwen2VLForConditionalGeneration": VLModelSpec(
        family="qwen2_vl",
        architecture="Qwen2VLForConditionalGeneration",
        runtime_name="QWEN2-VL",
    ),
    "Qwen2_5_VLForConditionalGeneration": VLModelSpec(
        family="qwen2_5_vl",
        architecture="Qwen2_5_VLForConditionalGeneration",
        runtime_name="QWEN2.5-VL",
    ),
}

_MODEL_TYPE_TO_ARCHITECTURE = {
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
}


def _read(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def detect_vl_model(config):
    """Return a stable model spec or fail before heavyweight model loading."""
    architectures = _read(config, "architectures", ()) or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    for architecture in architectures:
        if architecture in _MODEL_SPECS:
            return _MODEL_SPECS[architecture]

    architecture = _MODEL_TYPE_TO_ARCHITECTURE.get(
        _read(config, "model_type"))
    if architecture is not None:
        return _MODEL_SPECS[architecture]

    supported = ", ".join(sorted(_MODEL_SPECS))
    raise ValueError(
        "unsupported vision-language architecture; expected one of "
        f"{supported}")


def get_text_config(config):
    """Return the language config used by the AllSpark text serializer."""
    return _read(config, "text_config", None) or config


def get_vision_config(config):
    """Return the nested vision config when the wrapper exposes one."""
    vision_config = _read(config, "vision_config", None)
    if vision_config is None:
        raise ValueError("vision-language config is missing vision_config")
    return vision_config
