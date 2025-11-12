# Copyright (c) 2026 Segno System.
"""Unit tests for dependency-free Qwen VL configuration routing."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "multimodal"
    / "dashinfer_vlm"
    / "vl_inference"
    / "utils"
    / "vl_model_config.py"
)
SPEC = importlib.util.spec_from_file_location("vl_model_config", MODULE_PATH)
VL_MODEL_CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VL_MODEL_CONFIG)

RUNTIME_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "multimodal"
    / "dashinfer_vlm"
    / "vl_inference"
    / "utils"
    / "config"
    / "config.py"
)
RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "vl_runtime_config", RUNTIME_CONFIG_PATH)
VL_RUNTIME_CONFIG = importlib.util.module_from_spec(RUNTIME_SPEC)
RUNTIME_SPEC.loader.exec_module(VL_RUNTIME_CONFIG)


class VLModelConfigTest(unittest.TestCase):

    def test_detects_qwen2_architecture(self):
        config = {"architectures": ["Qwen2VLForConditionalGeneration"]}
        spec = VL_MODEL_CONFIG.detect_vl_model(config)
        self.assertEqual("qwen2_vl", spec.family)
        self.assertEqual("QWEN2-VL", spec.runtime_name)

    def test_detects_qwen25_architecture(self):
        config = SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"])
        spec = VL_MODEL_CONFIG.detect_vl_model(config)
        self.assertEqual("qwen2_5_vl", spec.family)
        self.assertEqual("QWEN2.5-VL", spec.runtime_name)

    def test_falls_back_to_model_type(self):
        spec = VL_MODEL_CONFIG.detect_vl_model({"model_type": "qwen2_5_vl"})
        self.assertEqual("Qwen2_5_VLForConditionalGeneration",
                         spec.architecture)

    def test_rejects_unknown_architecture(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            VL_MODEL_CONFIG.detect_vl_model({"architectures": ["OtherVL"]})

    def test_returns_nested_configs(self):
        text_config = object()
        vision_config = object()
        config = SimpleNamespace(
            text_config=text_config, vision_config=vision_config)
        self.assertIs(text_config, VL_MODEL_CONFIG.get_text_config(config))
        self.assertIs(vision_config, VL_MODEL_CONFIG.get_vision_config(config))

    def test_requires_vision_config(self):
        with self.assertRaisesRegex(ValueError, "vision_config"):
            VL_MODEL_CONFIG.get_vision_config({})

    def test_normalizes_qwen25_runtime_aliases(self):
        aliases = ("QWEN2.5-VL", "qwen2_5_vl", "qwen2-5-vl")
        for alias in aliases:
            with self.subTest(alias=alias):
                self.assertEqual(
                    "QWEN2-VL",
                    VL_RUNTIME_CONFIG.normalize_model_type(alias),
                )

    def test_keeps_existing_runtime_names(self):
        for model_type in VL_RUNTIME_CONFIG.QWEN_MODEL_TYPES:
            with self.subTest(model_type=model_type):
                self.assertEqual(
                    model_type,
                    VL_RUNTIME_CONFIG.normalize_model_type(model_type),
                )


if __name__ == "__main__":
    unittest.main()
