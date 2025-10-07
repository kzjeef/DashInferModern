# Copyright (c) 2026 Segno System.
"""Unit tests for native FP8 checkpoint metadata detection."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "pyhie"
    / "allspark"
    / "fp8_config.py"
)
SPEC = importlib.util.spec_from_file_location("fp8_config", MODULE_PATH)
FP8_CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FP8_CONFIG)


class NativeFP8DetectionTest(unittest.TestCase):

    def _write_json(self, directory, filename, content):
        path = Path(directory) / filename
        path.write_text(json.dumps(content), encoding="utf-8")

    def test_missing_directory_is_not_quantized(self):
        self.assertEqual({
            "enabled": False,
            "format": None,
            "block_size": (128, 128),
            "source": None,
        }, FP8_CONFIG.detect_native_fp8("/missing/fp8/model"))

    def test_reads_blockwise_config(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_json(directory, "config.json", {
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [64, 128],
                },
            })
            result = FP8_CONFIG.detect_native_fp8(directory)

        self.assertTrue(result["enabled"])
        self.assertEqual("blockwise", result["format"])
        self.assertEqual((64, 128), result["block_size"])

    def test_reads_nested_text_config(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_json(directory, "config.json", {
                "text_config": {
                    "quantization_config": {"quant_method": "fp8"},
                },
            })
            result = FP8_CONFIG.detect_native_fp8(directory)

        self.assertEqual("blockwise", result["format"])
        self.assertEqual((128, 128), result["block_size"])

    def test_reads_per_tensor_modelopt_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_json(directory, "hf_quant_config.json", {
                "quantization": {"quant_algo": "FP8"},
            })
            result = FP8_CONFIG.detect_native_fp8(directory)

        self.assertTrue(result["enabled"])
        self.assertEqual("per_tensor", result["format"])
        self.assertEqual("hf_quant_config.json", result["source"])

    def test_falls_back_to_scale_inv_index(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_json(directory, "model.safetensors.index.json", {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "part-1",
                    "model.layers.0.self_attn.q_proj.weight_scale_inv":
                    "part-1",
                },
            })
            result = FP8_CONFIG.detect_native_fp8(directory)

        self.assertTrue(result["enabled"])
        self.assertEqual("blockwise", result["format"])

    def test_rejects_invalid_block_size(self):
        for block_size in ([128], [128, 0], ["x", 128]):
            with self.subTest(block_size=block_size):
                with tempfile.TemporaryDirectory() as directory:
                    self._write_json(directory, "config.json", {
                        "quantization_config": {
                            "quant_method": "fp8",
                            "weight_block_size": block_size,
                        },
                    })
                    with self.assertRaises(ValueError):
                        FP8_CONFIG.detect_native_fp8(directory)


if __name__ == "__main__":
    unittest.main()
