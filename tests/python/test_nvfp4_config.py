# Copyright (c) 2026 Segno System.
"""Unit tests for the retired ModelOpt NVFP4 checkpoint guard."""

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
    / "nvfp4_config.py"
)
SPEC = importlib.util.spec_from_file_location("nvfp4_config", MODULE_PATH)
NVFP4_CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NVFP4_CONFIG)


class ModelOptNVFP4DetectionTest(unittest.TestCase):

    def _write_json(self, directory, filename, content):
        path = Path(directory) / filename
        path.write_text(json.dumps(content), encoding="utf-8")
        return path

    def test_missing_directory_is_not_quantized(self):
        result = NVFP4_CONFIG.detect_modelopt_nvfp4("/missing/nvfp4/model")
        self.assertEqual({
            "enabled": False,
            "block_size": 16,
            "source": None,
        }, result)

    def test_reads_modelopt_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_json(directory, "hf_quant_config.json", {
                "producer": {"name": "modelopt"},
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
            })

            result = NVFP4_CONFIG.detect_modelopt_nvfp4(directory)

        self.assertEqual({
            "enabled": True,
            "block_size": 16,
            "source": "hf_quant_config.json",
        }, result)

    def test_reads_embedded_quantization_config(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_json(directory, "config.json", {
                "quantization_config": {
                    "quant_method": "nvfp4",
                    "group_size": 32,
                },
            })

            result = NVFP4_CONFIG.detect_modelopt_nvfp4(directory)

        self.assertTrue(result["enabled"])
        self.assertEqual(32, result["block_size"])
        self.assertEqual("config.json", result["source"])

    def test_rejects_invalid_group_size(self):
        for group_size in (0, -1, "not-an-integer"):
            with self.subTest(group_size=group_size):
                with tempfile.TemporaryDirectory() as directory:
                    self._write_json(directory, "hf_quant_config.json", {
                        "quantization": {
                            "quant_algo": "NVFP4",
                            "group_size": group_size,
                        },
                    })
                    with self.assertRaises(ValueError):
                        NVFP4_CONFIG.detect_modelopt_nvfp4(directory)

    def test_ignores_other_quantization_algorithms(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_json(directory, "hf_quant_config.json", {
                "quantization": {"quant_algo": "FP8"},
            })

            result = NVFP4_CONFIG.detect_modelopt_nvfp4(directory)

        self.assertFalse(result["enabled"])

    def test_rejects_nvfp4_before_weight_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_json(directory, "hf_quant_config.json", {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
            })
            with self.assertRaisesRegex(ValueError, "SM100 backend"):
                NVFP4_CONFIG.reject_unsupported_modelopt_nvfp4(directory)


if __name__ == "__main__":
    unittest.main()
