# Copyright (c) 2026 Segno System.
"""Contract tests for the dense-only DeepSeek-V3 NVFP4 fixture."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_module(
    "create_tiny_dsv3_nvfp4", ROOT / "create_tiny_dsv3_nvfp4.py")
NVFP4_CONFIG = _load_module(
    "nvfp4_config",
    ROOT / "python" / "pyhie" / "allspark" / "nvfp4_config.py")
NVFP4_WEIGHTS = _load_module(
    "nvfp4_weights",
    ROOT / "python" / "pyhie" / "allspark" / "model"
    / "nvfp4_weights.py")


class TinyDeepSeekNVFP4Test(unittest.TestCase):

    def test_fixture_preserves_packed_dense_linears(self):
        with tempfile.TemporaryDirectory() as directory:
            GENERATOR.create_tiny_dsv3_nvfp4(
                directory, num_layers=1, seed=20250401)

            detected = NVFP4_CONFIG.detect_modelopt_nvfp4(directory)
            self.assertTrue(detected["enabled"])
            self.assertEqual(16, detected["block_size"])

            shard = Path(directory) / "model-00001-of-00001.safetensors"
            with safe_open(shard, framework="pt", device="cpu") as f:
                names = set(f.keys())
                packed_names = {
                    name for name in names
                    if name.endswith(".weight")
                    and f.get_tensor(name).dtype == torch.uint8
                }
                self.assertEqual(4, len(packed_names))
                self.assertFalse(any(".mlp.experts." in name for name in names))

                state_dict = {
                    name: f.get_tensor(name)
                    for name in names
                    if name.startswith("model.layers.0.mlp.gate_proj.")
                    or name.startswith("model.layers.0.mlp.up_proj.")
                }

            merged_shape = NVFP4_WEIGHTS.merge_nvfp4_linears(
                state_dict,
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.gate_up_proj.weight")
            self.assertEqual((1024, 256), merged_shape)
            self.assertEqual(
                torch.uint8,
                state_dict["model.layers.0.mlp.gate_up_proj.weight"].dtype)


if __name__ == "__main__":
    unittest.main()
