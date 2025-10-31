# Copyright (c) 2026 Segno System.
"""Contract tests for the dense-only DeepSeek-V3 FP8 fixture."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    from safetensors import safe_open
    import torch
except ImportError:
    safe_open = None
    torch = None


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    torch is not None and safe_open is not None
    and hasattr(torch, "float8_e4m3fn"),
    "PyTorch FP8 and safetensors are required",
)
class TinyDeepSeekFP8Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = _load_module(
            "create_tiny_dsv3_fp8", ROOT / "create_tiny_dsv3_fp8.py")
        cls.fp8_config = _load_module(
            "fp8_config",
            ROOT / "python" / "pyhie" / "allspark" / "fp8_config.py")
        cls.fp8_weights = _load_module(
            "fp8_weights",
            ROOT / "python" / "pyhie" / "allspark" / "model"
            / "fp8_weights.py")

    def test_fixture_preserves_native_dense_linears(self):
        with tempfile.TemporaryDirectory() as directory:
            self.generator.create_tiny_dsv3_fp8(
                directory, num_layers=1, seed=20251031)

            detected = self.fp8_config.detect_native_fp8(directory)
            self.assertTrue(detected["enabled"])
            self.assertEqual("blockwise", detected["format"])
            self.assertEqual((128, 128), detected["block_size"])

            shard = Path(directory) / "model-00001-of-00001.safetensors"
            with safe_open(shard, framework="pt", device="cpu") as f:
                names = set(f.keys())
                fp8_names = {
                    name for name in names
                    if name.endswith(".weight")
                    and f.get_tensor(name).dtype == torch.float8_e4m3fn
                }
                self.assertEqual(4, len(fp8_names))
                self.assertFalse(any(".mlp.experts." in name for name in names))
                state_dict = {
                    name: f.get_tensor(name)
                    for name in names
                    if name.startswith("model.layers.0.mlp.gate_proj.")
                    or name.startswith("model.layers.0.mlp.up_proj.")
                }

            merged_shape = self.fp8_weights.merge_fp8_linears(
                state_dict,
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.gate_up_proj.weight")
            self.assertEqual((1024, 256), merged_shape)
            self.assertEqual(
                torch.float8_e4m3fn,
                state_dict["model.layers.0.mlp.gate_up_proj.weight"].dtype)


if __name__ == "__main__":
    unittest.main()
