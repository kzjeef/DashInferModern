# Copyright (c) 2026 Segno System.
"""Unit tests for native block-wise FP8 weight helpers."""

import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "pyhie"
    / "allspark"
    / "model"
    / "fp8_weights.py"
)
if torch is not None:
    SPEC = importlib.util.spec_from_file_location("fp8_weights", MODULE_PATH)
    FP8_WEIGHTS = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(FP8_WEIGHTS)
else:
    FP8_WEIGHTS = None


@unittest.skipUnless(torch is not None and hasattr(torch, "float8_e4m3fn"),
                     "PyTorch does not expose FP8 E4M3")
class NativeFP8WeightsTest(unittest.TestCase):

    def _linear(self, prefix, n_dim=130, k_dim=129):
        return {
            prefix + ".weight": torch.zeros(
                (n_dim, k_dim), dtype=torch.float8_e4m3fn),
            prefix + ".weight_scale_inv": torch.ones(
                ((n_dim + 127) // 128, (k_dim + 127) // 128),
                dtype=torch.float32),
        }

    def test_validates_partial_edge_blocks(self):
        state_dict = self._linear("proj")
        shape = FP8_WEIGHTS.validate_fp8_linear(
            state_dict, "proj.weight", (128, 128))
        self.assertEqual((130, 129), tuple(shape))

    def test_rejects_missing_or_malformed_scale(self):
        state_dict = self._linear("proj")
        del state_dict["proj.weight_scale_inv"]
        with self.assertRaises(KeyError):
            FP8_WEIGHTS.validate_fp8_linear(state_dict, "proj.weight")

        state_dict = self._linear("proj")
        state_dict["proj.weight_scale_inv"] = torch.ones(
            (1, 1), dtype=torch.float32)
        with self.assertRaises(ValueError):
            FP8_WEIGHTS.validate_fp8_linear(state_dict, "proj.weight")

    def test_merges_codes_and_scales_without_dtype_change(self):
        state_dict = {}
        state_dict.update(self._linear("gate", n_dim=128, k_dim=128))
        state_dict.update(self._linear("up", n_dim=128, k_dim=128))

        shape = FP8_WEIGHTS.merge_fp8_linears(
            state_dict, "gate.weight", "up.weight", "gate_up.weight")

        self.assertEqual((256, 128), tuple(shape))
        self.assertEqual(torch.float8_e4m3fn,
                         state_dict["gate_up.weight"].dtype)
        self.assertEqual(torch.float32,
                         state_dict["gate_up.weight_scale_inv"].dtype)
        self.assertEqual((2, 1),
                         tuple(state_dict["gate_up.weight_scale_inv"].shape))

    def test_rejects_incompatible_k_dimensions(self):
        state_dict = {}
        state_dict.update(self._linear("gate", n_dim=128, k_dim=128))
        state_dict.update(self._linear("up", n_dim=128, k_dim=256))
        with self.assertRaises(ValueError):
            FP8_WEIGHTS.merge_fp8_linears(
                state_dict, "gate.weight", "up.weight", "gate_up.weight")


if __name__ == "__main__":
    unittest.main()
