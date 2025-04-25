# Copyright (c) 2026 Segno System.
"""Unit tests for packed NVFP4 tensor contracts."""

import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "pyhie"
    / "allspark"
    / "model"
    / "nvfp4_weights.py"
)
SPEC = importlib.util.spec_from_file_location("nvfp4_weights", MODULE_PATH)
NVFP4_WEIGHTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NVFP4_WEIGHTS)


class NVFP4WeightsTest(unittest.TestCase):

    @staticmethod
    def _linear(prefix, n_dim=8, k_dim=32, scalar=1.0):
        packed = torch.arange(
            n_dim * k_dim // 2, dtype=torch.int64).to(torch.uint8)
        packed = packed.reshape(n_dim, k_dim // 2)
        return {
            prefix + ".weight": packed,
            prefix + ".weight_scale": torch.ones(
                (n_dim, k_dim // 16), dtype=torch.float8_e4m3fn),
            prefix + ".weight_scale_2": torch.tensor(
                [scalar], dtype=torch.float32),
            prefix + ".input_scale": torch.tensor(
                scalar, dtype=torch.float32),
        }

    def test_validates_modelopt_layout(self):
        state_dict = self._linear("linear")

        shape = NVFP4_WEIGHTS.validate_nvfp4_linear(
            state_dict, "linear.weight")

        self.assertEqual((8, 32), shape)

    def test_rejects_unpacked_or_mistyped_weight(self):
        for invalid in (
                torch.zeros((8, 16), dtype=torch.bfloat16),
                torch.zeros((8, 16, 1), dtype=torch.uint8)):
            with self.subTest(dtype=invalid.dtype, ndim=invalid.ndim):
                state_dict = self._linear("linear")
                state_dict["linear.weight"] = invalid
                with self.assertRaises(ValueError):
                    NVFP4_WEIGHTS.validate_nvfp4_linear(
                        state_dict, "linear.weight")

    def test_rejects_missing_or_misshaped_scale(self):
        state_dict = self._linear("linear")
        del state_dict["linear.input_scale"]
        with self.assertRaises(KeyError):
            NVFP4_WEIGHTS.validate_nvfp4_linear(
                state_dict, "linear.weight")

        state_dict = self._linear("linear")
        state_dict["linear.weight_scale"] = torch.ones(
            (8, 1), dtype=torch.float8_e4m3fn)
        with self.assertRaises(ValueError):
            NVFP4_WEIGHTS.validate_nvfp4_linear(
                state_dict, "linear.weight")

    def test_merges_packed_rows_without_dtype_expansion(self):
        state_dict = {}
        state_dict.update(self._linear("gate"))
        state_dict.update(self._linear("up"))
        expected_bytes = torch.cat((
            state_dict["gate.weight"], state_dict["up.weight"]), dim=0)

        shape = NVFP4_WEIGHTS.merge_nvfp4_linears(
            state_dict,
            "gate.weight",
            "up.weight",
            "gate_up.weight")

        self.assertEqual((16, 32), shape)
        self.assertEqual(torch.uint8, state_dict["gate_up.weight"].dtype)
        self.assertTrue(torch.equal(
            expected_bytes, state_dict["gate_up.weight"]))
        self.assertEqual(
            torch.float8_e4m3fn,
            state_dict["gate_up.weight_scale"].dtype)

    def test_refuses_incompatible_scalar_scales(self):
        for suffix in ("weight_scale_2", "input_scale"):
            with self.subTest(suffix=suffix):
                state_dict = {}
                state_dict.update(self._linear("gate"))
                state_dict.update(self._linear("up"))
                state_dict["up." + suffix] = torch.tensor(
                    [2.0], dtype=torch.float32)

                with self.assertRaises(ValueError):
                    NVFP4_WEIGHTS.merge_nvfp4_linears(
                        state_dict,
                        "gate.weight",
                        "up.weight",
                        "gate_up.weight")


if __name__ == "__main__":
    unittest.main()
