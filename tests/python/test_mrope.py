# Copyright (c) 2026 Segno System.
"""Unit tests for dependency-free multimodal rotary positions."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest


UTILS_PATH = (
    Path(__file__).resolve().parents[2]
    / "multimodal"
    / "dashinfer_vlm"
    / "vl_inference"
    / "utils"
)
PACKAGE_NAME = "dashinfer_vl_test_utils"
PACKAGE = ModuleType(PACKAGE_NAME)
PACKAGE.__path__ = [str(UTILS_PATH)]
sys.modules.setdefault(PACKAGE_NAME, PACKAGE)
SPEC = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.mrope", UTILS_PATH / "mrope.py")
MROPE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MROPE
SPEC.loader.exec_module(MROPE)


class MRoPEPositionTest(unittest.TestCase):

    def test_text_only_positions_are_linear(self):
        positions = MROPE.build_mrope_positions([10, 11, 12], [], 99)
        self.assertEqual([[0, 1, 2]] * 3, positions)

    def test_places_one_vision_span_on_three_axes(self):
        positions = MROPE.build_mrope_positions(
            [10, 99, 99, 11], [(1, 2, 4)], 99)
        self.assertEqual(
            [
                [0, 1, 1, 3],
                [0, 1, 1, 3],
                [0, 1, 2, 3],
            ],
            positions,
        )

    def test_advances_after_multiple_vision_spans(self):
        positions = MROPE.build_mrope_positions(
            [1, 99, 2, 3, 99, 4],
            [(1, 2, 2), (1, 2, 2)],
            99,
        )
        self.assertEqual([[0, 1, 2, 3, 4, 5]] * 3, positions)

    def test_requires_one_contiguous_span_per_grid(self):
        with self.assertRaisesRegex(ValueError, "expects 2 contiguous"):
            MROPE.build_mrope_positions(
                [10, 99, 11, 99], [(1, 2, 4)], 99)

    def test_rejects_image_tokens_without_grid(self):
        with self.assertRaisesRegex(ValueError, "without a vision grid"):
            MROPE.build_mrope_positions([10, 99, 11], [], 99)

    def test_requires_a_span_for_every_grid(self):
        with self.assertRaisesRegex(ValueError, "grid 1"):
            MROPE.build_mrope_positions(
                [10, 99, 11], [(1, 2, 2), (1, 2, 2)], 99)


if __name__ == "__main__":
    unittest.main()
