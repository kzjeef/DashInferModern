# Copyright (c) 2026 Segno System.
"""Unit tests for dependency-free vision grid validation."""

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "multimodal"
    / "dashinfer_vlm"
    / "vl_inference"
    / "utils"
    / "vision_grid.py"
)
SPEC = importlib.util.spec_from_file_location("vision_grid", MODULE_PATH)
VISION_GRID = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VISION_GRID)


class VisionGridTest(unittest.TestCase):

    def test_normalizes_valid_grid(self):
        self.assertEqual((1, 58, 42),
                         VISION_GRID.normalize_vision_grid([1, 58, 42]))

    def test_counts_merged_vision_tokens(self):
        self.assertEqual(609,
                         VISION_GRID.vision_token_count((1, 58, 42)))
        self.assertEqual(8,
                         VISION_GRID.vision_token_count((2, 4, 4)))

    def test_rejects_invalid_shape(self):
        for grid in (None, (), (1, 2), (1, 2, 3, 4)):
            with self.subTest(grid=grid):
                with self.assertRaisesRegex(ValueError, "grid_thw"):
                    VISION_GRID.normalize_vision_grid(grid)

    def test_rejects_non_positive_or_non_integer_values(self):
        for grid in ((0, 2, 2), (1, -2, 2), (1, 2.0, 2), (True, 2, 2)):
            with self.subTest(grid=grid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    VISION_GRID.normalize_vision_grid(grid)

    def test_rejects_unmergeable_spatial_grid(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            VISION_GRID.normalize_vision_grid((1, 3, 4))


if __name__ == "__main__":
    unittest.main()
