# Copyright (c) 2026 Segno System.
"""Unit tests for multimodal prompt-token preparation."""

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "multimodal"
    / "dashinfer_vlm"
    / "vl_inference"
    / "utils"
    / "multimodal_tokens.py"
)
SPEC = importlib.util.spec_from_file_location(
    "multimodal_tokens", MODULE_PATH)
MULTIMODAL_TOKENS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MULTIMODAL_TOKENS)

BOS = 10
EOS = 11
TARGET = 12


class MultimodalTokensTest(unittest.TestCase):

    def test_expands_empty_marker(self):
        tokens = MULTIMODAL_TOKENS.prepare_multimodal_tokens(
            [1, BOS, EOS, 2], [3], BOS, EOS, TARGET)
        self.assertEqual([1, BOS, TARGET, TARGET, TARGET, EOS, 2], tokens)

    def test_expands_multiple_markers_in_order(self):
        tokens = MULTIMODAL_TOKENS.prepare_multimodal_tokens(
            [BOS, EOS, 1, BOS, EOS], [1, 2], BOS, EOS, TARGET)
        self.assertEqual(
            [BOS, TARGET, EOS, 1, BOS, TARGET, TARGET, EOS],
            tokens,
        )

    def test_accepts_matching_existing_spans(self):
        source = [1, TARGET, TARGET, 2, TARGET, 3]
        tokens = MULTIMODAL_TOKENS.prepare_multimodal_tokens(
            source, [2, 1], BOS, EOS, TARGET)
        self.assertEqual(source, tokens)

    def test_preserves_text_only_prompt(self):
        tokens = MULTIMODAL_TOKENS.prepare_multimodal_tokens(
            [1, 2, 3], [], BOS, EOS, TARGET)
        self.assertEqual([1, 2, 3], tokens)

    def test_rejects_marker_result_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "markers"):
            MULTIMODAL_TOKENS.prepare_multimodal_tokens(
                [BOS, EOS], [], BOS, EOS, TARGET)

    def test_rejects_existing_span_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "expected 3"):
            MULTIMODAL_TOKENS.prepare_multimodal_tokens(
                [TARGET, TARGET], [3], BOS, EOS, TARGET)

    def test_rejects_invalid_embedding_length(self):
        for length in (0, -1, 1.5, True):
            with self.subTest(length=length):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    MULTIMODAL_TOKENS.prepare_multimodal_tokens(
                        [BOS, EOS], [length], BOS, EOS, TARGET)


if __name__ == "__main__":
    unittest.main()
