# Copyright (c) 2026 Segno System.
"""Contracts and an optional native smoke test for the macOS Qwen mini path."""

import ast
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "python" / "0_basic" / "macos" / "qwen_mini.py"
PYTHON_ROOT = REPO_ROOT / "python"
ALLSPARK_ROOT = PYTHON_ROOT / "pyhie" / "allspark"


class MacOSQwenMiniTest(unittest.TestCase):

    def test_example_is_qwen_only_and_q8_block_aligned(self):
        source = EXAMPLE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        constants = {}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                    isinstance(node.targets[0], ast.Name) and
                    isinstance(node.value, ast.Constant)):
                constants[node.targets[0].id] = node.value.value

        self.assertEqual(0, constants["HIDDEN_SIZE"] % 32)
        self.assertEqual(0, constants["INTERMEDIATE_SIZE"] % 32)
        self.assertIn('model_type="Qwen_v20"', source)
        self.assertIn("use_ggml_q8_0=True", source)
        for unsupported in ("LLaMA", "DeepSeek", "ChatGLM", "Baichuan"):
            self.assertNotIn(unsupported, source)

    def test_public_serializer_propagates_q8_opt_in(self):
        paths = [
            ALLSPARK_ROOT / "engine.py",
            ALLSPARK_ROOT / "engine_utils.py",
            ALLSPARK_ROOT / "model_loader.py",
        ]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertIn("use_ggml_q8_0=False", source, path.name)
            self.assertIn("use_ggml_q8_0=use_ggml_q8_0", source, path.name)

    def test_apple_binary_fallback_covers_qwen_ops(self):
        source = (
            REPO_ROOT / "csrc" / "core" / "operator" / "general" /
            "binary" / "binary_op.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("defined(__APPLE__)", source)
        for binary_type in ("ADD", "MUL", "SWIGLU", "GEGLU"):
            self.assertIn(f"case BinaryType::{binary_type}", source)

    def test_runtime_thread_count_accepts_auto_but_rejects_negative(self):
        source = EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("if threads < 0:", source)
        self.assertIn("use 0 for auto", source)

    @unittest.skipUnless(
        sys.platform == "darwin" and platform.machine() == "arm64",
        "requires Apple Silicon",
    )
    def test_native_prefill_decode(self):
        extension = list(ALLSPARK_ROOT.glob("_allspark*.so"))
        if not extension:
            self.skipTest("build the macos-arm Python extension first")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PYTHON_ROOT)
        completed = subprocess.run(
            [
                sys.executable,
                str(EXAMPLE),
                "--max-length", "8",
                "--threads", "2",
                "--seed", "2025",
            ],
            check=True,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertIn(
            "PASS: Qwen mini GGML Q8_0 prefill/decode", completed.stdout)


if __name__ == "__main__":
    unittest.main()
