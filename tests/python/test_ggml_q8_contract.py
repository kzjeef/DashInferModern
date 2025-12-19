# Copyright (c) 2026 Segno System.
"""Contracts for the opt-in Qwen-only GGML Q8_0 path."""

import json
from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "python" / "pyhie" / "allspark" / "model"


class GGMLQ8ContractTest(unittest.TestCase):

    def test_macos_preset_enables_cpu_only_ggml(self):
        presets = json.loads(
            (REPO_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        macos = next(
            item for item in presets["configurePresets"]
            if item["name"] == "macos-arm")
        cache = macos["cacheVariables"]
        self.assertEqual("ON", cache["ENABLE_GGML"])
        self.assertEqual("OFF", cache["ENABLE_CUDA"])
        self.assertEqual("OFF", cache["ENABLE_NVFP4"])
        self.assertEqual("OFF", cache["ENABLE_FP8"])

    def test_ggml_revision_is_immutable_and_cpu_only(self):
        module = (REPO_ROOT / "cmake" / "ggml.cmake").read_text(
            encoding="utf-8")
        revision = re.search(r'"([0-9a-f]{40})"', module)
        self.assertIsNotNone(revision)
        self.assertNotIn("master", module)
        self.assertIn("set(GGML_METAL OFF", module)
        self.assertIn("set(GGML_BLAS OFF", module)

    def test_only_dense_qwen_serializes_q8_marker(self):
        qwen_sources = [
            (MODEL_ROOT / "qwen_v15.py").read_text(encoding="utf-8"),
            (MODEL_ROOT / "qwen_v20.py").read_text(encoding="utf-8"),
        ]
        for source in qwen_sources:
            self.assertIn("use_ggml_q8_0", source)

        for path in MODEL_ROOT.glob("*.py"):
            if path.name in {"qwen_v15.py", "qwen_v20.py"}:
                continue
            self.assertNotIn(
                "use_ggml_q8_0", path.read_text(encoding="utf-8"), path.name)

    def test_runtime_has_no_fp4_or_moe_quantized_branch(self):
        source = (
            REPO_ROOT / "csrc" / "core" / "operator" / "general" / "gemm"
            / "gemm_op_cpu.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("GGML_TYPE_Q8_0", source)
        self.assertNotIn("GGML_TYPE_MXFP4", source)
        self.assertNotIn("GGML_TYPE_Q4_", source)
        self.assertIn('attr_map.find("use_ggml_q8_0")', source)


if __name__ == "__main__":
    unittest.main()
