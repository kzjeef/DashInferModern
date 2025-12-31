# Copyright (c) 2026 Segno System.
"""Contracts for the CUDA support ceiling after retiring SM100."""

import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class SM100RetirementTest(unittest.TestCase):

    def test_default_cuda_targets_stop_at_sm90a(self):
        build_script = (REPO_ROOT / "build.sh").read_text(encoding="utf-8")
        self.assertIn('AS_CUDA_SM:-80;86;90a', build_script)
        self.assertNotIn("100a", build_script)

        presets = json.loads(
            (REPO_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        for preset in presets["configurePresets"]:
            cache = preset.get("cacheVariables", {})
            self.assertNotIn("ENABLE_NVFP4", cache)

    def test_sm100_build_and_runtime_sources_are_absent(self):
        removed_paths = (
            "cmake/cutlass4.cmake",
            "cmake/nvfp4-gemm.cmake",
            "csrc/core/kernel/cuda/gemm_lowp/"
            "gemm_nvfp4_blockwise_sm100.cu",
            "csrc/core/operator/general/gemm_lowp/"
            "gemm_nvfp4_blockwise_gpu.cpp",
        )
        for relative_path in removed_paths:
            self.assertFalse((REPO_ROOT / relative_path).exists(), relative_path)

        cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotIn("ENABLE_NVFP4", cmake)
        self.assertNotIn("ENABLE_SM100_NVFP4", cmake)

    def test_no_high_precision_nvfp4_fallback_was_added(self):
        forbidden = (
            "_dequant_nvfp4",
            "dequant_nvfp4_to_bf16",
            "dequant_nvfp4_to_fp16",
            "dequant_nvfp4_to_fp32",
        )
        roots = (
            REPO_ROOT / "python" / "pyhie" / "allspark",
            REPO_ROOT / "csrc",
        )
        for root in roots:
            for path in root.rglob("*"):
                if (not path.is_file() or
                        path.suffix not in {".py", ".cpp", ".cu", ".h", ".cuh"}):
                    continue
                source = path.read_text(encoding="utf-8", errors="ignore")
                for symbol in forbidden:
                    self.assertNotIn(symbol, source, str(path))


if __name__ == "__main__":
    unittest.main()
