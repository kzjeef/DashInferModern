# Copyright (c) 2026 Segno System.
"""Mini contract test for Qwen2.5-VL request preparation."""

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
PACKAGE_NAME = "dashinfer_vl_mini_utils"
PACKAGE = ModuleType(PACKAGE_NAME)
PACKAGE.__path__ = [str(UTILS_PATH)]
sys.modules.setdefault(PACKAGE_NAME, PACKAGE)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODEL_CONFIG = load_module(
    f"{PACKAGE_NAME}.vl_model_config", UTILS_PATH / "vl_model_config.py")
RUNTIME_CONFIG = load_module(
    f"{PACKAGE_NAME}.runtime_config", UTILS_PATH / "config" / "config.py")
TOKENS = load_module(
    f"{PACKAGE_NAME}.multimodal_tokens",
    UTILS_PATH / "multimodal_tokens.py",
)
MROPE = load_module(f"{PACKAGE_NAME}.mrope", UTILS_PATH / "mrope.py")


class MiniVLPipelineTest(unittest.TestCase):

    def test_qwen25_image_prompt_contract(self):
        model_spec = MODEL_CONFIG.detect_vl_model({
            "architectures": ["Qwen2_5_VLForConditionalGeneration"]
        })
        self.assertEqual(
            "QWEN2-VL",
            RUNTIME_CONFIG.normalize_model_type(model_spec.runtime_name),
        )

        grid = (1, 2, 4)
        vision_length = MROPE.vision_token_count(grid)
        input_tokens = TOKENS.prepare_multimodal_tokens(
            [7, 151652, 151653, 8],
            [vision_length],
            151652,
            151653,
            151859,
        )
        positions = MROPE.build_mrope_positions(
            input_tokens, [grid], 151859)

        self.assertEqual([7, 151652, 151859, 151859, 151653, 8],
                         input_tokens)
        self.assertEqual([0, 1, 2, 2, 4, 5], positions[0])
        self.assertEqual([0, 1, 2, 3, 4, 5], positions[2])
        self.assertTrue(all(len(axis) == len(input_tokens)
                            for axis in positions))


if __name__ == "__main__":
    unittest.main()
