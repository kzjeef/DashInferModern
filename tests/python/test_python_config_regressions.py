# Copyright (c) 2026 Segno System.
"""Regression coverage for the Python configuration builders."""

import enum
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
ALLSPARK_ROOT = REPO_ROOT / "python" / "pyhie" / "allspark"


def _load_generation_config():
    transformers = types.ModuleType("transformers")
    transformers.GenerationConfig = type("GenerationConfig", (), {})
    spec = importlib.util.spec_from_file_location(
        "generation_config_regression",
        ALLSPARK_ROOT / "generation_config.py",
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"transformers": transformers}):
        spec.loader.exec_module(module)
    return module


GENERATION_CONFIG = _load_generation_config()


class GenerationConfigBuilderRegressionTest(unittest.TestCase):

    def test_default_seed_is_generated_for_each_builder(self):
        with mock.patch.object(
                GENERATION_CONFIG.random, "randint", side_effect=[101, 202]):
            first = GENERATION_CONFIG.ASGenerationConfigBuilder().build()
            second = GENERATION_CONFIG.ASGenerationConfigBuilder().build()

        self.assertEqual(101, first["seed"])
        self.assertEqual(202, second["seed"])

    def test_multimedia_info_is_stored_by_key(self):
        info = object()
        config = (
            GENERATION_CONFIG.ASGenerationConfigBuilder()
            .with_mm_info(info)
            .build()
        )

        self.assertIs(info, config["mm_info"])


if __name__ == "__main__":
    unittest.main()
