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


class _FakeAsModelConfig:

    def __init__(self):
        self.lora_max_rank = 0
        self.lora_max_num = 0


class _FakeAsCacheMode(enum.Enum):
    AsCacheDefault = 0
    AsCacheQuantI8 = 1
    AsCacheQuantU4 = 2


class _FakeTargetDevice(enum.Enum):
    CUDA = 0
    CPU = 1
    CPU_NUMA = 2


def _load_runtime_config():
    package_name = "_allspark_config_regression"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ALLSPARK_ROOT)]

    native = types.ModuleType(f"{package_name}._allspark")
    native.AsModelConfig = _FakeAsModelConfig
    native.AsCacheMode = _FakeAsCacheMode

    engine = types.ModuleType(f"{package_name}.engine")
    engine.TargetDevice = _FakeTargetDevice

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.runtime_config",
        ALLSPARK_ROOT / "runtime_config.py",
    )
    module = importlib.util.module_from_spec(spec)
    stubs = {
        package_name: package,
        f"{package_name}._allspark": native,
        f"{package_name}.engine": engine,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


RUNTIME_CONFIG = _load_runtime_config()


class GenerationConfigBuilderRegressionTest(unittest.TestCase):

    def test_empty_eos_token_list_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            GENERATION_CONFIG.ASGenerationConfigBuilder.process_eos_tokens(
                [], {})

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


class RuntimeConfigBuilderRegressionTest(unittest.TestCase):

    def test_prefill_length_is_read_from_mapping(self):
        builder = RUNTIME_CONFIG.AsModelRuntimeConfigBuilder()
        builder.update_from_dict({"engine_max_prefill_length": "256"})

        self.assertEqual(256, builder.new_runtime_cfg.engine_max_prefill_length)


if __name__ == "__main__":
    unittest.main()
