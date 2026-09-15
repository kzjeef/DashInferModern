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

    def test_kv_cache_span_size_is_validated(self):
        builder = RUNTIME_CONFIG.AsModelRuntimeConfigBuilder()
        builder.kv_cache_span_size("64")
        self.assertEqual(64, builder.new_runtime_cfg.cache_span_size)

        with self.assertRaisesRegex(ValueError, "one of 16"):
            builder.kv_cache_span_size(48)

    def test_empty_numa_device_list_defaults_to_node_zero(self):
        config = (
            RUNTIME_CONFIG.AsModelRuntimeConfigBuilder()
            .compute_unit(_FakeTargetDevice.CPU_NUMA, [])
            .build()
        )

        self.assertEqual("CPU:0", config.compute_unit)

    def test_invalid_target_device_type_has_clear_error(self):
        builder = RUNTIME_CONFIG.AsModelRuntimeConfigBuilder()
        with self.assertRaisesRegex(TypeError, "target_device"):
            builder.compute_unit(object())

    def test_builder_reset_preserves_lora_defaults(self):
        builder = RUNTIME_CONFIG.AsModelRuntimeConfigBuilder()
        builder.build()
        second = builder.build()

        self.assertEqual(64, second.lora_max_rank)
        self.assertEqual(5, second.lora_max_num)

    def test_false_strings_do_not_enable_runtime_features(self):
        builder = RUNTIME_CONFIG.AsModelRuntimeConfigBuilder()
        builder.update_from_dict({
            "enable_prefix_cache": "false",
            "enable_sparsity_matmul": "0",
        })

        self.assertFalse(builder.new_runtime_cfg.enable_prefix_cache)
        self.assertFalse(builder.new_runtime_cfg.enable_sparsity_matmul)

    def test_invalid_boolean_strings_are_rejected(self):
        builder = RUNTIME_CONFIG.AsModelRuntimeConfigBuilder()
        with self.assertRaisesRegex(ValueError, "invalid boolean"):
            builder.update_from_dict({"enable_prefix_cache": "sometimes"})

    def test_prefill_length_is_read_from_mapping(self):
        builder = RUNTIME_CONFIG.AsModelRuntimeConfigBuilder()
        builder.update_from_dict({"engine_max_prefill_length": "256"})

        self.assertEqual(256, builder.new_runtime_cfg.engine_max_prefill_length)


if __name__ == "__main__":
    unittest.main()
