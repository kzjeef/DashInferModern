# Copyright (c) 2026 Segno System.
"""Portable checks for the Apple Silicon build surface."""

import json
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class MacOSBuildConfigTest(unittest.TestCase):

    def test_build_jobs_follow_apple_logical_cpu_count(self):
        build_script = (REPO_ROOT / "build.sh").read_text(encoding="utf-8")
        self.assertIn('build_jobs="${AS_BUILD_JOBS:-}"', build_script)
        self.assertIn('sysctl -n hw.logicalcpu', build_script)
        self.assertIn('--parallel "${build_jobs}"', build_script)

    def test_native_build_checks_required_tools(self):
        build_script = (REPO_ROOT / "build.sh").read_text(encoding="utf-8")
        self.assertIn("for tool in brew cmake ninja", build_script)
        self.assertIn('command -v "${tool}"', build_script)
        self.assertIn("Missing macOS build tool", build_script)

    def test_deployment_target_is_explicit_and_overridable(self):
        presets = json.loads(
            (REPO_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        macos = next(
            item for item in presets["configurePresets"]
            if item["name"] == "macos-arm")
        self.assertEqual(
            "13.0", macos["cacheVariables"]["CMAKE_OSX_DEPLOYMENT_TARGET"])

        build_script = (REPO_ROOT / "build.sh").read_text(encoding="utf-8")
        self.assertIn("AS_MACOS_DEPLOYMENT_TARGET:-13.0", build_script)
        self.assertIn("-DCMAKE_OSX_DEPLOYMENT_TARGET=", build_script)

    def test_preset_is_cpu_only_apple_silicon(self):
        presets = json.loads(
            (REPO_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        macos = next(
            item for item in presets["configurePresets"]
            if item["name"] == "macos-arm")
        cache = macos["cacheVariables"]
        self.assertEqual("ARM", cache["CONFIG_HOST_CPU_TYPE"])
        self.assertEqual("OFF", cache["ENABLE_CUDA"])
        self.assertEqual("OFF", cache["ENABLE_AVX2"])
        self.assertEqual("ACCELERATE", cache["ALLSPARK_CBLAS"])

    def test_conan_profile_targets_apple_clang(self):
        profile = (
            REPO_ROOT / "conan" / "conanprofile.macos_arm64"
        ).read_text(encoding="utf-8")
        self.assertIn("os=Macos", profile)
        self.assertIn("arch=armv8", profile)
        self.assertIn("compiler=apple-clang", profile)
        self.assertIn("compiler.libcxx=libc++", profile)

    def test_native_build_uses_an_isolated_directory(self):
        build_script = (REPO_ROOT / "build.sh").read_text(encoding="utf-8")
        self.assertIn('build_folder="build/macos-arm"', build_script)
        self.assertIn('build_folder="${AS_BUILD_FOLDER}"', build_script)
        self.assertIn('mkdir -p "${build_folder}"', build_script)

    def test_native_build_selects_the_active_macos_sdk(self):
        build_script = (REPO_ROOT / "build.sh").read_text(encoding="utf-8")
        self.assertIn("command -v xcrun", build_script)
        self.assertIn("xcrun --sdk macosx --show-sdk-path", build_script)
        self.assertIn('-DCMAKE_OSX_SYSROOT="${macos_sdk_root}"', build_script)

    def test_native_build_honors_the_selected_python(self):
        build_script = (REPO_ROOT / "build.sh").read_text(encoding="utf-8")
        self.assertIn('python_executable="${AS_PYTHON_EXECUTABLE:-}"',
                      build_script)
        self.assertIn("command -v python3", build_script)
        self.assertIn('-DPYTHON_EXECUTABLE="${python_executable}"',
                      build_script)

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS SDK")
    def test_accelerate_cblas_header_compiles(self):
        source = r"""
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
int main() {
  float value = 1.0f;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              1, 1, 1, 1.0f, &value, 1, &value, 1, 0.0f, &value, 1);
  return 0;
}
"""
        subprocess.run(
            ["clang++", "-std=c++17", "-x", "c++", "-fsyntax-only", "-"],
            input=source,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS SDK")
    def test_cpp_ipc_posix_sources_compile(self):
        ipc_root = REPO_ROOT / "third_party" / "from_source" / "cpp-ipc"
        sources = [
            "src/libipc/platform/platform.cpp",
            "src/libipc/sync/condition.cpp",
            "src/libipc/sync/mutex.cpp",
            "src/libipc/sync/semaphore.cpp",
            "src/libipc/sync/waiter.cpp",
        ]
        subprocess.run(
            [
                "clang++",
                "-std=c++17",
                "-fsyntax-only",
                f"-I{ipc_root / 'include'}",
                f"-I{ipc_root / 'src'}",
                *[str(ipc_root / source) for source in sources],
            ],
            check=True,
            cwd=REPO_ROOT,
        )


if __name__ == "__main__":
    unittest.main()
