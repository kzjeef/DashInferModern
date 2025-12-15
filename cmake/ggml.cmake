# Copyright (c) 2026 Segno System.

include(FetchContent)

# Last upstream synchronization available before the initial DashInfer GGML
# integration. Keep this immutable so Q8_0 kernels do not change underneath us.
set(GGML_GIT_REVISION
    "3e9f2ba3b934c20b26873b3c60dbf41b116978ff"
    CACHE STRING "Pinned ggml revision")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static dependencies" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "Build ggml tests" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Build ggml examples" FORCE)
set(GGML_METAL OFF CACHE BOOL "Use the ggml Metal backend" FORCE)
set(GGML_BLAS OFF CACHE BOOL "Use the ggml BLAS backend" FORCE)
set(GGML_OPENMP ON CACHE BOOL "Use OpenMP in ggml CPU kernels" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Build portable ggml CPU kernels" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "Enable ggml warnings" FORCE)

FetchContent_Declare(
  ggml
  GIT_REPOSITORY https://github.com/ggml-org/ggml.git
  GIT_TAG ${GGML_GIT_REVISION}
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(ggml)

set(GGML_LIBRARY ggml::ggml)
