# Copyright (c) 2026 Segno System.
# Build the native ModelOpt NVFP4 GEMM in an isolated CUTLASS 4 target.

set(NVFP4_GEMM_ENABLED FALSE)
set(_NVFP4_BUILD_ARCHS "")
foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "-(real|virtual)$" "" _arch_base "${_arch}")
  if(_arch_base STREQUAL "100" OR _arch_base STREQUAL "100a")
    list(APPEND _NVFP4_BUILD_ARCHS "100a")
  endif()
endforeach()
list(REMOVE_DUPLICATES _NVFP4_BUILD_ARCHS)

if(NOT _NVFP4_BUILD_ARCHS)
  message(STATUS "Native NVFP4 GEMM disabled: no SM100 target")
  return()
endif()

if(NOT TARGET project_cutlass4)
  message(FATAL_ERROR "Native NVFP4 GEMM requires the CUTLASS 4 dependency")
endif()

set(_NVFP4_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/csrc/core/kernel/cuda/gemm_lowp")
add_library(nvfp4_gemm STATIC
    "${_NVFP4_SOURCE_DIR}/gemm_nvfp4_blockwise_sm100.cu")
add_dependencies(nvfp4_gemm project_cutlass4)

target_include_directories(nvfp4_gemm PRIVATE
    "${CUTLASS4_SOURCE_DIR}/include"
    "${CUTLASS4_SOURCE_DIR}/tools/util/include"
    "${PROJECT_SOURCE_DIR}/csrc"
    "${PROJECT_SOURCE_DIR}/csrc/core/kernel"
    "${_NVFP4_SOURCE_DIR}")
target_link_libraries(nvfp4_gemm PRIVATE CUDA::cuda_driver CUDA::cudart)
target_compile_options(nvfp4_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr
      --extended-lambda
      -Xcompiler=-Wno-psabi
      -Xcompiler=-Wno-unused-parameter>)
set_target_properties(nvfp4_gemm PROPERTIES
    CUDA_ARCHITECTURES "${_NVFP4_BUILD_ARCHS}"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

set(NVFP4_GEMM_ENABLED TRUE)
message(STATUS
        "Native NVFP4 GEMM enabled for ${_NVFP4_BUILD_ARCHS} with CUTLASS 4")
