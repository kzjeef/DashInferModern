# Copyright (c) 2026 Segno System.
# Provide pinned CUTLASS 4 headers for native SM100 NVFP4 kernels.

set(NVFP4_CUTLASS4_ENABLED FALSE)
set(_NVFP4_HAS_SM100 FALSE)
foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "-(real|virtual)$" "" _arch_base "${_arch}")
  if(_arch_base STREQUAL "100" OR _arch_base STREQUAL "100a")
    set(_NVFP4_HAS_SM100 TRUE)
  endif()
endforeach()

if(NOT _NVFP4_HAS_SM100)
  message(STATUS "CUTLASS 4 NVFP4 headers disabled: no SM100 target")
  return()
endif()

if(CUDA_VERSION VERSION_LESS "12.8")
  message(FATAL_ERROR "Native NVFP4 requires CUDA 12.8 or newer")
endif()

include(ExternalProject)

set(CUTLASS4_VERSION "4.3.5")
set(CUTLASS4_SHA256
    "73d8c3914a6049ff5c43b7dfb9d70f26e44dc9e10e36049db5a999b9faf6dbf0")
set(CUTLASS4_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/cutlass4")
set(CUTLASS4_SOURCE_DIR "${CUTLASS4_PREFIX}/src/project_cutlass4")
set(CUTLASS4_LOCAL_ARCHIVE
    "${PROJECT_SOURCE_DIR}/third_party/cutlass_${CUTLASS4_VERSION}.tgz")

if(EXISTS "${CUTLASS4_LOCAL_ARCHIVE}")
  set(CUTLASS4_URL "file://${CUTLASS4_LOCAL_ARCHIVE}")
else()
  set(CUTLASS4_URL
      "https://github.com/NVIDIA/cutlass/archive/refs/tags/v${CUTLASS4_VERSION}.tar.gz")
endif()

ExternalProject_Add(
  project_cutlass4
  PREFIX "${CUTLASS4_PREFIX}"
  URL "${CUTLASS4_URL}"
  URL_HASH "SHA256=${CUTLASS4_SHA256}"
  SOURCE_DIR "${CUTLASS4_SOURCE_DIR}"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  UPDATE_COMMAND ""
)

set(NVFP4_CUTLASS4_ENABLED TRUE)
message(STATUS
        "Native NVFP4 will use CUTLASS ${CUTLASS4_VERSION} from ${CUTLASS4_SOURCE_DIR}")
