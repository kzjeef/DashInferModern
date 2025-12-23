# Copyright (c) 2026 Segno System.

set(protobuf_MODULE_COMPATIBLE ON)
find_package(Protobuf CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

add_library(CONAN_PKG::protobuf INTERFACE IMPORTED GLOBAL)
set_property(TARGET CONAN_PKG::protobuf PROPERTY
             INTERFACE_LINK_LIBRARIES protobuf::libprotobuf)

add_library(CONAN_PKG::glog INTERFACE IMPORTED GLOBAL)
set_property(TARGET CONAN_PKG::glog PROPERTY
             INTERFACE_LINK_LIBRARIES glog::glog)

add_library(CONAN_PKG::zlib INTERFACE IMPORTED GLOBAL)
set_property(TARGET CONAN_PKG::zlib PROPERTY
             INTERFACE_LINK_LIBRARIES ZLIB::ZLIB)

if(BUILD_UTEST)
  find_package(GTest CONFIG REQUIRED)
  add_library(CONAN_PKG::gtest INTERFACE IMPORTED GLOBAL)
  set_property(TARGET CONAN_PKG::gtest PROPERTY
               INTERFACE_LINK_LIBRARIES GTest::gtest)
endif()
