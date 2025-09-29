# Copyright (c) 2026 Segno System.

add_executable(dashinfer_c_mini mini_inference.c)
target_include_directories(dashinfer_c_mini PRIVATE
                           ${PROJECT_SOURCE_DIR}/csrc/interface)
target_link_libraries(dashinfer_c_mini PRIVATE allspark_framework)
set_target_properties(dashinfer_c_mini PROPERTIES
                      C_STANDARD 11
                      C_STANDARD_REQUIRED ON)
