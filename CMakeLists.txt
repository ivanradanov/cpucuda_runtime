
cmake_minimum_required (VERSION 3.11)

# TODO need to require clang >= 12, cuda includes don't seem to be working with
# clang 11
set(CMAKE_C_COMPILER_NAMES clang)
set(CMAKE_CXX_COMPILER_NAMES clang++)
project (cpucuda_runtime)

add_subdirectory(src)
