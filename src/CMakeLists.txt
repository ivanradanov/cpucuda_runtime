find_package(OpenMP REQUIRED)

add_library (cpucudart
  cpucuda_runtime.cpp
  cpucuda_libdevice.c
  )

set_property(TARGET cpucudart PROPERTY CXX_STANDARD 14)

target_compile_options(cpucudart PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
target_compile_options(cpucudart PRIVATE -x cuda --cuda-host-only --cuda-path=${CUDA_PATH})

target_link_libraries(cpucudart PRIVATE OpenMP::OpenMP_CXX)
