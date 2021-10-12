find_package(OpenMP REQUIRED)

add_library (cpucudart
  cpucuda_runtime.cpp
  )
target_compile_options(cpucudart PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
target_compile_options(cpucudart PRIVATE $<$<COMPILE_LANGUAGE:C>:${OpenMP_CXX_FLAGS}>)
target_link_libraries(cpucudart PRIVATE OpenMP::OpenMP_CXX)
