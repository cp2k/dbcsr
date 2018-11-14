
set(CUSMM_KERNELS
  kernels/cusmm_common.h
  kernels/cusmm_dnt_largeDB1.h
  kernels/cusmm_dnt_largeDB2.h
  kernels/cusmm_dnt_medium.h
  kernels/cusmm_dnt_small.h
  kernels/cusmm_dnt_tiny.h
  kernels/cusmm_transpose.h
)

add_custom_command(
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_parameters.py --gpu_version=${WITH_GPU} --base_dir=${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS generate_parameters.py parameters_${WITH_GPU}.json
  OUTPUT parameters.h
  COMMENT "libcusmm: generating parameters for GPU ${WITH_GPU}"
)

add_custom_command(
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_kernels.py ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  DEPENDS generate_kernels.py ${CUSMM_KERNELS}
  OUTPUT cusmm_kernels.h
  COMMENT "libcusmm: generating kernels"
)

add_library(libcusmm OBJECT
  libcusmm.cpp
  libcusmm_benchmark.cu
  parameters.h
  cusmm_kernels.h
  )

# use legacy directives here due to missing target_include_directories support
# for object libraries with CMake < 3.12
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_definitions(-DARCH_NUMBER=${CUDA_ARCH_NUMBER})
