
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
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_parameters.py --gpu_version=${WITH_GPU} --base_dir=${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS generate_parameters.py parameters_${WITH_GPU}.json
  OUTPUT parameters.h
  COMMENT "libcusmm: generating parameters for GPU ${WITH_GPU}"
)

add_custom_command(
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_kernels.py ${CMAKE_CURRENT_SOURCE_DIR}/kernels
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

if (OpenMP_FOUND)
  # with CMake 3.12+ the following can be replaced by:
  #   target_link_libraries(libcusmm PRIVATE OpenMP::OpenMP_CXX)
  target_compile_options(libcusmm PRIVATE
      $<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_OPTIONS>
    )
endif ()

target_include_directories(libcusmm PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

target_include_directories(libcusmm SYSTEM PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_compile_definitions(libcusmm PRIVATE -DARCH_NUMBER=${CUDA_ARCH_NUMBER})
