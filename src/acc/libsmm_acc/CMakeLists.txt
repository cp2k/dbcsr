
set(CUSMM_KERNELS
  kernels/cusmm_common.h
  kernels/cusmm_dnt_largeDB1.h
  kernels/cusmm_dnt_largeDB2.h
  kernels/cusmm_dnt_medium.h
  kernels/cusmm_dnt_small.h
  kernels/cusmm_dnt_tiny.h
  kernels/cusmm_transpose.h
)

set(LIBSMM_SRC_FILES
  libsmm.cpp
  libsmm_benchmark.cpp
)

set(LIBSMM_HEADER_FILES
  libsmm.h
  libsmm_benchmark.h
  parameters.h
  parameters_utils.h
  cusmm_kernels.h
)

set(LIBSMM_FILES
  ${LIBSMM_SRC_FILES}
  ${LIBSMM_HEADER_FILES}
)

add_custom_command(
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_parameters.py --gpu_version=${WITH_GPU} --base_dir=${CMAKE_CURRENT_SOURCE_DIR}/parameters
  DEPENDS generate_parameters.py parameters/parameters_${WITH_GPU}.json
  OUTPUT parameters.h
  COMMENT "libsmm: generating parameters for GPU ${WITH_GPU}"
)

add_custom_command(
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_kernels.py ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  DEPENDS generate_kernels.py ${CUSMM_KERNELS}
  OUTPUT cusmm_kernels.h
  COMMENT "libsmm: generating kernels"
)

if (USE_CUDA)

  add_library(libsmm OBJECT ${LIBSMM_FILES})
  target_compile_definitions(libsmm PRIVATE __CUDA)
  target_compile_definitions(libsmm PRIVATE ARCH_NUMBER=${ACC_ARCH_NUMBER})

else () # i.e. USE_HIP

  set_source_files_properties(${LIBSMM_SRC_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  hip_add_library(libsmm STATIC ${LIBSMM_FILES})  # hip_add_library does not support OBJECT libraries

  # Workaround issue in hip_add_library: explicitely write dependency between __ & __
  add_custom_target(generate_cusmm_kernels_h DEPENDS cusmm_kernels.h)
  add_dependencies(libsmm generate_cusmm_kernels_h)
  add_custom_target(generate_parameters_h DEPENDS parameters.h)
  add_dependencies(libsmm generate_parameters_h)

  install(TARGETS libsmm
    EXPORT libsmmTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
  target_compile_definitions(libsmm PRIVATE __HIP)
  target_compile_definitions(libsmm PRIVATE ARCH_NUMBER=${ACC_ARCH_NUMBER})

endif()

if (OpenMP_FOUND)
  # with CMake 3.12+ the following can be replaced by:
  #   target_link_libraries(libcusmm PRIVATE OpenMP::OpenMP_CXX)
  target_compile_options(libsmm PRIVATE
      $<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_OPTIONS>
    )
endif ()

target_include_directories(libsmm PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

target_include_directories(libsmm SYSTEM PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

