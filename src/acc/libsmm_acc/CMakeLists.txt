
set(SMM_ACC_KERNELS
  kernels/smm_acc_common.h
  kernels/smm_acc_dnt_largeDB1.h
  kernels/smm_acc_dnt_largeDB2.h
  kernels/smm_acc_dnt_medium.h
  kernels/smm_acc_dnt_small.h
  kernels/smm_acc_dnt_tiny.h
  kernels/smm_acc_transpose.h
)

set(LIBSMM_ACC_SRC_FILES
  libsmm_acc.cpp
  libsmm_acc_benchmark.cpp
  libsmm_acc_init.cpp
)

set(LIBSMM_ACC_HEADER_FILES
  libsmm_acc.h
  libsmm_acc_benchmark.h
  libsmm_acc_init.h
  parameters.h
  parameters_utils.h
  smm_acc_kernels.h
)

set(LIBSMM_ACC_FILES
  ${LIBSMM_ACC_SRC_FILES}
  ${LIBSMM_ACC_HEADER_FILES}
)

add_custom_command(
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_parameters.py --gpu_version=${WITH_GPU} --base_dir=${CMAKE_CURRENT_SOURCE_DIR}/parameters
  DEPENDS generate_parameters.py parameters/parameters_${WITH_GPU}.json
  OUTPUT parameters.h
  COMMENT "libsmm_acc: generating parameters for GPU ${WITH_GPU}"
)

add_custom_command(
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_kernels.py ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  DEPENDS generate_kernels.py ${SMM_ACC_KERNELS}
  OUTPUT smm_acc_kernels.h
  COMMENT "libsmm_acc: generating kernels"
)

if (USE_CUDA)

  add_library(libsmm_acc OBJECT ${LIBSMM_ACC_FILES})
  target_compile_definitions(libsmm_acc PRIVATE __CUDA)
  target_compile_definitions(libsmm_acc PRIVATE ARCH_NUMBER=${ACC_ARCH_NUMBER})

else () # i.e. USE_HIP

  set_source_files_properties(${LIBSMM_ACC_SRC_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  # hip_add_library does not support OBJECT libraries. For this reason, we are forced to use a static library
  hip_add_library(libsmm_acc STATIC ${LIBSMM_ACC_FILES})

  # Workaround issue in hip_add_library: explicitely write dependency between
  # the generation of header files and the compilation of the libsmm_acc files
  add_custom_target(generate_smm_acc_kernels_h DEPENDS smm_acc_kernels.h)
  add_dependencies(libsmm_acc generate_smm_acc_kernels_h)
  add_custom_target(generate_parameters_h DEPENDS parameters.h)
  add_dependencies(libsmm_acc generate_parameters_h)

  install(TARGETS libsmm_acc
    EXPORT libsmm_accTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
  target_compile_definitions(libsmm_acc PRIVATE __HIP)
  target_compile_definitions(libsmm_acc PRIVATE ARCH_NUMBER=${ACC_ARCH_NUMBER})

endif()

if (OpenMP_FOUND)
  # with CMake 3.12+ the following can be replaced by:
  #   target_link_libraries(libcusmm PRIVATE OpenMP::OpenMP_CXX)
  target_compile_options(libsmm_acc PRIVATE
      $<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_OPTIONS>
    )
endif ()

target_include_directories(libsmm_acc PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

target_include_directories(libsmm_acc SYSTEM PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

