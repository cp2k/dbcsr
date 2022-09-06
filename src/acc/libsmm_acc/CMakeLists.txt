set(SMM_ACC_KERNELS
    kernels/smm_acc_common.h
    kernels/smm_acc_dnt_largeDB1.h
    kernels/smm_acc_dnt_largeDB2.h
    kernels/smm_acc_dnt_medium.h
    kernels/smm_acc_dnt_small.h
    kernels/smm_acc_dnt_tiny.h
    kernels/smm_acc_transpose.h)

add_custom_target(
  parameters ALL
  COMMAND
    ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_parameters.py
    --gpu_version=${WITH_GPU} --base_dir=${CMAKE_CURRENT_SOURCE_DIR}/parameters
  DEPENDS generate_parameters.py parameters/parameters_${WITH_GPU_PARAMS}.json
  BYPRODUCTS parameters.h
  COMMENT "libsmm_acc: generating parameters for GPU ${WITH_GPU_PARAMS}")

add_custom_target(
  smm_acc_kernels ALL
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_kernels.py
          ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  DEPENDS generate_kernels.py ${SMM_ACC_KERNELS}
  BYPRODUCTS smm_acc_kernels.h
  COMMENT "libsmm_acc: generating kernels")

add_dependencies(dbcsr smm_acc_kernels parameters)
target_include_directories(dbcsr PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                         ${CMAKE_CURRENT_SOURCE_DIR})

# Note: this library is only used in some of the tests, it's just to get include
# paths to generated header files.
add_library(libsmm_acc INTERFACE)
target_include_directories(libsmm_acc INTERFACE ${CMAKE_CURRENT_BINARY_DIR}
                                                ${CMAKE_CURRENT_SOURCE_DIR})
