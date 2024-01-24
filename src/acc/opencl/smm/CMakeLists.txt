set(LIBSMM_ACC_HEADER_KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/opencl_kernels.h)

set(SMM_ACC_KERNEL_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/../acc_opencl.sh)
set(SMM_ACC_COMMON ../common/opencl_atomics.h ../common/opencl_common.h)
set(SMM_ACC_KERNELS kernels/multiply.cl kernels/transpose.cl)
list(TRANSFORM SMM_ACC_KERNELS PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)

set(SMM_ACC_PARAMS_WITHGPU
    ${CMAKE_CURRENT_SOURCE_DIR}/params/tune_multiply_${WITH_GPU}.csv)
set(SMM_ACC_PARAMS_DEFAULT ${CMAKE_CURRENT_SOURCE_DIR}/tune_multiply.csv)

if (EXISTS ${SMM_ACC_PARAMS_WITHGPU})
  set(SMM_ACC_PARAMS ${SMM_ACC_PARAMS_WITHGPU})
elseif (EXISTS ${SMM_ACC_PARAMS_DEFAULT})
  set(SMM_ACC_PARAMS ${SMM_ACC_PARAMS_DEFAULT})
endif ()

add_custom_target(
  parameters ALL
  COMMAND ${SMM_ACC_KERNEL_SCRIPT} ${SMM_ACC_KERNELS} ${SMM_ACC_PARAMS}
          ${LIBSMM_ACC_HEADER_KERNELS}
  DEPENDS ${SMM_ACC_KERNEL_SCRIPT} ${SMM_ACC_KERNELS} ${SMM_ACC_COMMON}
  BYPRODUCTS ${LIBSMM_ACC_HEADER_KERNELS}
  COMMENT "ACC/LIBSMM OpenCL: collecting tuned kernel parameters...")

add_dependencies(dbcsr parameters)
target_include_directories(dbcsr PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                         ${CMAKE_CURRENT_SOURCE_DIR})
