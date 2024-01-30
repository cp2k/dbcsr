set(LIBSMM_ACC_KERNELS_INC ${CMAKE_CURRENT_SOURCE_DIR}/opencl_kernels.h)

set(SMM_ACC_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/../acc_opencl.sh)
set(SMM_ACC_COMMON ../common/opencl_atomics.h ../common/opencl_common.h)
set(SMM_ACC_KERNELS kernels/multiply.cl kernels/transpose.cl)
list(TRANSFORM SMM_ACC_KERNELS PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)

set(SMM_ACC_PARAMS_WITHGPU
    ${CMAKE_CURRENT_SOURCE_DIR}/params/tune_multiply_${WITH_GPU}.csv)
set(SMM_ACC_PARAMS_CUSTOM ${CMAKE_CURRENT_SOURCE_DIR}/tune_multiply.csv)

if (EXISTS ${SMM_ACC_PARAMS_WITHGPU})
  set(SMM_ACC_SCRIPT_MSG "ACC/LIBSMM OpenCL: using parameters for ${WITH_GPU}")
  set(SMM_ACC_PARAMS ${SMM_ACC_PARAMS_WITHGPU})
elseif (WITH_GPU MATCHES "none")
  set(SMM_ACC_SCRIPT_MSG "ACC/LIBSMM OpenCL: no tuned parameters used")
  set(SMM_ACC_SCRIPT_ARGS -p \"\")
elseif (EXISTS ${SMM_ACC_PARAMS_CUSTOM})
  set(SMM_ACC_SCRIPT_MSG "ACC/LIBSMM OpenCL: using custom parameters")
  set(SMM_ACC_PARAMS ${SMM_ACC_PARAMS_CUSTOM})
else ()
  set(SMM_ACC_SCRIPT_MSG "ACC/LIBSMM OpenCL: using all tuned parameters")
endif ()
message(STATUS ${SMM_ACC_SCRIPT_MSG})

add_custom_target(
  parameters # ALL
  COMMAND ${SMM_ACC_SCRIPT} ${SMM_ACC_SCRIPT_ARGS} ${SMM_ACC_KERNELS}
          ${SMM_ACC_PARAMS} ${LIBSMM_ACC_KERNELS_INC}
  DEPENDS ${SMM_ACC_SCRIPT} ${SMM_ACC_KERNELS} ${SMM_ACC_COMMON}
  BYPRODUCTS ${LIBSMM_ACC_KERNELS_INC}
  COMMENT ${SMM_ACC_SCRIPT_MSG})

add_dependencies(dbcsr parameters)
target_include_directories(dbcsr PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                         ${CMAKE_CURRENT_SOURCE_DIR})
