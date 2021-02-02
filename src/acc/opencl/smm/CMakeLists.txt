set(LIBSMM_ACC_SRC_FILES opencl_libsmm.c)

set(LIBSMM_ACC_HEADER_KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/opencl_kernels.h)
set(LIBSMM_ACC_HEADER_FILES opencl_libsmm.h ${LIBSMM_ACC_HEADER_KERNELS})
set(LIBSMM_ACC_FILES ${LIBSMM_ACC_SRC_FILES} ${LIBSMM_ACC_HEADER_FILES})

set(SMM_ACC_KERNEL_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/../acc_opencl.sh)
set(SMM_ACC_KERNELS kernels/multiply.cl kernels/transpose.cl)
list(TRANSFORM SMM_ACC_KERNELS PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)

if (USE_ACCEL MATCHES "opencl")
  add_custom_command(
    COMMAND
      ${SMM_ACC_KERNEL_SCRIPT} ${SMM_ACC_KERNELS}
      # parameter file is optional (does not need to exist)
      ${CMAKE_CURRENT_SOURCE_DIR}/tune_multiply.csv ${LIBSMM_ACC_HEADER_KERNELS}
    # param file is missing here (manual/clean rebuild needed)
    DEPENDS ${SMM_ACC_KERNEL_SCRIPT} ${SMM_ACC_KERNELS}
    OUTPUT ${LIBSMM_ACC_HEADER_KERNELS}
    COMMENT "libsmm_acc: generating kernels")
  add_library(libsmm_acc OBJECT ${LIBSMM_ACC_FILES})
  target_compile_definitions(libsmm_acc PRIVATE __OPENCL)
  target_compile_definitions(libsmm_acc PRIVATE __LIBXSMM)
  # account for DBCSR not calling libsmm_acc_init() (DBCSR only calls acc_init)
  target_compile_definitions(libsmm_acc PRIVATE __DBCSR_ACC)
  target_include_directories(libsmm_acc PRIVATE ${LIBXSMM_INCLUDE_DIRS})
  if (OpenMP_FOUND)
    target_link_libraries(libsmm_acc PRIVATE OpenMP::OpenMP_C)
  endif ()
endif ()
