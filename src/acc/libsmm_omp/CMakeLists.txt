
set(LIBSMM_OMP_SRCS
  libsmm_omp.c
  libsmm_omp_transpose.c
  libsmm_omp_process.c
)

add_library(libsmm_omp OBJECT ${LIBSMM_OMP_SRCS})

if (OpenMP_FOUND)
  # with CMake 3.12+ the following can be replaced by:
  #   target_link_libraries(libsmm_omp PRIVATE OpenMP::OpenMP_C)
  target_compile_options(libsmm_omp PRIVATE
      $<TARGET_PROPERTY:OpenMP::OpenMP_C,INTERFACE_COMPILE_OPTIONS>
    )
endif ()
