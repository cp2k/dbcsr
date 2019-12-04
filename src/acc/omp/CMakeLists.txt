
set(DBCSR_OMP_SRCS
  dbcsr_omp.c
  dbcsr_omp_event.c
  dbcsr_omp_mem.c
  dbcsr_omp_stream.c
)

add_library(dbcsr_omp OBJECT ${DBCSR_OMP_SRCS})

if (OpenMP_FOUND)
  # with CMake 3.12+ the following can be replaced by:
  #   target_link_libraries(dbcsr_omp PRIVATE OpenMP::OpenMP_C)
  target_compile_options(dbcsr_omp PRIVATE
      $<TARGET_PROPERTY:OpenMP::OpenMP_C,INTERFACE_COMPILE_OPTIONS>
    )
endif ()
