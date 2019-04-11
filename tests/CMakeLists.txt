
if (USE_MPI)
  if (TEST_MPI_RANKS STREQUAL "auto")
    include(ProcessorCount)
    ProcessorCount(nproc)
    math(EXPR num_ranks "(${nproc}+${TEST_OMP_THREADS}-1)/${TEST_OMP_THREADS}")  # get 1/$TEST_OMP_THREADS the number of procs (rounded up)
  else ()
    set(num_ranks ${TEST_MPI_RANKS})
  endif ()
  message("Tests will run with ${num_ranks} MPI ranks and ${TEST_OMP_THREADS} OpenMP threads each")
endif ()

set(DBCSR_PERF_SRCS
  dbcsr_performance_driver.F
  dbcsr_performance_multiply.F)
add_executable(dbcsr_perf ${DBCSR_PERF_SRCS})
target_link_libraries(dbcsr_perf dbcsr)

file(GLOB DBCSR_PERF_TESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "inputs/*.perf"
    )

foreach (dbcsr_perf_test ${DBCSR_PERF_TESTS})
  if (USE_MPI)
    add_test(NAME dbcsr_perf:${dbcsr_perf_test} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_ranks} ${MPIEXEC_PREFLAGS} ./dbcsr_perf ${MPIEXEC_POSTFLAGS} "${CMAKE_CURRENT_SOURCE_DIR}/${dbcsr_perf_test}")
  else ()
    add_test(NAME dbcsr_perf:${dbcsr_perf_test} COMMAND ./dbcsr_perf "${CMAKE_CURRENT_SOURCE_DIR}/${dbcsr_perf_test}")
  endif ()
  set_tests_properties(dbcsr_perf:${dbcsr_perf_test} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=${TEST_OMP_THREADS})
endforeach ()

# Define all the tests here, will be used as the executable name
set(DBCSR_TESTS
  dbcsr_unittest1
  dbcsr_unittest2
  dbcsr_unittest3
  dbcsr_tensor_unittest
  dbcsr_test_csr_conversions
  )

# Common object files linked to all tests
set(dbcsr_unittest_common_SRCS
  dbcsr_test_add.F
  dbcsr_test_multiply.F
  )

# For each test, set a variable testname_SRCS defining the sources of that test
set(dbcsr_unittest1_SRCS dbcsr_unittest1.F)
set(dbcsr_unittest2_SRCS dbcsr_unittest2.F)
set(dbcsr_unittest3_SRCS dbcsr_unittest3.F)
set(dbcsr_tensor_unittest_SRCS dbcsr_tensor_unittest.F)
set(dbcsr_test_csr_conversions_SRCS dbcsr_test_csr_conversions.F)

# instead of building a full-blown lib, it would be better to simply build an OBJECT lib,
# but we would need cmake 3.12 to be able to specify target_link_libraries on those to get
# the proper compile flags
add_library(dbcsr_unittest_common STATIC ${dbcsr_unittest_common_SRCS})
if (APPLE AND BLAS_LIBRARIES MATCHES "Accelerate")
  target_compile_definitions(dbcsr_unittest_common PRIVATE __ACCELERATE)
endif()
target_link_libraries(dbcsr_unittest_common dbcsr)

foreach (dbcsr_test ${DBCSR_TESTS})
  add_executable(${dbcsr_test} ${${dbcsr_test}_SRCS})
  target_link_libraries(${dbcsr_test} dbcsr_unittest_common)
  # register unittest executable with CMake
  if (USE_MPI)
    add_test(NAME ${dbcsr_test} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_ranks} ${MPIEXEC_PREFLAGS} ./${dbcsr_test} ${MPIEXEC_POSTFLAGS})
  else ()
    add_test(NAME ${dbcsr_test} COMMAND ./${dbcsr_test})
  endif ()
  set_tests_properties(${dbcsr_test} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=${num_threads})
endforeach ()

if (USE_CUDA)
  set(LIBCUSMM_SIMPLE_TESTS
    libcusmm_unittest_multiply
    libcusmm_unittest_transpose
    )

  set(LIBCUSMM_TESTS_BUILD
    ${LIBCUSMM_SIMPLE_TESTS}
    libcusmm_timer_multiply
    )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/libcusmm_unittest_multiply.cu
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_libcusmm_unittest_multiply.py --base_folder ${CMAKE_CURRENT_SOURCE_DIR}/.. --gpu_version=${WITH_GPU}
    DEPENDS libcusmm_unittest_multiply.template generate_libcusmm_unittest_multiply.py
    COMMENT "Generate tests/libcusmm_unittest_multiply.cu"
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/libcusmm_timer_multiply.cu
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_libcusmm_timer_multiply.py --base_folder ${CMAKE_CURRENT_SOURCE_DIR}/.. --gpu_version=${WITH_GPU}
    DEPENDS libcusmm_timer_multiply.template generate_libcusmm_timer_multiply.py
    COMMENT "Generate tests/libcusmm_timer_multiply.cu"
  )

  foreach (libcusmm_test ${LIBCUSMM_TESTS_BUILD})
    add_executable(${libcusmm_test} ${CMAKE_CURRENT_SOURCE_DIR}/${libcusmm_test}.cu)
    target_link_libraries(${libcusmm_test} dbcsr)
  endforeach ()

  foreach (libcusmm_test ${LIBCUSMM_SIMPLE_TESTS})
    add_test(NAME ${libcusmm_test} COMMAND ./${libcusmm_test})
  endforeach ()

  # tests needing additional arguments:
  add_test(NAME libcusmm_timer_multiply-autotuned COMMAND ./libcusmm_timer_multiply autotuned)
  add_test(NAME libcusmm_timer_multiply-predicted COMMAND ./libcusmm_timer_multiply predicted)
endif ()
