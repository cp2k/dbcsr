
# =================================== MPI
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

# =================================== DBCSR PERF TESTS
set(DBCSR_PERF_SRCS
  dbcsr_performance_driver.F
  dbcsr_performance_multiply.F)
if (USE_HIP)
  hip_add_executable(dbcsr_perf ${DBCSR_PERF_SRCS})
else ()
  add_executable(dbcsr_perf ${DBCSR_PERF_SRCS})
endif ()
target_link_libraries(dbcsr_perf dbcsr)
set_target_properties(dbcsr_perf PROPERTIES LINKER_LANGUAGE Fortran)
if (OpenMP_FOUND)
  target_link_libraries(dbcsr_perf OpenMP::OpenMP_Fortran)
endif ()

file(GLOB DBCSR_PERF_TESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "inputs/*.perf"
    )

foreach (dbcsr_perf_test ${DBCSR_PERF_TESTS})
  if (USE_MPI)
    separate_arguments(MPIEXEC_PREFLAGS)
    add_test(NAME dbcsr_perf:${dbcsr_perf_test} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_ranks} ${MPIEXEC_PREFLAGS} ./dbcsr_perf ${MPIEXEC_POSTFLAGS} "${CMAKE_CURRENT_SOURCE_DIR}/${dbcsr_perf_test}")
  else ()
    add_test(NAME dbcsr_perf:${dbcsr_perf_test} COMMAND ./dbcsr_perf "${CMAKE_CURRENT_SOURCE_DIR}/${dbcsr_perf_test}")
  endif ()
  set_tests_properties(dbcsr_perf:${dbcsr_perf_test} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=${TEST_OMP_THREADS})
endforeach ()

# =================================== DBCSR CORRECTNESS TESTS
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
if (OpenMP_FOUND)
  target_link_libraries(dbcsr_unittest_common OpenMP::OpenMP_Fortran)
endif ()

if (APPLE AND BLAS_LIBRARIES MATCHES "Accelerate")
  target_compile_definitions(dbcsr_unittest_common PRIVATE __ACCELERATE)
endif()
target_link_libraries(dbcsr_unittest_common dbcsr)

foreach (dbcsr_test ${DBCSR_TESTS})
  if (USE_HIP)
    hip_add_executable(${dbcsr_test} ${${dbcsr_test}_SRCS})
  else ()
    add_executable(${dbcsr_test} ${${dbcsr_test}_SRCS})
  endif ()
  target_link_libraries(${dbcsr_test} dbcsr_unittest_common)
  set_target_properties(${dbcsr_test} PROPERTIES LINKER_LANGUAGE Fortran)
  # register unittest executable with CMake
  if (USE_MPI)
    separate_arguments(MPIEXEC_PREFLAGS)
    add_test(NAME ${dbcsr_test} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_ranks} ${MPIEXEC_PREFLAGS} ./${dbcsr_test} ${MPIEXEC_POSTFLAGS})
  else ()
    add_test(NAME ${dbcsr_test} COMMAND ./${dbcsr_test})
  endif ()
  if (OpenMP_FOUND)
    target_link_libraries(${dbcsr_test} OpenMP::OpenMP_Fortran)
    set_tests_properties(${dbcsr_test} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=${TEST_OMP_THREADS})
  endif ()
endforeach ()


# =================================== GPU BACKEND TESTS (CUDA)
if (USE_CUDA)

  # All libcusmm tests
  set(LIBCUSMM_TESTS_BUILD
    libcusmm_unittest_multiply
    libcusmm_unittest_transpose
    libcusmm_timer_multiply
    )

  # Tests that need no additional arguments to be run
  set(LIBCUSMM_SIMPLE_TESTS
    libcusmm_unittest_multiply
    libcusmm_unittest_transpose
    )

  # Tests whose source needs to be generated from a template
  set(LIBCUSMM_TESTS_TO_GENERATE
    libcusmm_unittest_multiply
    libcusmm_timer_multiply
    )

  # Add custom commands for the test files that need to be generated from a template
  file(RELATIVE_PATH CURRENT_BINARY_DIR_RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR})
  foreach (libcusmm_test ${LIBCUSMM_TESTS_TO_GENERATE})
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${libcusmm_test}.cu
      COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_${libcusmm_test}.py --base_dir ${CMAKE_CURRENT_SOURCE_DIR}/.. --out_dir ${CURRENT_BINARY_DIR_RELATIVE} --gpu_version=${WITH_GPU}
      DEPENDS ${libcusmm_test}.template generate_${libcusmm_test}.py
      COMMENT "Generate tests/" ${libcusmm_test}".cu"
    )
    add_executable(${libcusmm_test} ${CMAKE_CURRENT_BINARY_DIR}/${libcusmm_test}.cu)
    target_link_libraries(${libcusmm_test} dbcsr)
    if (OpenMP_FOUND)
      target_link_libraries(${libcusmm_test} OpenMP::OpenMP_CXX)
    endif ()
  endforeach ()

  # Add executables for test files that do not need to be generated by a template
  add_executable(libcusmm_unittest_transpose ${CMAKE_CURRENT_SOURCE_DIR}/libcusmm_unittest_transpose.cu)
  target_link_libraries(libcusmm_unittest_transpose dbcsr)
  if (OpenMP_FOUND)
    target_link_libraries(libcusmm_unittest_transpose OpenMP::OpenMP_CXX)
  endif ()

  # Add tests that do not need additional arguments
  foreach (libcusmm_test ${LIBCUSMM_SIMPLE_TESTS})
    add_test(NAME ${libcusmm_test} COMMAND ./${libcusmm_test})
  endforeach ()

  # Add tests needing additional arguments:
  add_test(NAME libcusmm_timer_multiply-autotuned COMMAND ./libcusmm_timer_multiply autotuned)
  add_test(NAME libcusmm_timer_multiply-predicted COMMAND ./libcusmm_timer_multiply predicted)
endif ()

# =================================== GPU BACKEND TESTS (HIP)
if (USE_HIP)

  # All libhipsmm tests
  set(LIBHIPSMM_TESTS_BUILD
    libhipsmm_unittest_multiply
    libhipsmm_unittest_transpose
    libhipsmm_timer_multiply
    )
  set_source_files_properties(${LIBHIPSMM_TESTS_BUILD} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

  # Tests that need no additional arguments to be run
  set(LIBHIPSMM_SIMPLE_TESTS
    libhipsmm_unittest_multiply
    libhipsmm_unittest_transpose
    )

  # Tests whose source needs to be generated from a template
  set(LIBHIPSMM_TESTS_TO_GENERATE
    libhipsmm_unittest_multiply
    libhipsmm_timer_multiply
    )

  # Add custom commands for the test files that need to be generated from a template
  file(RELATIVE_PATH CURRENT_BINARY_DIR_RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR})
  foreach (libhipsmm_test ${LIBHIPSMM_TESTS_TO_GENERATE})
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${libhipsmm_test}.cpp
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_${libhipsmm_test}.py --base_dir ${CMAKE_CURRENT_SOURCE_DIR}/.. --out_dir ${CURRENT_BINARY_DIR_RELATIVE} --gpu_version=${WITH_GPU}
      DEPENDS ${libhipsmm_test}.template generate_${libhipsmm_test}.py
      COMMENT "Generate tests/" ${libhipsmm_test}".cpp"
    )
    hip_add_executable(${libhipsmm_test} ${CMAKE_CURRENT_BINARY_DIR}/${libhipsmm_test}.cpp)
    target_link_libraries(${libhipsmm_test} dbcsr)
    if (OpenMP_FOUND)
      target_link_libraries(${libhipsmm_test} OpenMP::OpenMP_CXX)
    endif ()
  endforeach ()

  # Add executables for test files that do not need to be generated by a template
  hip_add_executable(libhipsmm_unittest_transpose libhipsmm_unittest_transpose.cpp)
  target_link_libraries(libhipsmm_unittest_transpose dbcsr)
  if (OpenMP_FOUND)
    target_link_libraries(libhipsmm_unittest_transpose OpenMP::OpenMP_CXX)
  endif ()

  # Add tests that do not need additional arguments
  foreach (libhipsmm_test ${LIBHIPSMM_SIMPLE_TESTS})
    add_test(NAME ${libhipsmm_test} COMMAND ./${libhipsmm_test})
  endforeach ()

  # Add tests needing additional arguments:
  add_test(NAME libhipsmm_timer_multiply-autotuned COMMAND ./libhipsmm_timer_multiply autotuned)
  add_test(NAME libhipsmm_timer_multiply-predicted COMMAND ./libhipsmm_timer_multiply predicted)
endif ()
