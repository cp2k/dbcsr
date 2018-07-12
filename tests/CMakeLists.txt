
if (USE_MPI)
  include(ProcessorCount)
  ProcessorCount(nproc)
  math(EXPR half_nproc "(${nproc}+1)/2")  # get half the number of procs (rounded up)
endif ()

set(DBCSR_PERF_SRCS
  dbcsr_performance_driver.F
  dbcsr_performance_multiply.F)
add_executable(dbcsr_perf ${DBCSR_PERF_SRCS})
target_link_libraries(dbcsr_perf dbcsr)

set(DBCSR_PERF_TESTS
  test.perf
  )

foreach (dbcsr_perf_test ${DBCSR_PERF_TESTS})
  if (USE_MPI)
    add_test(NAME dbcsr_perf:${dbcsr_perf_test} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${half_nproc} ${MPIEXEC_PREFLAGS} dbcsr_perf ${MPIEXEC_POSTFLAGS} "${CMAKE_CURRENT_SOURCE_DIR}/${dbcsr_perf_test}")
    set_tests_properties(dbcsr_perf:${dbcsr_perf_test} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)
  else ()
    add_test(NAME dbcsr_perf:${dbcsr_perf_test} COMMAND dbcsr_perf "${CMAKE_CURRENT_SOURCE_DIR}/${dbcsr_perf_test}")
  endif ()
endforeach ()

# Define all the tests here, will be used as the executable name
set(DBCSR_TESTS
  dbcsr_unittest
  dbcsr_tensor_unittest
  dbcsr_test_csr_conversions
  )

# For each test, set a variable testname_SRCS defining the sources of that test
set(dbcsr_unittest_SRCS
  dbcsr_unittest.F
  dbcsr_test_add.F
  dbcsr_test_multiply.F
  )
set(dbcsr_tensor_unittest_SRCS
  dbcsr_tensor_unittest.F)
set(dbcsr_test_csr_conversions_SRCS
  dbcsr_test_csr_conversions.F)

foreach (dbcsr_test ${DBCSR_TESTS})
  add_executable(${dbcsr_test} ${${dbcsr_test}_SRCS})
  target_link_libraries(${dbcsr_test} dbcsr)
  # register unittest executable with CMake
  if (USE_MPI)
    add_test(NAME ${dbcsr_test} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${half_nproc} ${MPIEXEC_PREFLAGS} ${dbcsr_test} ${MPIEXEC_POSTFLAGS})
    set_tests_properties(${dbcsr_test} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)
  else ()
    add_test(NAME ${dbcsr_test} COMMAND ${dbcsr_test})
  endif ()
endforeach ()
