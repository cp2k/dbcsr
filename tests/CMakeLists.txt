
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
  target_link_options(dbcsr_perf PRIVATE ${HIP_ARCH_FLAGS})
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

if ("${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI v2.1" OR "${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI v3.1")
    list(FILTER DBCSR_PERF_TESTS EXCLUDE REGEX "_rma")
endif ()

foreach (dbcsr_perf_test ${DBCSR_PERF_TESTS})
  if (USE_MPI)
    separate_arguments(MPIEXEC_PREFLAGS)
    add_test(NAME dbcsr_perf:${dbcsr_perf_test}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_ranks} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dbcsr_perf> ${MPIEXEC_POSTFLAGS} "${CMAKE_CURRENT_SOURCE_DIR}/${dbcsr_perf_test}")
  else ()
    add_test(NAME dbcsr_perf:${dbcsr_perf_test}
             COMMAND $<TARGET_FILE:dbcsr_perf> "${CMAKE_CURRENT_SOURCE_DIR}/${dbcsr_perf_test}")
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
  dbcsr_tas_unittest
  dbcsr_test_csr_conversions
  )

if (NOT (CMAKE_Fortran_COMPILER_ID STREQUAL "Cray"))
  set(DBCSR_TESTS_CPP
     dbcsr_tensor_test_cpp
     )
endif ()

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
set(dbcsr_tas_unittest_SRCS dbcsr_tas_unittest.F)
set(dbcsr_test_csr_conversions_SRCS dbcsr_test_csr_conversions.F)
set(dbcsr_tensor_test_cpp_SRCS dbcsr_tensor_test.cpp)

# instead of building a full-blown lib, it would be better to simply build an OBJECT lib,
# but we would need cmake 3.12 to be able to specify target_link_libraries on those to get
# the proper compile flags
add_library(dbcsr_unittest_common STATIC ${dbcsr_unittest_common_SRCS})
target_link_libraries(dbcsr_unittest_common PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
if (OpenMP_FOUND)
  target_link_libraries(dbcsr_unittest_common PUBLIC OpenMP::OpenMP_Fortran)
endif ()

if (APPLE AND BLAS_LIBRARIES MATCHES "Accelerate")
  target_compile_definitions(dbcsr_unittest_common PRIVATE __ACCELERATE)
endif()
target_link_libraries(dbcsr_unittest_common PUBLIC dbcsr)

foreach (dbcsr_test ${DBCSR_TESTS})
  if (USE_HIP)
    hip_add_executable(${dbcsr_test} ${${dbcsr_test}_SRCS})
    target_link_options(${dbcsr_test} PRIVATE ${HIP_ARCH_FLAGS})
  else ()
    add_executable(${dbcsr_test} ${${dbcsr_test}_SRCS})
  endif ()
  target_link_libraries(${dbcsr_test} dbcsr_unittest_common)
  set_target_properties(${dbcsr_test} PROPERTIES LINKER_LANGUAGE Fortran)
  # register unittest executable with CMake
  if (USE_MPI)
    separate_arguments(MPIEXEC_PREFLAGS)
    add_test(NAME ${dbcsr_test}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_ranks} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${dbcsr_test}> ${MPIEXEC_POSTFLAGS})
  else ()
      add_test(NAME ${dbcsr_test} COMMAND ${dbcsr_test})
  endif ()
  if (OpenMP_FOUND)
    target_link_libraries(${dbcsr_test} OpenMP::OpenMP_Fortran)
    set_tests_properties(${dbcsr_test} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=${TEST_OMP_THREADS})
  endif ()
endforeach ()

# set the __SHORT_FILE__ per file for dbcsr sources
foreach (tests_src ${DBCSR_PERF_SRCS} ${dbcsr_unittest_common_SRCS} ${dbcsr_unittest1_SRCS} ${dbcsr_unittest2_SRCS} ${dbcsr_unittest3_SRCS} ${dbcsr_tensor_unittest_SRCS} ${dbcsr_tas_unittest_SRCS} ${dbcsr_test_csr_conversions_SRCS})
  # add_fypp_sources returns a path in the current binary dir
  get_filename_component(short_file "${tests_src}" NAME)
  set_source_files_properties(${tests_src} PROPERTIES COMPILE_DEFINITIONS __SHORT_FILE__="${short_file}")
endforeach ()

if (WITH_C_API)
  foreach (dbcsr_test_cpp ${DBCSR_TESTS_CPP})
    add_executable(${dbcsr_test_cpp} ${${dbcsr_test_cpp}_SRCS})
    target_link_libraries(${dbcsr_test_cpp} dbcsr_c MPI::MPI_CXX)
    # register unittest executable with CMake
    if (USE_MPI)
      separate_arguments(MPIEXEC_PREFLAGS)
      add_test(NAME ${dbcsr_test_cpp} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_ranks} ${MPIEXEC_PREFLAGS} ./${dbcsr_test_cpp} ${MPIEXEC_POSTFLAGS})
    else ()
      add_test(NAME ${dbcsr_test_cpp} COMMAND ./${dbcsr_test_cpp})
    endif ()
    if (OpenMP_FOUND)
      set_tests_properties(${dbcsr_test_cpp} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=${TEST_OMP_THREADS})
    endif ()
  endforeach ()
endif ()

# =================================== GPU BACKEND TESTS (CUDA / HIP)
if (USE_CUDA OR USE_HIP)

  # All libsmm_acc tests
  set(LIBSMM_ACC_TESTS_BUILD
    libsmm_acc_unittest_multiply
    libsmm_acc_unittest_transpose
    libsmm_acc_timer_multiply
    )

  # Tests that need no additional arguments to be run
  set(LIBSMM_ACC_SIMPLE_TESTS
    libsmm_acc_unittest_multiply
    libsmm_acc_unittest_transpose
    )

  # Add custom commands for the test files that need to be generated from a template
  # libsmm_acc_unittest_multiply, libsmm_acc_timer_multiply
  file(RELATIVE_PATH CURRENT_BINARY_DIR_RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR})

  add_custom_command(
    OUTPUT libsmm_acc_unittest_multiply.cpp
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_libsmm_acc_unittest_multiply.py --base_dir ${CMAKE_CURRENT_SOURCE_DIR}/.. --out_dir ${CURRENT_BINARY_DIR_RELATIVE} --gpu_version=${WITH_GPU}
    DEPENDS libsmm_acc_unittest_multiply.cpp.template generate_libsmm_acc_unittest_multiply.py
    COMMENT "Generate tests/libsmm_acc_unittest_multiply.cpp"
  )

  add_custom_command(
    OUTPUT libsmm_acc_timer_multiply.cpp
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_libsmm_acc_timer_multiply.py --base_dir ${CMAKE_CURRENT_SOURCE_DIR}/.. --out_dir ${CURRENT_BINARY_DIR_RELATIVE} --gpu_version=${WITH_GPU}
    DEPENDS libsmm_acc_timer_multiply.cpp.template generate_libsmm_acc_timer_multiply.py
    COMMENT "Generate tests/libsmm_acc_unittest_transpose.cpp"
  )

  # Add executables for all libsmm_acc tests
  if (USE_CUDA)

    foreach (libsmm_acc_test ${LIBSMM_ACC_TESTS_BUILD})

      add_executable(${libsmm_acc_test} ${libsmm_acc_test}.cpp)
      target_compile_definitions(${libsmm_acc_test} PRIVATE __CUDA)
      target_include_directories(${libsmm_acc_test} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

      target_link_libraries(${libsmm_acc_test} dbcsr)

      if (OpenMP_FOUND)
        target_link_libraries(${libsmm_acc_test} OpenMP::OpenMP_CXX)
      endif ()

    endforeach ()

  else()  # i.e. USE_HIP

    foreach (libsmm_acc_test ${LIBSMM_ACC_TESTS_BUILD})
      set_source_files_properties(${libsmm_acc_test}.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    endforeach ()

    hip_add_executable(libsmm_acc_unittest_multiply ${CMAKE_CURRENT_BINARY_DIR}/libsmm_acc_unittest_multiply.cpp)
    target_link_options(libsmm_acc_unittest_multiply PRIVATE ${HIP_ARCH_FLAGS})
    hip_add_executable(libsmm_acc_timer_multiply ${CMAKE_CURRENT_BINARY_DIR}/libsmm_acc_timer_multiply.cpp)
    target_link_options(libsmm_acc_timer_multiply PRIVATE ${HIP_ARCH_FLAGS})
    hip_add_executable(libsmm_acc_unittest_transpose libsmm_acc_unittest_transpose.cpp)
    target_link_options(libsmm_acc_unittest_transpose PRIVATE ${HIP_ARCH_FLAGS})

    # Workaround issue in hip_add_library: explicitely write dependency between __ & __
    add_custom_target(generate_libsmm_acc_unittest_multiply_test_cpp DEPENDS libsmm_acc_unittest_multiply.cpp)
    add_dependencies(libsmm_acc_unittest_multiply generate_libsmm_acc_unittest_multiply_test_cpp)
    add_custom_target(generate_libsmm_acc_timer_multiply_test_cpp DEPENDS libsmm_acc_timer_multiply.cpp)
    add_dependencies(libsmm_acc_timer_multiply generate_libsmm_acc_timer_multiply_test_cpp)

    foreach (libsmm_acc_test ${LIBSMM_ACC_TESTS_BUILD})

      target_link_libraries(${libsmm_acc_test} dbcsr)
      target_compile_definitions(${libsmm_acc_test} PRIVATE __HIP)

      if (OpenMP_FOUND)
        target_link_libraries(${libsmm_acc_test} OpenMP::OpenMP_CXX)
        target_link_libraries(${libsmm_acc_test} /opt/rh/llvm-toolset-7/root/usr/lib64/libomp.so)
      endif ()

    endforeach ()

  endif()

  # Add tests that do not need additional arguments
  foreach (libsmm_acc_test ${LIBSMM_ACC_SIMPLE_TESTS})
      add_test(NAME ${libsmm_acc_test} COMMAND ${libsmm_acc_test})
  endforeach ()

  # Add tests needing additional arguments:
  add_test(NAME libsmm_acc_timer_multiply-autotuned COMMAND libsmm_acc_timer_multiply autotuned)
  add_test(NAME libsmm_acc_timer_multiply-predicted COMMAND libsmm_acc_timer_multiply predicted)

endif ()
