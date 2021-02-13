# =================================== MPI
if (USE_MPI)
  if (TEST_MPI_RANKS STREQUAL "auto")
    include(ProcessorCount)
    ProcessorCount(nproc)
    math(EXPR num_ranks "(${nproc}+${TEST_OMP_THREADS}-1)/${TEST_OMP_THREADS}"
    )# get 1/$TEST_OMP_THREADS the number of procs (rounded up)
  else ()
    set(num_ranks ${TEST_MPI_RANKS})
  endif ()
  message(
    "Tests will run with ${num_ranks} MPI ranks and ${TEST_OMP_THREADS} OpenMP threads each"
  )
endif ()

# =================================== DBCSR PERF TESTS
set(DBCSR_PERF_SRCS dbcsr_performance_driver.F dbcsr_performance_multiply.F)
add_executable(dbcsr_perf ${DBCSR_PERF_SRCS})
target_link_libraries(dbcsr_perf dbcsr)
set_target_properties(dbcsr_perf PROPERTIES LINKER_LANGUAGE Fortran)
target_link_libraries(dbcsr_perf
                      $<$<BOOL:${USE_OPENMP}>:OpenMP::OpenMP_Fortran>)

file(
  GLOB DBCSR_PERF_TESTS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "inputs/*.perf")

if ("${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI v2.1"
    OR "${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI v3.1")
  list(FILTER DBCSR_PERF_TESTS EXCLUDE REGEX "_rma")
endif ()

foreach (dbcsr_perf_test ${DBCSR_PERF_TESTS})
  if (USE_MPI)
    separate_arguments(MPIEXEC_PREFLAGS)
    add_test(
      NAME dbcsr_perf:${dbcsr_perf_test}
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_ranks}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dbcsr_perf> ${MPIEXEC_POSTFLAGS}
        "${CMAKE_CURRENT_SOURCE_DIR}/${dbcsr_perf_test}")
  else ()
    add_test(NAME dbcsr_perf:${dbcsr_perf_test}
             COMMAND $<TARGET_FILE:dbcsr_perf>
                     "${CMAKE_CURRENT_SOURCE_DIR}/${dbcsr_perf_test}")
  endif ()
  set_tests_properties(
    dbcsr_perf:${dbcsr_perf_test}
    PROPERTIES ENVIRONMENT OMP_NUM_THREADS=${TEST_OMP_THREADS})
endforeach ()

# =================================== DBCSR CORRECTNESS TESTS Define all the
# tests here, will be used as the executable name
set(DBCSR_TESTS_FTN
    dbcsr_unittest1
    dbcsr_unittest2
    dbcsr_unittest3
    dbcsr_unittest4
    dbcsr_tensor_unittest
    dbcsr_tas_unittest
    dbcsr_test_csr_conversions)

if (USE_ACCEL)
  set(DBCSR_TESTS_C dbcsr_acc_test_c)
endif ()

if (NOT (CMAKE_Fortran_COMPILER_ID STREQUAL "Cray"))
    set(DBCSR_TESTS_SRCS_CPP
       dbcsr_test.cpp
       dbcsr_tensor_test.cpp
    )
endif ()

# For each test, set a variable testname_SRCS defining the sources of that test
set(dbcsr_unittest1_SRCS dbcsr_unittest1.F)
set(dbcsr_unittest2_SRCS dbcsr_unittest2.F)
set(dbcsr_unittest3_SRCS dbcsr_unittest3.F)
set(dbcsr_unittest4_SRCS dbcsr_unittest4.F dbcsr_test_scale_by_vector.F)
set(dbcsr_tensor_unittest_SRCS dbcsr_tensor_unittest.F)
set(dbcsr_tas_unittest_SRCS dbcsr_tas_unittest.F)
set(dbcsr_test_csr_conversions_SRCS dbcsr_test_csr_conversions.F)
set(dbcsr_test_cpp_SRCS dbcsr_test.cpp)
set(dbcsr_tensor_test_cpp_SRCS dbcsr_tensor_test.cpp)
set(dbcsr_acc_test_c_SRCS dbcsr_acc_test.c)

# Make a list of the source files of fortran tests
set(DBCSR_TESTS_SRCS_FTN)
foreach (dbcsr_test ${DBCSR_TESTS_FTN})
  set(DBCSR_TESTS_SRCS_FTN ${DBCSR_TESTS_SRCS_FTN} ${${dbcsr_test}_SRCS})
endforeach ()

# Common object files linked to all tests
set(dbcsr_unittest_common_SRCS dbcsr_test_add.F dbcsr_test_multiply.F)

# instead of building a full-blown lib, it would be better to simply build an
# OBJECT lib, but we would need cmake 3.12 to be able to specify
# target_link_libraries on those to get the proper compile flags
add_library(dbcsr_unittest_common STATIC ${dbcsr_unittest_common_SRCS})
target_link_libraries(dbcsr_unittest_common PUBLIC ${BLAS_LIBRARIES}
                                                   ${LAPACK_LIBRARIES})
if (OpenMP_FOUND)
  target_link_libraries(dbcsr_unittest_common PUBLIC OpenMP::OpenMP_Fortran)
endif ()

if (APPLE AND BLAS_LIBRARIES MATCHES "Accelerate")
  target_compile_definitions(dbcsr_unittest_common PRIVATE __ACCELERATE)
endif ()
target_link_libraries(dbcsr_unittest_common PUBLIC dbcsr)

# Compile Fortran tests
foreach (dbcsr_test ${DBCSR_TESTS_FTN})
  add_executable(${dbcsr_test} ${${dbcsr_test}_SRCS})
  target_link_libraries(${dbcsr_test} dbcsr_unittest_common)
  set_target_properties(${dbcsr_test} PROPERTIES LINKER_LANGUAGE Fortran)
  # register unittest executable with CMake
  if (USE_MPI)
    separate_arguments(MPIEXEC_PREFLAGS)
    add_test(
      NAME ${dbcsr_test}
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_ranks}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${dbcsr_test}> ${MPIEXEC_POSTFLAGS})
  else ()
    add_test(NAME ${dbcsr_test} COMMAND ${dbcsr_test})
  endif ()
  if (OpenMP_FOUND)
    target_link_libraries(${dbcsr_test} OpenMP::OpenMP_Fortran)
    set_tests_properties(
      ${dbcsr_test} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=${TEST_OMP_THREADS})
  endif ()
endforeach ()

# set the __SHORT_FILE__ per file for dbcsr sources
foreach (tests_src ${DBCSR_PERF_SRCS} ${dbcsr_unittest_common_SRCS}
                   ${DBCSR_TESTS_SRCS_FTN})
  # add_fypp_sources returns a path in the current binary dir
  get_filename_component(short_file "${tests_src}" NAME)
  set_source_files_properties(
    ${tests_src} PROPERTIES COMPILE_DEFINITIONS __SHORT_FILE__="${short_file}")
endforeach ()

if (WITH_C_API)
  foreach (dbcsr_test_cpp_src ${DBCSR_TESTS_SRCS_CPP})
    get_filename_component(dbcsr_test_cpp_name ${dbcsr_test_cpp_src} NAME_WE)
    add_executable(${dbcsr_test_cpp_name} ${dbcsr_test_cpp_src})
    target_link_libraries(${dbcsr_test_cpp_name} dbcsr_c MPI::MPI_CXX)
    # register unittest executable with CMake
    if (USE_MPI)
      separate_arguments(MPIEXEC_PREFLAGS)
      add_test(
        NAME ${dbcsr_test_cpp_name}
        COMMAND
          ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_ranks}
          ${MPIEXEC_PREFLAGS} ./${dbcsr_test_cpp_name} ${MPIEXEC_POSTFLAGS})
    else ()
      add_test(NAME ${dbcsr_test_cpp_name} COMMAND ./${dbcsr_test_cpp_name})
    endif ()
    if (OpenMP_FOUND)
      set_tests_properties(
        ${dbcsr_test_cpp_name} PROPERTIES ENVIRONMENT
                                          OMP_NUM_THREADS=${TEST_OMP_THREADS})
    endif ()
  endforeach ()

  foreach (dbcsr_test_c ${DBCSR_TESTS_C})
    add_executable(${dbcsr_test_c} ${${dbcsr_test_c}_SRCS})
    target_link_libraries(${dbcsr_test_c} dbcsr_c dbcsr)
    # register unittest executable with CMake
    if (USE_MPI)
      separate_arguments(MPIEXEC_PREFLAGS)
      add_test(
        NAME ${dbcsr_test_c}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_ranks}
                ${MPIEXEC_PREFLAGS} ./${dbcsr_test_c} ${MPIEXEC_POSTFLAGS})
    else ()
      add_test(NAME ${dbcsr_test_c} COMMAND ./${dbcsr_test_c})
    endif ()
    if (OpenMP_FOUND)
      set_tests_properties(
        ${dbcsr_test_c} PROPERTIES ENVIRONMENT
                                   OMP_NUM_THREADS=${TEST_OMP_THREADS})
    endif ()
  endforeach ()
endif ()

# =================================== GPU BACKEND TESTS (CUDA / HIP)

# Add custom commands for the test files that need to be generated from a
# template
file(RELATIVE_PATH CURRENT_BINARY_DIR_RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/..
     ${CMAKE_CURRENT_BINARY_DIR})

# libsmm_acc_unittest_multiply
add_custom_target(
  generate_libsmm_acc_unittest_multiply_test_cpp
  COMMAND
    ${Python_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_libsmm_acc_unittest_multiply.py
    --base_dir ${CMAKE_CURRENT_SOURCE_DIR}/.. --out_dir
    ${CURRENT_BINARY_DIR_RELATIVE} --gpu_version=${WITH_GPU}
  DEPENDS libsmm_acc_unittest_multiply.cpp.template
          generate_libsmm_acc_unittest_multiply.py
  BYPRODUCTS libsmm_acc_unittest_multiply.cpp
  COMMENT "Generate tests/libsmm_acc_unittest_multiply.cpp")

# libsmm_acc_timer_multiply
add_custom_target(
  generate_libsmm_acc_timer_multiply_test_cpp
  COMMAND
    ${Python_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_libsmm_acc_timer_multiply.py --base_dir
    ${CMAKE_CURRENT_SOURCE_DIR}/.. --out_dir ${CURRENT_BINARY_DIR_RELATIVE}
    --gpu_version=${WITH_GPU}
  DEPENDS libsmm_acc_timer_multiply.cpp.template
          generate_libsmm_acc_timer_multiply.py
  BYPRODUCTS libsmm_acc_timer_multiply.cpp
  COMMENT "Generate tests/generate_libsmm_acc_timer_multiply.cpp")

if (USE_ACCEL MATCHES "cuda|hip")

  # All libsmm_acc tests
  set(LIBSMM_ACC_TESTS_SRCS
      ${CMAKE_CURRENT_BINARY_DIR}/libsmm_acc_unittest_multiply.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/libsmm_acc_timer_multiply.cpp
      libsmm_acc_unittest_transpose.cpp)

  # Add executables for all libsmm_acc tests
  foreach (libsmm_acc_test ${LIBSMM_ACC_TESTS_SRCS})
    get_filename_component(libsmm_acc_test_name ${libsmm_acc_test} NAME_WE)
    add_executable(${libsmm_acc_test_name} ${libsmm_acc_test})
    target_compile_definitions(
      ${libsmm_acc_test_name} PRIVATE $<$<STREQUAL:${USE_ACCEL},hip>:__HIP>
                                      $<$<STREQUAL:${USE_ACCEL},cuda>:__CUDA>)
    target_link_libraries(${libsmm_acc_test_name} dbcsr
                          $<$<STREQUAL:${USE_ACCEL},hip>:hip::host> libsmm_acc)
  endforeach ()

  add_test(NAME libsmm_acc_unittest_multiply
           COMMAND libsmm_acc_unittest_multiply)
  add_test(NAME libsmm_acc_unittest_transpose
           COMMAND libsmm_acc_unittest_transpose)
  add_test(NAME libsmm_acc_timer_multiply-autotuned
           COMMAND libsmm_acc_timer_multiply autotuned)
  add_test(NAME libsmm_acc_timer_multiply-predicted
           COMMAND libsmm_acc_timer_multiply predicted)

endif ()

# =================================== DOCUMENTATION GENERATION Copy test source
# files into the build directory so that their documentation can be generated by
# FORD
set(DBCSR_TESTS dbcsr_performance_driver.F ${DBCSR_TESTS_SRCS_FTN}
                ${DBCSR_TESTS_SRCS_CPP} libsmm_acc_unittest_transpose.cpp)

# Make a list of the copy commands
set(test_copy_commands)
foreach (test ${DBCSR_TESTS})
  list(
    APPEND
    test_copy_commands
    COMMAND
    ${CMAKE_COMMAND}
    -E
    copy
    ${CMAKE_SOURCE_DIR}/tests/${test}
    ${CMAKE_BINARY_DIR}/tests)
endforeach ()

add_custom_target(
  doc_copy_tests
  COMMENT "Copy tests for documentation generation"
  COMMAND mkdir -p ${CMAKE_BINARY_DIR}/tests ${test_copy_commands}
  VERBATIM)

# libsmm_acc_unittest_multiply.cpp and libsmm_acc_timer_multiply.cpp do not need
# to be copied to the build directory since they are generated at build-time and
# written to the build directory directly. We just need to make sure that the
# documentation generation depends on the generation of these tests.
add_dependencies(doc_copy_tests generate_libsmm_acc_unittest_multiply_test_cpp)
add_dependencies(doc_copy_tests generate_libsmm_acc_timer_multiply_test_cpp)
