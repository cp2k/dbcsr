cmake_minimum_required (VERSION 3.0)

# If available, enable developer warnings for the build system
if (POLICY CMP0005)
    cmake_policy(SET CMP0005 NEW)
endif ()

enable_language(Fortran)
project(dbcsr Fortran)
enable_testing() # enables the `make test` target

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

option(USE_MPI "Build with MPI support" ON)
if (USE_MPI)
  find_package(MPI COMPONENTS Fortran REQUIRED)
endif ()

# make sure that the default is a RELEASE
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE CACHE STRING
    "Choose the type of build, options are: Debug Release."
    FORCE)
endif ()

# Compiler detection
get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)

if (Fortran_COMPILER_NAME MATCHES "gfortran.*")
  set(CMAKE_Fortran_FLAGS "-ffree-form -ffree-line-length-none -std=f2003 -fopenmp")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -funroll-loops -ffast-math")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -ggdb")
else ()
  message("CMAKE_Fortran_COMPILER full path: " ${CMAKE_Fortran_COMPILER})
  message("Fortran compiler: " ${Fortran_COMPILER_NAME})
  message("No optimized Fortran compiler flags are known, we just try -O2...")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O2")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g")
endif ()


if (MPI_FOUND)
  add_definitions(-D__parallel -D__MPI_VERSION=${MPI_Fortran_VERSION_MAJOR})

  # Instead of resetting the compiler, we are adding the compiler flags
  # originally added by the mpifort-wrapper, based on
  #   https://cmake.org/pipermail/cmake/2012-June/050991.html
  # Here we assume that the MPI implementation found uses the same compiler
  # as the Fortran compiler we found prior. Otherwise we might be adding incompatible
  # compiler flags at this point.
  add_definitions(${MPI_Fortran_COMPILE_FLAGS})
  include_directories(${MPI_Fortran_INCLUDE_DIRS})
  link_directories(${MPI_Fortran_LIBRARY_DIRS})
endif ()

# Use a system-provided fypp if available, otherwise the bundled one
find_program(FYPP_EXECUTABLE fypp DOC "The FYPP preprocessor" PATHS tools/build_utils/)
if (NOT FYPP_EXECUTABLE)
  message(FATAL_ERROR "Failed to find the FYPP preprocessor.")
else ()
  message(STATUS "FYPP preprocessor found.")
endif ()

function (ADD_FYPP_SOURCES OUTVAR)
  set(outfiles)
  foreach (f ${ARGN})
    # first we might need to make the input file absolute
    get_filename_component(f "${f}" ABSOLUTE)
    # get the relative path of the file to the current source dir
    file(RELATIVE_PATH rf "${CMAKE_CURRENT_SOURCE_DIR}" "${f}")
    # set the output filename of fypped sources
    set(of "${CMAKE_CURRENT_BINARY_DIR}/${rf}")

    # create the output directory if it doesn't exist
    get_filename_component(d "${of}" PATH)
    if (NOT IS_DIRECTORY "${d}")
      file(MAKE_DIRECTORY "${d}")
    endif ()

    if ("${f}" MATCHES ".F$")
      # append the output file to the list of outputs
      list(APPEND outfiles "${of}")
      # now add the custom command to generate the output file
      add_custom_command(OUTPUT "${of}" COMMAND ${FYPP_EXECUTABLE} ARGS "${f}" "${of}" DEPENDS "${f}")
    else ()
      configure_file("${f}" "${of}" COPYONLY)
    endif ()
  endforeach ()
  # set the output list in the calling scope
  set(${OUTVAR} ${outfiles} PARENT_SCOPE)
endfunction ()

add_fypp_sources(DBCSR_SRCS
  src/acc/dbcsr_acc_device.F
  src/acc/dbcsr_acc_devmem.F
  src/acc/dbcsr_acc_event.F
  src/acc/dbcsr_acc_hostmem.F
  src/acc/dbcsr_acc_stream.F
  src/acc/cublaswrap/dbcsr_cublas.F
  src/base/dbcsr_base_hooks.F
  src/base/dbcsr_kinds.F
  src/base/dbcsr_machine.F
  src/base/dbcsr_machine_internal.F
  src/block/dbcsr_block_access.F
  src/block/dbcsr_block_operations.F
  src/block/dbcsr_data_operations.F
  src/block/dbcsr_index_operations.F
  src/block/dbcsr_iterator_operations.F
  src/block/dbcsr_work_operations.F
  src/common/dbcsr_array_sort.F
  src/common/dbcsr_array_utils.F
  src/common/dbcsr_cuda_profiling.F
  src/common/dbcsr_dict.F
  src/common/dbcsr_distribution_1d_types.F
  src/common/dbcsr_error_handling.F
  src/common/dbcsr_files.F
  src/common/dbcsr_iter_types.F
  src/common/dbcsr_kahan_sum.F
  src/common/dbcsr_list.F
  src/common/dbcsr_list_callstackentry.F
  src/common/dbcsr_list_routinereport.F
  src/common/dbcsr_list_routinestat.F
  src/common/dbcsr_list_timerenv.F
  src/common/dbcsr_log_handling.F
  src/common/dbcsr_memory_utilities.F
  src/common/dbcsr_para_env.F
  src/common/dbcsr_para_types.F
  src/common/dbcsr_parallel_rng_types.F
  src/common/dbcsr_print_messages.F
  src/common/dbcsr_result_methods.F
  src/common/dbcsr_result_types.F
  src/common/dbcsr_string_table.F
  src/common/dbcsr_string_utilities.F
  src/common/dbcsr_timings.F
  src/common/dbcsr_timings_base_type.F
  src/common/dbcsr_timings_report.F
  src/common/dbcsr_timings_types.F
  src/common/dbcsr_toollib.F
  src/common/dbcsr_util.F
  src/common/dbcsr_xml_parser.F
  src/data/dbcsr_data_methods.F
  src/data/dbcsr_data_methods_low.F
  src/data/dbcsr_data_types.F
  src/data/dbcsr_mem_methods.F
  src/data/dbcsr_ptr_util.F
  src/dbcsr_api.F
  src/dbcsr_base/dbcsr_array_types.F
  src/dbcsr_base/dbcsr_btree.F
  src/dbcsr_base/dbcsr_config.F
  src/dbcsr_base/dbcsr_methods.F
  src/dbcsr_base/dbcsr_types.F
  src/dbcsr_lib.F
  src/dbcsr_test_methods.F
  src/dbcsr_tests.F
  src/dist/dbcsr_dist_methods.F
  src/dist/dbcsr_dist_operations.F
  src/dist/dbcsr_dist_util.F
  src/dist/dbcsr_min_heap.F
  src/dist/dbcsr_mp_methods.F
  src/dist/dbcsr_mp_operations.F
  src/mm/dbcsr_acc_operations.F
  src/mm/dbcsr_mm.F
  src/mm/dbcsr_mm_3d.F
  src/mm/dbcsr_mm_accdrv.F
  src/mm/dbcsr_mm_cannon.F
  src/mm/dbcsr_mm_common.F
  src/mm/dbcsr_mm_csr.F
  src/mm/dbcsr_mm_dist_operations.F
  src/mm/dbcsr_mm_hostdrv.F
  src/mm/dbcsr_mm_multrec.F
  src/mm/dbcsr_mm_sched.F
  src/mm/dbcsr_mm_types.F
  src/mm/dbcsr_multiply_api.F
  src/mpiwrap/dbcsr_message_passing.F
  src/ops/dbcsr_blas_operations.F
  src/ops/dbcsr_csr_conversions.F
  src/ops/dbcsr_io.F
  src/ops/dbcsr_operations.F
  src/ops/dbcsr_transformations.F
  src/tensors/dbcsr_allocate_wrap.F
  src/tensors/dbcsr_array_list_methods.F
  src/tensors/dbcsr_tensor.F
  src/tensors/dbcsr_tensor_api.F
  src/tensors/dbcsr_tensor_block.F
  src/tensors/dbcsr_tensor_index.F
  src/tensors/dbcsr_tensor_reshape.F
  src/tensors/dbcsr_tensor_test.F
  src/tensors/dbcsr_tensor_types.F
  )

add_library(dbcsr SHARED ${DBCSR_SRCS})
target_link_libraries(dbcsr ${LAPACK_LIBRARIES} ${MPI_Fortran_LIBRARIES})
include_directories(src/ src/base src/common)
set_target_properties(dbcsr PROPERTIES COMPILE_DEFINITIONS "__HAS_NO_SHARED_GLIBC; __STATM_TOTAL")
set_target_properties(dbcsr PROPERTIES LINKER_LANGUAGE Fortran)

set(DBCSR_PERF_SRCS
  tests/dbcsr_performance_driver.F
  tests/dbcsr_performance_multiply.F)
add_executable(dbcsr_perf ${DBCSR_PERF_SRCS})
target_link_libraries(dbcsr_perf dbcsr ${MPI_Fortran_LIBRARIES})


set(DBCSR_UNITTEST_SRCS
  tests/dbcsr_unittest.F
  tests/dbcsr_test_add.F
  tests/dbcsr_test_multiply.F
  )
add_executable(dbcsr_unittest ${DBCSR_UNITTEST_SRCS})
target_link_libraries(dbcsr_unittest dbcsr)
# register unittest executable with CMake
add_test(NAME dbcsr_unittest COMMAND dbcsr_unittest)

add_executable(dbcsr_test_csr_conversions tests/dbcsr_test_csr_conversions.F)
target_link_libraries(dbcsr_test_csr_conversions dbcsr)
add_test(NAME dbcsr_test_csr_conversions COMMAND dbcsr_test_csr_conversions)


set(DBCSR_PROGRAM_SRCS
  examples/dbcsr_example_1.F
  examples/dbcsr_example_2.F
  examples/dbcsr_example_3.F
  )

# register each program source file as executable
foreach (dbcsr_program_src ${DBCSR_PROGRAM_SRCS})
  get_filename_component(dbcsr_program_name ${dbcsr_program_src} NAME_WE)
  add_executable(${dbcsr_program_name} ${dbcsr_program_src})
  target_link_libraries(${dbcsr_program_name} dbcsr ${MPI_Fortran_LIBRARIES})
endforeach ()

# set the __SHORT_FILE__ per file for dbcsr sources
foreach (dbcsr_src ${DBCSR_SRCS} ${DBCSR_PERF_SRCS} ${DBCSR_UNITTEST_SRCS} ${DBCSR_PROGRAM_SRCS})
  # add_fypp_sources returns a path in the current binary dir
  get_filename_component(dbcsr_src "${dbcsr_src}" ABSOLUTE)
  file(RELATIVE_PATH short_file "${CMAKE_CURRENT_BINARY_DIR}" "${dbcsr_src}")
  set_source_files_properties(${dbcsr_src} PROPERTIES COMPILE_DEFINITIONS __SHORT_FILE__="dbcsr/${short_file}")
endforeach ()
