
include(fypp-sources)

add_fypp_sources(DBCSR_SRCS
  dbcsr_api.F
  acc/cublaswrap/dbcsr_cublas.F
  acc/dbcsr_acc_device.F
  acc/dbcsr_acc_devmem.F
  acc/dbcsr_acc_event.F
  acc/dbcsr_acc_hostmem.F
  acc/dbcsr_acc_stream.F
  base/dbcsr_base_hooks.F
  base/dbcsr_kinds.F
  base/dbcsr_machine.F
  base/dbcsr_machine_internal.F
  block/dbcsr_block_access.F
  block/dbcsr_block_operations.F
  block/dbcsr_data_operations.F
  block/dbcsr_index_operations.F
  block/dbcsr_iterator_operations.F
  block/dbcsr_work_operations.F
  core/dbcsr_array_types.F
  core/dbcsr_config.F
  core/dbcsr_lib.F
  core/dbcsr_methods.F
  core/dbcsr_types.F
  data/dbcsr_data_methods.F
  data/dbcsr_data_methods_low.F
  data/dbcsr_data_types.F
  data/dbcsr_mem_methods.F
  data/dbcsr_ptr_util.F
  dist/dbcsr_dist_methods.F
  dist/dbcsr_dist_operations.F
  dist/dbcsr_dist_util.F
  dist/dbcsr_mp_methods.F
  dist/dbcsr_mp_operations.F
  extras/dbcsr_cuda_profiling.F
  extras/dbcsr_dict.F
  extras/dbcsr_error_handling.F
  extras/dbcsr_iter_types.F
  extras/dbcsr_list_callstackentry.F
  extras/dbcsr_list.F
  extras/dbcsr_list_routinereport.F
  extras/dbcsr_list_routinestat.F
  extras/dbcsr_list_timerenv.F
  extras/dbcsr_log_handling.F
  extras/dbcsr_memory_utilities.F
  extras/dbcsr_para_env.F
  extras/dbcsr_para_types.F
  extras/dbcsr_print_messages.F
  extras/dbcsr_timings_base_type.F
  extras/dbcsr_timings.F
  extras/dbcsr_timings_report.F
  extras/dbcsr_timings_types.F
  mm/dbcsr_acc_operations.F
  mm/dbcsr_mm_3d.F
  mm/dbcsr_mm_accdrv.F
  mm/dbcsr_mm_cannon.F
  mm/dbcsr_mm_common.F
  mm/dbcsr_mm_csr.F
  mm/dbcsr_mm_dist_operations.F
  mm/dbcsr_mm.F
  mm/dbcsr_mm_hostdrv.F
  mm/dbcsr_mm_multrec.F
  mm/dbcsr_mm_sched.F
  mm/dbcsr_mm_types.F
  mm/dbcsr_multiply_api.F
  mpiwrap/dbcsr_message_passing.F
  ops/dbcsr_blas_operations.F
  ops/dbcsr_csr_conversions.F
  ops/dbcsr_io.F
  ops/dbcsr_operations.F
  ops/dbcsr_test_methods.F
  ops/dbcsr_tests.F
  ops/dbcsr_transformations.F
  tensors/dbcsr_allocate_wrap.F
  tensors/dbcsr_array_list_methods.F
  tensors/dbcsr_tensor_api.F
  tensors/dbcsr_tensor_block.F
  tensors/dbcsr_tensor.F
  tensors/dbcsr_tensor_index.F
  tensors/dbcsr_tensor_reshape.F
  tensors/dbcsr_tensor_test.F
  tensors/dbcsr_tensor_types.F
  utils/dbcsr_array_sort.F
  utils/dbcsr_btree.F
  utils/dbcsr_files.F
  utils/dbcsr_min_heap.F
  utils/dbcsr_string_utilities.F
  utils/dbcsr_toollib.F
  )

add_library(dbcsr SHARED ${DBCSR_SRCS})
target_link_libraries(dbcsr ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${MPI_Fortran_LIBRARIES})
target_include_directories(dbcsr PRIVATE base) # do not export those includes
# but make sure dependencies of dbcsr find the dbcsr_api.mod file plus some files they usually include:
target_include_directories(dbcsr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}) 
target_compile_definitions(dbcsr PRIVATE __STATM_TOTAL)
set_target_properties(dbcsr PROPERTIES LINKER_LANGUAGE Fortran)

if (MPI_FOUND)
  # once built, a user of the dbcsr library can not influence anything anmore by setting those flags:
  target_compile_definitions(dbcsr PRIVATE __parallel __MPI_VERSION=${MPI_Fortran_VERSION_MAJOR})

  # Instead of resetting the compiler for MPI, we are adding the compiler flags
  # otherwise added by the mpifort-wrapper directly; based on hints from:
  #   https://cmake.org/pipermail/cmake/2012-June/050991.html
  # Here we assume that the MPI implementation found uses the same compiler
  # as the Fortran compiler we found prior. Otherwise we might be adding incompatible
  # compiler flags at this point.
  # when built against MPI, a dbcsr consumer has to specify the MPI flags as well, therefore: PUBLIC
  target_compile_options(dbcsr PUBLIC ${MPI_Fortran_COMPILE_FLAGS})
  target_include_directories(dbcsr PUBLIC ${MPI_Fortran_INCLUDE_DIRS})
endif ()

# set the __SHORT_FILE__ per file for dbcsr sources
foreach (dbcsr_src ${DBCSR_SRCS})
  # add_fypp_sources returns a path in the current binary dir
  get_filename_component(dbcsr_src "${dbcsr_src}" ABSOLUTE)
  file(RELATIVE_PATH short_file "${CMAKE_CURRENT_BINARY_DIR}" "${dbcsr_src}")
  set_source_files_properties(${dbcsr_src} PROPERTIES COMPILE_DEFINITIONS __SHORT_FILE__="dbcsr/${short_file}")
endforeach ()
