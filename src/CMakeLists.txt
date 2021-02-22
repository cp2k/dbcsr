# =================================================================================================
# INCLUDE
include(fypp-sources)
include(GNUInstallDirs) # required to get a proper LIBDIR variable
include(CMakePackageConfigHelpers)

# =================================================================================================
# SOURCE FILE LISTS
add_fypp_sources(
  DBCSR_FORTRAN_SRCS
  dbcsr_api.F
  acc/cuda/dbcsr_cuda_profiling.F
  acc/dbcsr_acc_device.F
  acc/dbcsr_acc_devmem.F
  acc/dbcsr_acc_event.F
  acc/dbcsr_acc_hostmem.F
  acc/dbcsr_acc_init.F
  acc/dbcsr_acc_stream.F
  acc/dbcsr_acc_timings.F
  acc/hip/dbcsr_hipblas.F
  base/dbcsr_base_hooks.F
  base/dbcsr_kinds.F
  base/dbcsr_machine.F
  base/dbcsr_machine_internal.F
  block/dbcsr_block_access.F
  block/dbcsr_block_operations.F
  block/dbcsr_index_operations.F
  block/dbcsr_iterator_operations.F
  core/dbcsr_array_types.F
  core/dbcsr_config.F
  core/dbcsr_lib.F
  core/dbcsr_methods.F
  core/dbcsr_types.F
  core/dbcsr_dict.F
  core/dbcsr_error_handling.F
  core/dbcsr_iter_types.F
  core/dbcsr_list_callstackentry.F
  core/dbcsr_list.F
  core/dbcsr_list_routinereport.F
  core/dbcsr_list_routinestat.F
  core/dbcsr_list_timerenv.F
  core/dbcsr_log_handling.F
  core/dbcsr_print_messages.F
  core/dbcsr_timings_base_type.F
  core/dbcsr_timings.F
  core/dbcsr_timings_report.F
  core/dbcsr_timings_types.F
  data/dbcsr_data_operations.F
  data/dbcsr_data_methods.F
  data/dbcsr_data_methods_low.F
  data/dbcsr_data_types.F
  data/dbcsr_mem_methods.F
  data/dbcsr_ptr_util.F
  dist/dbcsr_dist_methods.F
  dist/dbcsr_dist_operations.F
  dist/dbcsr_dist_util.F
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
  mpi/dbcsr_mp_methods.F
  mpi/dbcsr_mp_operations.F
  mpi/dbcsr_mpiwrap.F
  ops/dbcsr_csr_conversions.F
  ops/dbcsr_io.F
  ops/dbcsr_operations.F
  ops/dbcsr_test_methods.F
  ops/dbcsr_tests.F
  ops/dbcsr_transformations.F
  tas/dbcsr_tas_base.F
  tas/dbcsr_tas_global.F
  tas/dbcsr_tas_io.F
  tas/dbcsr_tas_mm.F
  tas/dbcsr_tas_reshape_ops.F
  tas/dbcsr_tas_split.F
  tas/dbcsr_tas_test.F
  tas/dbcsr_tas_types.F
  tas/dbcsr_tas_util.F
  tensors/dbcsr_allocate_wrap.F
  tensors/dbcsr_array_list_methods.F
  tensors/dbcsr_tensor_api.F
  tensors/dbcsr_tensor_block.F
  tensors/dbcsr_tensor.F
  tensors/dbcsr_tensor_index.F
  tensors/dbcsr_tensor_io.F
  tensors/dbcsr_tensor_reshape.F
  tensors/dbcsr_tensor_split.F
  tensors/dbcsr_tensor_test.F
  tensors/dbcsr_tensor_types.F
  utils/dbcsr_array_sort.F
  utils/dbcsr_blas_operations.F
  utils/dbcsr_btree.F
  utils/dbcsr_files.F
  utils/dbcsr_min_heap.F
  utils/dbcsr_string_utilities.F
  utils/dbcsr_toollib.F
  work/dbcsr_work_operations.F)

set(DBCSR_HIP_AND_CUDA_SRCS
    acc/libsmm_acc/libsmm_acc_benchmark.cpp
    acc/libsmm_acc/libsmm_acc_init.cpp
    acc/libsmm_acc/libsmm_acc.cpp
    acc/cuda/acc_cublas.cpp
    acc/cuda/acc_dev.cpp
    acc/cuda/acc_error.cpp
    acc/cuda/acc_event.cpp
    acc/cuda/acc_init.cpp
    acc/cuda/acc_mem.cpp
    acc/cuda/acc_stream.cpp)

set(DBCSR_CUDA_SRCS ${DBCSR_HIP_AND_CUDA_SRCS} acc/cuda/acc_cuda.cpp
                    acc/cuda/dbcsr_cuda_nvtx_cu.cpp)

set(DBCSR_HIP_SRCS ${DBCSR_HIP_AND_CUDA_SRCS} acc/hip/acc_hip.cpp)

set(DBCSR_OPENCL_SRCS
    acc/opencl/smm/opencl_libsmm.c acc/opencl/acc_opencl.c
    acc/opencl/acc_opencl_event.c acc/opencl/acc_opencl_mem.c
    acc/opencl/acc_opencl_stream.c)

# set the __SHORT_FILE__ per file for dbcsr sources
foreach (dbcsr_src ${DBCSR_FORTRAN_SRCS})
  # add_fypp_sources returns a path in the current binary dir
  get_filename_component(short_file "${dbcsr_src}" NAME)
  set_source_files_properties(
    ${dbcsr_src} PROPERTIES COMPILE_DEFINITIONS __SHORT_FILE__="${short_file}")
endforeach ()

set(DBCSR_SRCS ${DBCSR_FORTRAN_SRCS})

if (USE_ACCEL MATCHES "cuda")
  set(DBCSR_SRCS ${DBCSR_SRCS} ${DBCSR_CUDA_SRCS})
elseif (USE_ACCEL MATCHES "hip")
  set(DBCSR_SRCS ${DBCSR_SRCS} ${DBCSR_HIP_SRCS})
elseif (USE_ACCEL MATCHES "opencl")
  set(DBCSR_SRCS ${DBCSR_SRCS} ${DBCSR_OPENCL_SRCS})
endif ()

# =================================================================================================
# DBCSR LIBRARY
add_library(dbcsr ${DBCSR_SRCS})

set_target_properties(dbcsr PROPERTIES VERSION ${dbcsr_VERSION}
                                       SOVERSION ${dbcsr_APIVERSION})

if (USE_SMM MATCHES "libxsmm")
  target_compile_definitions(dbcsr PRIVATE __LIBXSMM)
  target_link_libraries(dbcsr PRIVATE PkgConfig::LIBXSMM)
endif ()

if (BLAS_LIBRARIES MATCHES "mkl_")
  target_compile_definitions(dbcsr PRIVATE __MKL)
endif ()

if (APPLE)
  # fix /proc/self/statm can not be opened on macOS
  target_compile_definitions(dbcsr PRIVATE __NO_STATM_ACCESS)

  if (BLAS_LIBRARIES MATCHES "Accelerate")
    target_compile_definitions(dbcsr PRIVATE __ACCELERATE)
  endif ()
endif ()

# set -DNDEBUG for Release builds
target_compile_definitions(dbcsr PRIVATE $<$<CONFIG:Release>:NDEBUG>)

target_link_libraries(dbcsr PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
target_include_directories(
  dbcsr PRIVATE base) # do not export those includes, but some srcs do an
                      # unprefixed include
# make sure dependencies of dbcsr find the dbcsr_api.mod file plus some files
# they usually include:
target_include_directories(
  dbcsr
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_definitions(dbcsr PRIVATE __STATM_TOTAL)
set_target_properties(dbcsr PROPERTIES LINKER_LANGUAGE Fortran)

if (MPI_FOUND)
  # once built, a user of the dbcsr library can not influence anything anymore
  # by setting those flags:
  target_compile_definitions(
    dbcsr PRIVATE __parallel __MPI_VERSION=${MPI_Fortran_VERSION_MAJOR})

  # Instead of resetting the compiler for MPI, we are adding the compiler flags
  # otherwise added by the mpifort-wrapper directly; based on hints from:
  # https://cmake.org/pipermail/cmake/2012-June/050991.html Here we assume that
  # the MPI implementation found uses the same compiler as the Fortran compiler
  # we found prior. Otherwise we might be adding incompatible compiler flags at
  # this point. when built against MPI, a dbcsr consumer has to specify the MPI
  # flags as well, therefore: PUBLIC
  target_link_libraries(dbcsr PUBLIC MPI::MPI_Fortran)
endif ()

target_link_libraries(
  dbcsr
  PRIVATE $<$<BOOL:${USE_OPENMP}>:OpenMP::OpenMP_C>
          $<$<BOOL:${USE_OPENMP}>:OpenMP::OpenMP_CXX>
          $<$<BOOL:${USE_OPENMP}>:OpenMP::OpenMP_Fortran>)

# todo, make this a bit better with opencl.
if (USE_ACCEL MATCHES "cuda|hip")
  add_subdirectory(acc/libsmm_acc)
endif ()

if (USE_ACCEL MATCHES "opencl")
  add_subdirectory(acc/opencl/smm)
endif ()

if (USE_ACCEL)
  target_compile_definitions(
    dbcsr
    PRIVATE __DBCSR_ACC
            $<$<STREQUAL:${USE_ACCEL},cuda>:__CUDA>
            $<$<STREQUAL:${USE_ACCEL},opencl>:__OPENCL>
            $<$<STREQUAL:${USE_ACCEL},cuda>:ARCH_NUMBER=${ACC_ARCH_NUMBER}>
            $<$<STREQUAL:${USE_ACCEL},hip>:__HIP>
            $<$<STREQUAL:${USE_ACCEL},hip>:ARCH_NUMBER=${ACC_ARCH_NUMBER}>
            $<$<BOOL:${WITH_CUDA_PROFILING}>:__CUDA_PROFILING>)

  target_link_libraries(
    dbcsr
    PRIVATE $<$<STREQUAL:${USE_ACCEL},cuda>:CUDA::cudart>
            $<$<STREQUAL:${USE_ACCEL},cuda>:CUDA::cublas>
            $<$<STREQUAL:${USE_ACCEL},cuda>:CUDA::nvrtc>
            $<$<BOOL:${WITH_CUDA_PROFILING}>:CUDA::nvToolsExt>
            $<$<STREQUAL:${USE_ACCEL},hip>:roc::hipblas>
            $<$<STREQUAL:${USE_ACCEL},hip>:hip::host>
            $<$<STREQUAL:${USE_ACCEL},opencl>:OpenCL::OpenCL>)
endif ()

# =================================================================================================
# DBCSR's C API
if (WITH_C_API)
  # Build the C API as a separate library
  add_fypp_sources(DBCSR_C_SRCS dbcsr.h dbcsr_api_c.F tensors/dbcsr_tensor_api_c.F
                   tensors/dbcsr_tensor.h)

  add_library(dbcsr_c ${DBCSR_C_SRCS})
  set_target_properties(dbcsr_c PROPERTIES LINKER_LANGUAGE Fortran)

  set_target_properties(dbcsr_c PROPERTIES VERSION ${dbcsr_VERSION}
                                           SOVERSION ${dbcsr_APIVERSION})

  target_link_libraries(dbcsr_c PRIVATE dbcsr)
  target_link_libraries(dbcsr_c PUBLIC MPI::MPI_C) # the C API always needs MPI
  target_include_directories(
    dbcsr_c
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> # change order so compiler
                                                     # first checks binary
                                                     # directory
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
endif ()

# =================================================================================================
# INSTALL

set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
set(config_namespace "DBCSR::")

# Install targets
install(
  TARGETS dbcsr
  EXPORT DBCSRTargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dbcsr_api.mod"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dbcsr_tensor_api.mod"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

if (WITH_C_API)
  install(
    TARGETS dbcsr_c
    EXPORT DBCSRTargets
    COMPONENT C
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  install(
	  FILES "${CMAKE_CURRENT_BINARY_DIR}/dbcsr.h"
          "${CMAKE_CURRENT_BINARY_DIR}/tensors/dbcsr_tensor.h"
    COMPONENT C
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif ()

configure_package_config_file(
  cmake/DBCSRConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/DBCSRConfig.cmake"
  INSTALL_DESTINATION "${config_install_dir}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DBCSRConfigVersion.cmake"
  VERSION "${dbcsr_VERSION}"
  COMPATIBILITY SameMajorVersion)
install(
  EXPORT DBCSRTargets
  NAMESPACE "${config_namespace}"
  DESTINATION "${config_install_dir}")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/DBCSRConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/DBCSRConfigVersion.cmake
        DESTINATION ${config_install_dir})
