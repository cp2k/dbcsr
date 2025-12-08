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
  acc/hip/dbcsr_hip_profiling.F
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

set(DBCSR_HIP_AND_CUDA_COMMON_SRCS
    acc/cuda_hip/acc_blas.cpp
    acc/cuda_hip/acc_dev.cpp
    acc/cuda_hip/acc_error.cpp
    acc/cuda_hip/acc_event.cpp
    acc/cuda_hip/acc_utils.cpp
    acc/cuda_hip/acc_init.cpp
    acc/cuda_hip/acc_mem.cpp
    acc/cuda_hip/acc_stream.cpp)

set(DBCSR_ARCH_DEPENDENT_SRCS
    acc/cuda_hip/calculate_norms.cpp acc/libsmm_acc/libsmm_acc_benchmark.cpp
    acc/libsmm_acc/libsmm_acc_init.cpp acc/libsmm_acc/libsmm_acc.cpp)

set(DBCSR_CUDA_SRCS
    ${DBCSR_HIP_AND_CUDA_COMMON_SRCS} ${DBCSR_ARCH_DEPENDENT_SRCS}
    acc/cuda/acc_cuda.cpp acc/cuda/dbcsr_cuda_nvtx_cu.cpp)

set(DBCSR_HIP_SRCS ${DBCSR_HIP_AND_CUDA_COMMON_SRCS}
                   ${DBCSR_ARCH_DEPENDENT_SRCS} acc/hip/acc_hip.cpp)

if (USE_ACCEL MATCHES "hip")
  set_source_files_properties(acc/cuda_hip/calculate_norms.cpp
                              PROPERTIES LANGUAGE HIP)
  set_source_files_properties(acc/cuda_hip/calculate_norms.cpp
                              PROPERTIES COMPILE_FLAGS "-fPIE")
elseif (USE_ACCEL MATCHES "cuda")
  set_source_files_properties(acc/cuda_hip/calculate_norms.cpp
                              PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(acc/cuda_hip/calculate_norms.cpp
                              PROPERTIES COMPILE_FLAGS "--x cu")
endif ()

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

# override -Werror for certain translation units
if ((CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    AND (CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL 10))
  set_source_files_properties(mpi/dbcsr_mpiwrap.F PROPERTIES COMPILE_FLAGS
                                                             -Wno-error)
endif ()

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
if (MULTI_ARCH AND MULTI_GPU_BUILD)
  # Multi-architecture build: create separate libraries per GPU architecture

  # First, create a common library with all architecture-independent code
  set(DBCSR_COMMON_SRCS ${DBCSR_FORTRAN_SRCS})
  if (USE_ACCEL MATCHES "cuda")
    # Add only common CUDA sources (manually specify to exclude arch-specific
    # ones)
    list(APPEND DBCSR_COMMON_SRCS ${DBCSR_HIP_AND_CUDA_COMMON_SRCS}
         acc/cuda/acc_cuda.cpp acc/cuda/dbcsr_cuda_nvtx_cu.cpp)
    # Note: calculate_norms.cpp and libsmm_acc files are intentionally excluded
  elseif (USE_ACCEL MATCHES "hip")
    # Add only common HIP sources (manually specify to exclude arch-specific
    # ones)
    list(APPEND DBCSR_COMMON_SRCS ${DBCSR_HIP_AND_CUDA_COMMON_SRCS}
         acc/hip/acc_hip.cpp)
    # Note: calculate_norms.cpp and libsmm_acc files are intentionally excluded
  elseif (USE_ACCEL MATCHES "opencl")
    list(APPEND DBCSR_COMMON_SRCS ${DBCSR_OPENCL_SRCS})
  endif ()

  # Create common object library (compiled once, reused for all architectures)
  add_library(dbcsr_common OBJECT ${DBCSR_COMMON_SRCS})

  # Set common properties
  set_target_properties(dbcsr_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # Now create separate libraries for each GPU architecture
  foreach (GPU_ARCH IN LISTS WITH_GPU_LIST)
    # Get architecture number for this GPU
    set(ARCH_NUM ${GPU_ARCH_NUMBER_${GPU_ARCH}})

    # Create architecture-specific object library for calculate_norms.cpp
    add_library(dbcsr_arch_${GPU_ARCH} OBJECT)

    if (USE_ACCEL MATCHES "hip|cuda")
      target_sources(dbcsr_arch_${GPU_ARCH}
                     PRIVATE ${DBCSR_ARCH_DEPENDENT_SRCS})

      target_include_directories(
        dbcsr_arch_${GPU_ARCH}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/acc/libsmm_acc
                ${CMAKE_CURRENT_SOURCE_DIR}/acc/libsmm_acc)

      if (USE_ACCEL MATCHES "hip")
        set_source_files_properties(acc/cuda_hip/calculate_norms.cpp
                                    PROPERTIES LANGUAGE HIP)
        set_source_files_properties(acc/cuda_hip/calculate_norms.cpp
                                    PROPERTIES COMPILE_FLAGS "-fPIE")
        set_target_properties(dbcsr_arch_${GPU_ARCH}
                              PROPERTIES HIP_ARCHITECTURES "${ARCH_NUM}")
      elseif (USE_ACCEL MATCHES "cuda")
        set_source_files_properties(acc/cuda_hip/calculate_norms.cpp
                                    PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(acc/cuda_hip/calculate_norms.cpp
                                    PROPERTIES COMPILE_FLAGS "--x cu")
        set_target_properties(dbcsr_arch_${GPU_ARCH}
                              PROPERTIES CUDA_ARCHITECTURES "${ARCH_NUM}")
      endif ()

      # Set architecture-specific compile definitions
      target_compile_definitions(
        dbcsr_arch_${GPU_ARCH}
        PRIVATE __DBCSR_ACC
                $<$<STREQUAL:${USE_ACCEL},cuda>:__CUDA>
                $<$<STREQUAL:${USE_ACCEL},hip>:__HIP>
                ARCH_NUMBER=${ARCH_NUM}
                $<$<BOOL:${WITH_CUDA_PROFILING}>:__CUDA_PROFILING>
                $<$<BOOL:${WITH_HIP_PROFILING}>:__HIP_PROFILING>)
    endif ()

    # Create the final library combining common + arch-specific code
    add_library(dbcsr_${GPU_ARCH} $<TARGET_OBJECTS:dbcsr_common>
                                  $<TARGET_OBJECTS:dbcsr_arch_${GPU_ARCH}>)

    # Set library properties
    set_target_properties(
      dbcsr_${GPU_ARCH}
      PROPERTIES VERSION ${dbcsr_VERSION}
                 SOVERSION ${dbcsr_APIVERSION}
                 POSITION_INDEPENDENT_CODE ON
                 OUTPUT_NAME "dbcsr_${GPU_ARCH}"
                 EXPORT_NAME "dbcsr_${GPU_ARCH}")

    set_target_properties(dbcsr_${GPU_ARCH} PROPERTIES LINKER_LANGUAGE Fortran)

    message(
      STATUS
        "Created library target: dbcsr_${GPU_ARCH} for architecture ${GPU_ARCH} (${ARCH_NUM})"
    )
  endforeach ()

  # Create a convenience target that builds all GPU variants
  add_custom_target(dbcsr_all_gpus)
  foreach (GPU_ARCH IN LISTS WITH_GPU_LIST)
    add_dependencies(dbcsr_all_gpus dbcsr_${GPU_ARCH})
  endforeach ()

  # Set the first GPU architecture as the default "dbcsr" target for
  # compatibility
  list(GET WITH_GPU_LIST 0 FIRST_GPU_ARCH)
  add_library(dbcsr ALIAS dbcsr_${FIRST_GPU_ARCH})
  message(STATUS "Default 'dbcsr' target aliased to: dbcsr_${FIRST_GPU_ARCH}")

else ()
  # Single architecture build (existing logic)
  add_library(dbcsr ${DBCSR_SRCS})

  # Set properties for single arch build
  set_target_properties(
    dbcsr
    PROPERTIES VERSION ${dbcsr_VERSION}
               SOVERSION ${dbcsr_APIVERSION}
               POSITION_INDEPENDENT_CODE ON)

  if (USE_ACCEL MATCHES "hip")
    set_target_properties(dbcsr PROPERTIES HIP_ARCHITECTURES
                                           "${ACC_ARCH_NUMBER}")
  elseif (USE_ACCEL MATCHES "cuda")
    set_target_properties(dbcsr PROPERTIES CUDA_ARCHITECTURES
                                           "${ACC_ARCH_NUMBER}")
  endif ()

  set_target_properties(dbcsr PROPERTIES LINKER_LANGUAGE Fortran)
endif ()

# =================================================================================================
# APPLY COMMON SETTINGS TO ALL TARGETS

# Function to apply common settings to a target
function (apply_common_dbcsr_settings target_name)
  # Apply all the existing settings...
  if (USE_SMM MATCHES "libxsmm" OR (USE_SMM MATCHES "auto" AND LIBXSMM_FOUND))
    target_compile_definitions(${target_name} PRIVATE __LIBXSMM)
    target_link_directories(${target_name} PUBLIC ${LIBXSMM_LIBRARY_DIRS})
    if (USE_OPENMP)
      target_link_libraries(${target_name} PRIVATE PkgConfig::LIBXSMMEXT)
    endif ()
    target_link_libraries(${target_name} PRIVATE PkgConfig::LIBXSMM)
    target_link_libraries(${target_name} PRIVATE ${BLAS_LIBRARIES})
  endif ()

  if (BLAS_LIBRARIES MATCHES "mkl_")
    target_compile_definitions(${target_name} PRIVATE __MKL)
  endif ()

  if (APPLE)
    target_compile_definitions(${target_name} PRIVATE __NO_STATM_ACCESS)
    if (BLAS_LIBRARIES MATCHES "Accelerate")
      target_compile_definitions(${target_name} PRIVATE __ACCELERATE)
    endif ()
  endif ()

  target_compile_definitions(${target_name} PRIVATE $<$<CONFIG:Release>:NDEBUG>)
  target_link_libraries(${target_name} PRIVATE ${BLAS_LIBRARIES}
                                               ${LAPACK_LIBRARIES})
  target_include_directories(${target_name} PRIVATE base)
  target_include_directories(
    ${target_name}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_definitions(${target_name} PRIVATE __STATM_TOTAL)

  if (MPI_FOUND)
    target_compile_definitions(${target_name} PRIVATE __parallel)
    if (USE_MPI_F08)
      target_compile_definitions(${target_name} PRIVATE __USE_MPI_F08)
    endif ()
    target_link_libraries(${target_name} PUBLIC MPI::MPI_Fortran)
  endif ()

  target_link_libraries(
    ${target_name}
    PRIVATE $<$<BOOL:${USE_OPENMP}>:OpenMP::OpenMP_C>
            $<$<BOOL:${USE_OPENMP}>:OpenMP::OpenMP_CXX>
            $<$<BOOL:${USE_OPENMP}>:OpenMP::OpenMP_Fortran>)

  if (USE_ACCEL)
    # For multi-arch, we already set ARCH_NUMBER per target, for single-arch use
    # existing logic
    if (NOT MULTI_ARCH)
      target_compile_definitions(
        ${target_name}
        PRIVATE __DBCSR_ACC
                $<$<STREQUAL:${USE_ACCEL},cuda>:__CUDA>
                $<$<STREQUAL:${USE_ACCEL},opencl>:__OPENCL>
                $<$<STREQUAL:${USE_ACCEL},cuda>:ARCH_NUMBER=${ACC_ARCH_NUMBER}>
                $<$<STREQUAL:${USE_ACCEL},hip>:__HIP>
                $<$<STREQUAL:${USE_ACCEL},hip>:ARCH_NUMBER=${ACC_ARCH_NUMBER}>
                $<$<BOOL:${WITH_CUDA_PROFILING}>:__CUDA_PROFILING>
                $<$<BOOL:${WITH_HIP_PROFILING}>:__HIP_PROFILING>)
    else ()
      # For multi-arch common target, set basic acceleration flags
      target_compile_definitions(
        ${target_name}
        PRIVATE __DBCSR_ACC
                $<$<STREQUAL:${USE_ACCEL},cuda>:__CUDA>
                $<$<STREQUAL:${USE_ACCEL},opencl>:__OPENCL>
                $<$<STREQUAL:${USE_ACCEL},hip>:__HIP>
                $<$<BOOL:${WITH_CUDA_PROFILING}>:__CUDA_PROFILING>
                $<$<BOOL:${WITH_HIP_PROFILING}>:__HIP_PROFILING>)
    endif ()

    target_link_libraries(
      ${target_name}
      PRIVATE $<$<STREQUAL:${USE_ACCEL},cuda>:CUDA::cudart>
              $<$<STREQUAL:${USE_ACCEL},cuda>:CUDA::cuda_driver>
              $<$<STREQUAL:${USE_ACCEL},cuda>:CUDA::cublas>
              $<$<STREQUAL:${USE_ACCEL},cuda>:CUDA::nvrtc>
              $<$<BOOL:${WITH_CUDA_PROFILING}>:CUDA::nvToolsExt>
              $<$<STREQUAL:${USE_ACCEL},hip>:roc::hipblas>
              $<$<STREQUAL:${USE_ACCEL},hip>:hiprtc>
              $<$<STREQUAL:${USE_ACCEL},hip>:hip::host>
              $<$<BOOL:${WITH_HIP_PROFILING}>:roctx64>
              $<$<BOOL:${WITH_HIP_PROFILING}>:roctracer64>
              $<$<STREQUAL:${USE_ACCEL},opencl>:OpenCL::OpenCL>)
  endif ()
endfunction ()

# Apply settings to all targets
if (MULTI_ARCH AND MULTI_GPU_BUILD)
  # Apply to common target
  apply_common_dbcsr_settings(dbcsr_common)
  # Apply to all GPU-specific targets
  foreach (GPU_ARCH IN LISTS WITH_GPU_LIST)
    apply_common_dbcsr_settings(dbcsr_${GPU_ARCH})
  endforeach ()
else ()
  # Apply to single target
  apply_common_dbcsr_settings(dbcsr)
endif ()

# =================================================================================================
# DBCSR's C API
if (WITH_C_API)
  # Build the C API as a separate library
  add_fypp_sources(DBCSR_C_SRCS dbcsr.h dbcsr_api_c.F
                   tensors/dbcsr_tensor_api_c.F tensors/dbcsr_tensor.h)

  add_library(dbcsr_c ${DBCSR_C_SRCS})
  set_target_properties(dbcsr_c PROPERTIES LINKER_LANGUAGE Fortran)

  set_target_properties(
    dbcsr_c
    PROPERTIES VERSION ${dbcsr_VERSION}
               SOVERSION ${dbcsr_APIVERSION}
               POSITION_INDEPENDENT_CODE ON)

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
if (MULTI_ARCH AND MULTI_GPU_BUILD)
  # Install all GPU-specific libraries
  foreach (GPU_ARCH IN LISTS WITH_GPU_LIST)
    install(
      TARGETS dbcsr_${GPU_ARCH}
      EXPORT DBCSRTargets
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endforeach ()
  message(
    STATUS
      "Multi-arch install: Installing libraries for all GPU architectures: ${WITH_GPU_LIST}"
  )
else ()
  # Single architecture install (existing logic)
  install(
    TARGETS dbcsr
    EXPORT DBCSRTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif ()
# See https://gitlab.kitware.com/cmake/cmake/-/issues/19608
# CMAKE_INSTALL_Fortran_MODULES may not be an "official" variable
if (NOT CMAKE_INSTALL_Fortran_MODULES)
  set(CMAKE_INSTALL_Fortran_MODULES "${CMAKE_INSTALL_INCLUDEDIR}")
endif ()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dbcsr_api.mod"
        DESTINATION "${CMAKE_INSTALL_Fortran_MODULES}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dbcsr_tensor_api.mod"
        DESTINATION "${CMAKE_INSTALL_Fortran_MODULES}")

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

if (NOT USE_ACCEL MATCHES "none")
  foreach (FILE ${DBCSR_ACC_HEADER})
    cmake_path(GET FILE PARENT_PATH SUBDIR)
    install(FILES "${FILE}" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${SUBDIR}")
  endforeach ()
endif ()

if (USE_ACCEL MATCHES "opencl")
  install(PROGRAMS "${DBCSR_OPENCL_SCRIPT}"
          DESTINATION "${CMAKE_INSTALL_DATADIR}/opencl")
  foreach (FILE ${DBCSR_OPENCL_COMMON})
    cmake_path(GET FILE PARENT_PATH SUBDIR)
    install(FILES "${FILE}" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${SUBDIR}")
  endforeach ()
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
