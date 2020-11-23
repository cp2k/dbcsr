# =================================================================================================
# INCLUDE
include(fypp-sources)
include(GNUInstallDirs) # required to get a proper LIBDIR variable
include(CMakePackageConfigHelpers)

# =================================================================================================
# SOURCE FILE LISTS
add_fypp_sources(
  DBCSR_SRCS
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

set(DBCSR_CUDA_SRCS
    acc/cuda/acc_cublas.cu
    acc/cuda/acc_cuda.cpp
    acc/cuda/acc_dev.cpp
    acc/cuda/acc_error.cpp
    acc/cuda/acc_event.cpp
    acc/cuda/acc_init.cpp
    acc/cuda/acc_mem.cpp
    acc/cuda/acc_stream.cpp
    acc/cuda/dbcsr_cuda_nvtx_cu.cu)

set(DBCSR_HIP_SRCS
    acc/hip/acc_hip.cpp
    acc/cuda/acc_cublas.cu
    acc/cuda/acc_dev.cpp
    acc/cuda/acc_error.cpp
    acc/cuda/acc_event.cpp
    acc/cuda/acc_init.cpp
    acc/cuda/acc_mem.cpp
    acc/cuda/acc_stream.cpp)

# set the __SHORT_FILE__ per file for dbcsr sources
foreach (dbcsr_src ${DBCSR_SRCS})
  # add_fypp_sources returns a path in the current binary dir
  get_filename_component(short_file "${dbcsr_src}" NAME)
  set_source_files_properties(
    ${dbcsr_src} PROPERTIES COMPILE_DEFINITIONS __SHORT_FILE__="${short_file}")
endforeach ()

# =================================================================================================
# DBCSR LIBRARY
add_library(dbcsr ${DBCSR_SRCS})

set_target_properties(dbcsr PROPERTIES VERSION ${dbcsr_VERSION}
                                       SOVERSION ${dbcsr_APIVERSION})

if (TARGET PkgConfig::deps)
  target_link_libraries(dbcsr PRIVATE PkgConfig::deps)
endif ()

if (USE_SMM MATCHES "libxsmm")
  # linker/include flags are managed by pkg-config (above)
  target_compile_definitions(dbcsr PRIVATE __LIBXSMM)
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
target_compile_definitions(dbcsr PRIVATE "$<$<CONFIG:Release>:NDEBUG>")

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

# =================================================================================================
# Link OpenMP runtime library even if DBCSR main code is not built with OpenMP

if (OpenMP_FOUND)
  target_link_libraries(dbcsr PRIVATE OpenMP::OpenMP_Fortran)
endif ()

# =================================================================================================
# DBCSR LIBRARY's CUDA BACKEND

# adapted from
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html: replaces
# in the Fortran and CXX targets the interface compile options by a more
# differentiated version for Fortran, CXX and CUDA, where the CXX and CUDA
# options are taken from the CXX target
function (CUDA_CONVERT_FLAGS EXISTING_TARGET)
  get_property(
    old_fflags
    TARGET ${EXISTING_TARGET}_Fortran
    PROPERTY INTERFACE_COMPILE_OPTIONS)
  get_property(
    old_cxxflags
    TARGET ${EXISTING_TARGET}_CXX
    PROPERTY INTERFACE_COMPILE_OPTIONS)

  string(REPLACE ";" "," CUDA_flags "${old_cxxflags}")
  if (NOT "${CUDA_flags}" STREQUAL "")
    set(CUDA_flags "-Xcompiler=${CUDA_flags}")
  endif ()

  set_property(
    TARGET ${EXISTING_TARGET}_Fortran
    PROPERTY
      INTERFACE_COMPILE_OPTIONS
      "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:Fortran>>:${old_fflags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_cxxflags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:${CUDA_flags}>"
  )
  set_property(
    TARGET ${EXISTING_TARGET}_CXX
    PROPERTY
      INTERFACE_COMPILE_OPTIONS
      "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:Fortran>>:${old_fflags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_cxxflags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:${CUDA_flags}>"
  )
endfunction ()

if (USE_CUDA)
  if (${CMAKE_VERSION} VERSION_LESS 3.16)
    # workaround for CUDA support with CMake <3.16, see also see
    # https://gitlab.kitware.com/cmake/cmake/issues/17929 and
    # https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
    if (OpenMP_FOUND)
      cuda_convert_flags(OpenMP::OpenMP)
    endif ()
    if (MPI_FOUND)
      cuda_convert_flags(MPI::MPI)
    endif ()
  endif ()

  target_link_libraries(dbcsr PUBLIC cuda)

  # add libsmm_acc which has its own CMakeLists.txt due to code generation
  add_subdirectory(acc/libsmm_acc)
  target_sources(dbcsr PRIVATE $<TARGET_OBJECTS:libsmm_acc>)
  target_link_libraries(dbcsr PRIVATE nvrtc)

  # Complete list of GPU-support sources
  set(DBCSR_ACC_SRCS ${DBCSR_CUDA_SRCS})

  # Make an object library
  add_library(acc OBJECT ${DBCSR_ACC_SRCS})
  target_compile_definitions(acc PRIVATE __CUDA)
  target_compile_definitions(
    acc PRIVATE $<$<BOOL:${WITH_CUDA_PROFILING}>:__CUDA_PROFILING>)
  target_include_directories(acc
                             PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_sources(dbcsr PRIVATE $<TARGET_OBJECTS:acc>)
  target_compile_definitions(acc PRIVATE __DBCSR_ACC)
  target_link_libraries(acc PRIVATE ${CUBLAS} ${CUDA_NVTOOLSEXT})

  # tests need some of the libsmm_acc headers and for CMake <3.12 compatibility,
  # we can't set it on the the object library itself
  target_include_directories(
    dbcsr PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/acc/libsmm_acc>)

  target_compile_definitions(dbcsr PRIVATE __DBCSR_ACC)
  target_link_libraries(dbcsr PRIVATE ${CUBLAS} ${CUDA_NVTOOLSEXT})
  target_compile_definitions(dbcsr PRIVATE __CUDA)
  # __CUDA_PROFILING also affects core/ parts like the timings
  target_compile_definitions(
    dbcsr PRIVATE $<$<BOOL:${WITH_CUDA_PROFILING}>:__CUDA_PROFILING>)

  if (APPLE)
    # Help the static cuda runtime find the driver (libcuda.dyllib) at runtime.
    set_property(TARGET dbcsr PROPERTY BUILD_RPATH
                                       ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif ()
endif ()

# =================================================================================================
# DBCSR LIBRARY's HIP BACKEND

if (USE_HIP)

  if (USE_OPENMP)
    set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif ()

  # add libsmm_acc which has its own CMakeLists.txt due to code generation
  add_subdirectory(acc/libsmm_acc)
  target_link_libraries(dbcsr PUBLIC libsmm_acc)
  target_link_libraries(dbcsr PUBLIC ${ROCM_HIPRTC_LIB})

  # Complete list of GPU-support sources
  set(DBCSR_ACC_SRCS ${DBCSR_HIP_SRCS})

  # Compile the rest of the HIP source files into a static library
  set_source_files_properties(${DBCSR_ACC_SRCS}
                              PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  hip_add_library(acc STATIC ${DBCSR_ACC_SRCS})
  target_include_directories(acc PRIVATE ${HIP_PATH}/../include)
  target_compile_definitions(acc PRIVATE ARCH_NUMBER=${ACC_ARCH_NUMBER})
  target_compile_definitions(acc PRIVATE __HIP)
  target_compile_options(acc PRIVATE "-fPIC")
  target_link_libraries(acc PUBLIC ${HIPBLAS})
  target_link_libraries(dbcsr PUBLIC acc)
  install(
    TARGETS acc
    EXPORT accTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

  # tests need some of the libsmm_acc headers and for CMake <3.12 compatibility,
  # we can't set it on the the object library itself
  target_include_directories(
    dbcsr PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/acc>)
  target_include_directories(
    dbcsr PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/acc/libsmm_acc>)

  target_compile_definitions(dbcsr PRIVATE __DBCSR_ACC)
  target_compile_definitions(dbcsr PRIVATE __HIP)

endif ()

# =================================================================================================
# DBCSR's C API
if (WITH_C_API)
  # Build the C API as a separate library
  add_fypp_sources(DBCSR_C_SRCS dbcsr_api_c.F tensors/dbcsr_tensor_api_c.F
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
# FORD - DOCUMENTATION GENERATION
find_program(
  FORD_EXE ford
  DOC "path to the ford executable (required to generate the documentation)")

# Copy the FORD project-file into the build directory
set(FORD_PROJECT_FILE "${CMAKE_BINARY_DIR}/DBCSR.md")
configure_file(${CMAKE_SOURCE_DIR}/DBCSR.md "${FORD_PROJECT_FILE}")

# Copy the FORD project-file into the build directory
add_custom_target(
  doc
  COMMENT "Generating API documentation and doc pages"
  COMMAND "${FORD_EXE}" "${FORD_PROJECT_FILE}"
  VERBATIM)
add_dependencies(doc doc_copy_tests)
if (WITH_C_API)
  add_dependencies(doc doc_copy_examples)
endif ()
add_dependencies(doc fypp) # only depend on the fypp step to avoid building
                           # everything just for the docs

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
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/dbcsr.h"
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
if (USE_HIP)
  install(
    EXPORT libsmm_accTargets
    NAMESPACE "${config_namespace}"
    DESTINATION "${config_install_dir}")
  install(
    EXPORT accTargets
    NAMESPACE "${config_namespace}"
    DESTINATION "${config_install_dir}")
endif ()
install(
  EXPORT DBCSRTargets
  NAMESPACE "${config_namespace}"
  DESTINATION "${config_install_dir}")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/DBCSRConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/DBCSRConfigVersion.cmake
        DESTINATION ${config_install_dir})
