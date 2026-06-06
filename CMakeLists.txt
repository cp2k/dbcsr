cmake_minimum_required(VERSION 3.22)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE FORCE)

# include our cmake snippets
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# DBCSR's source directory
set(DBCSR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# =================================================================================================
# REQUIRE OUT-OF-SOURCE BUILDS
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if (EXISTS "${LOC_PATH}")
  message(
    FATAL_ERROR
      "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory."
  )
endif ()

# =================================================================================================
# PROJECT AND VERSION
include(GetGitRevisionDescription)

git_describe(GIT_DESC)

if (GIT_DESC)
  string(REGEX REPLACE "^v([0-9]+)\\..*" "\\1" VERSION_MAJOR "${GIT_DESC}")
  string(REGEX REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR
                       "${GIT_DESC}")
  string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH
                       "${GIT_DESC}")
  string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.[0-9]+(.*)" "\\1" VERSION_GIT
                       "${GIT_DESC}")

  git_local_changes(GIT_STATE)
  if ("${GIT_STATE}" STREQUAL "DIRTY")
    set(VERSION_GIT "${VERSION_GIT}-dirty")
  endif ()

  execute_process(
    COMMAND git log -1 --format=%ai
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # take only the date from the git timestamp:
  string(REGEX REPLACE "^([0-9\\-]+) .*" "\\1" VERSION_DATE
                       "${GIT_COMMIT_DATE}")
else ()
  file(STRINGS VERSION VERSION_INFO)
  foreach (line ${VERSION_INFO})
    if (${line} MATCHES "^([^#].*)=[ \t]*(.*)$")
      set(key ${CMAKE_MATCH_1})
      set(value ${CMAKE_MATCH_2})
      string(REGEX REPLACE "[ \t\n]+$" "" key "${key}")
      string(REGEX REPLACE "[ \t\n]+$" "" value "${value}")
      set(VERSION_${key} "${value}")
      continue()
    endif ()
  endforeach ()
endif ()

project(
  dbcsr
  DESCRIPTION
    "DBCSR: Distributed Block Compressed Sparse Row matrix library (https://dbcsr.cp2k.org)"
)
set(dbcsr_VERSION
    ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}${VERSION_GIT})
set(dbcsr_APIVERSION ${VERSION_MAJOR}.${VERSION_MINOR})

# =================================================================================================
# OPTIONS
include(CMakeDependentOption)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(USE_OPENMP "Build with OpenMP support" ON)
option(USE_MPI "Build with MPI support" OFF)
option(BUILD_TESTING "Build dbcsr unit tests" OFF)
option(USE_LIBXS "Use LIBXS for host-side Small Matrix Multiplication" OFF)
option(USE_LIBXSMM "Use LIBXSMM for JIT-compiled GEMM kernels" OFF)
option(WITH_CUDA_PROFILING "Enable profiling within CUDA" OFF)
option(WITH_HIP_PROFILING "Enable profiling within HIP" OFF)

cmake_dependent_option(USE_MPI_F08 "Build with the mpi_f08 module support" OFF
                       "USE_MPI" OFF) # MPI_F08 is a module included in MPI
cmake_dependent_option(
  WITH_C_API "Build the C API (ISO_C_BINDINGS)" ON "USE_MPI" OFF
)# the ISO_C_BINDINGS require MPI unconditionally
cmake_dependent_option(WITH_EXAMPLES "Build the examples" ON "USE_MPI" OFF
)# all examples require MPI

set(TEST_MPI_RANKS
    2
    CACHE STRING "Number of MPI ranks for testing")
set(TEST_OMP_THREADS
    2
    CACHE STRING "Number of OpenMP threads for testing")

set(USE_ACCEL
    ""
    CACHE STRING "Build with acceleration support (default: none)")
set_property(CACHE USE_ACCEL PROPERTY STRINGS "" opencl cuda hip)
string(TOLOWER "${USE_ACCEL}" USE_ACCEL)

set(SUPPORTED_CUDA_ARCHITECTURES
    K20X
    K40
    K80
    P100
    V100
    A100
    H100)
set(SUPPORTED_HIP_ARCHITECTURES Mi50 Mi100 Mi250 Mi300 Mi350)
set(WITH_GPU
    $<IF:$<STREQUAL:${USE_ACCEL},"opencl">,"","P100">
    CACHE
      STRING
      "Select GPU arch. and embed parameters (default: CUDA/HIP=P100, OPENCL=all)"
)
set(WITH_GPU_PARAMS "${WITH_GPU}")
set_property(CACHE WITH_GPU PROPERTY STRINGS ${SUPPORTED_CUDA_ARCHITECTURES}
                                     ${SUPPORTED_HIP_ARCHITECTURES})

# =================================================================================================
# LANGUAGES AND TESTING
enable_language(Fortran)

if ((WITH_C_API AND WITH_EXAMPLES) OR (NOT USE_ACCEL MATCHES "none"))
  enable_language(CXX)
  enable_language(C)

  if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif ()

  if (NOT DEFINED CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
  endif ()

endif ()

# =================================== OpenMP
if (USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif ()

# =================================== SMM (Small Matrix-Matrix multiplication)
# Ensure all targets (including fetched dependencies) are built with -fPIC so
# they can be linked into shared libraries.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# LIBXS provides host-side batched GEMM for all backends. LIBXSMM provides JIT
# kernels within LIBXS and is optional. Prefer already installed packages and
# standard CMake/pkg-config search paths before falling back to FetchContent.
include(FetchContent)
include(cmake/dependencies.cmake)
find_package(PkgConfig QUIET)

set(DBCSR_USE_LIBXSMM FALSE)
set(LIBXSMM_COMPILE_DEFINITIONS "")

if (USE_LIBXS)
  set(_dbcsr_libxs_pkg_names libxs)
  if (BUILD_SHARED_LIBS)
    list(PREPEND _dbcsr_libxs_pkg_names libxs-shared)
  else ()
    list(PREPEND _dbcsr_libxs_pkg_names libxs-static)
  endif ()

  if (PkgConfig_FOUND)
    foreach (_dbcsr_libxs_pkg IN LISTS _dbcsr_libxs_pkg_names)
      pkg_check_modules(DBCSR_LIBXS QUIET IMPORTED_TARGET GLOBAL
                        ${_dbcsr_libxs_pkg})
      if (DBCSR_LIBXS_FOUND)
        break()
      endif ()
    endforeach ()
  endif ()

  if (DBCSR_LIBXS_FOUND)
    set(LIBXS_FOUND TRUE)
    set(LIBXS_INCLUDE_DIRS ${DBCSR_LIBXS_INCLUDE_DIRS})
    set(LIBXS_LINK_LIBRARIES PkgConfig::DBCSR_LIBXS)
    set(LIBXS_LIBRARY TRUE)
    message(STATUS "Using prebuilt LIBXS from pkg-config")
  else ()
    find_path(
      LIBXS_INCLUDE_DIRS
      NAMES libxs.h
      PATH_SUFFIXES include)
    find_library(LIBXS_LIBRARY NAMES xs)
    if (LIBXS_INCLUDE_DIRS AND LIBXS_LIBRARY)
      set(LIBXS_FOUND TRUE)
      set(LIBXS_LINK_LIBRARIES ${LIBXS_LIBRARY})
      message(STATUS "Using prebuilt LIBXS")
    endif ()
  endif ()

  if (LIBXS_FOUND AND NOT TARGET DBCSR::LIBXS)
    add_library(DBCSR::LIBXS INTERFACE IMPORTED GLOBAL)
    target_include_directories(DBCSR::LIBXS INTERFACE "${LIBXS_INCLUDE_DIRS}")
    target_link_libraries(DBCSR::LIBXS INTERFACE ${LIBXS_LINK_LIBRARIES})
  endif ()

  if (NOT LIBXS_FOUND)
    message(STATUS "LIBXS not found locally -- downloading via FetchContent")
    FetchContent_Declare(
      libxs
      GIT_REPOSITORY ${LIBXS_GIT_REPOSITORY}
      GIT_TAG ${LIBXS_GIT_TAG})
    set(LIBXS_FORTRAN
        ON
        CACHE BOOL "" FORCE)
    set(LIBXS_SHARED
        ${BUILD_SHARED_LIBS}
        CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(libxs)
    set(LIBXS_FETCHED TRUE)
    set(LIBXS_FOUND TRUE)
    set(LIBXS_INCLUDE_DIRS "${libxs_SOURCE_DIR}/include")
    set(LIBXS_LINK_LIBRARIES libxs::libxs)
    if (NOT TARGET DBCSR::LIBXS)
      add_library(DBCSR::LIBXS INTERFACE IMPORTED GLOBAL)
      target_link_libraries(DBCSR::LIBXS INTERFACE libxs::libxs)
      target_include_directories(DBCSR::LIBXS INTERFACE "${LIBXS_INCLUDE_DIRS}")
    endif ()
  endif ()

  if (USE_LIBXSMM)
    set(_dbcsr_libxsmm_pkg_names libxsmm)
    if (BUILD_SHARED_LIBS)
      list(PREPEND _dbcsr_libxsmm_pkg_names libxsmm-shared)
    else ()
      list(PREPEND _dbcsr_libxsmm_pkg_names libxsmm-static)
    endif ()

    if (PkgConfig_FOUND)
      foreach (_dbcsr_libxsmm_pkg IN LISTS _dbcsr_libxsmm_pkg_names)
        pkg_check_modules(DBCSR_LIBXSMM QUIET IMPORTED_TARGET GLOBAL
                          ${_dbcsr_libxsmm_pkg})
        if (DBCSR_LIBXSMM_FOUND)
          break()
        endif ()
      endforeach ()
    endif ()

    if (DBCSR_LIBXSMM_FOUND)
      set(LIBXSMM_FOUND TRUE)
      set(LIBXSMM_INCLUDE_DIRS ${DBCSR_LIBXSMM_INCLUDE_DIRS})
      set(LIBXSMM_LINK_LIBRARIES PkgConfig::DBCSR_LIBXSMM)
      set(DBCSR_USE_LIBXSMM TRUE)
      message(STATUS "Using prebuilt LIBXSMM from pkg-config")
    else ()
      find_path(
        LIBXSMM_INCLUDE_DIRS
        NAMES libxsmm.h
        PATH_SUFFIXES include)
      find_library(LIBXSMM_LIBRARY NAMES xsmm)
      if (LIBXSMM_INCLUDE_DIRS AND LIBXSMM_LIBRARY)
        set(LIBXSMM_FOUND TRUE)
        set(LIBXSMM_LINK_LIBRARIES ${LIBXSMM_LIBRARY})
        set(DBCSR_USE_LIBXSMM TRUE)
        message(STATUS "Using prebuilt LIBXSMM")
      endif ()
    endif ()

    if (NOT LIBXSMM_FOUND)
      message(
        STATUS "LIBXSMM not found locally -- downloading via FetchContent")
      FetchContent_Declare(
        libxsmm
        GIT_REPOSITORY "https://github.com/libxsmm/libxsmm.git"
        GIT_TAG ${LIBXSMM_GIT_TAG})
      set(XSMM_STATIC
          ON
          CACHE BOOL "" FORCE)
      FetchContent_MakeAvailable(libxsmm)
      set(LIBXSMM_FETCHED TRUE)
      set(DBCSR_USE_LIBXSMM TRUE)
      set(LIBXSMM_INCLUDE_DIRS "${libxsmm_SOURCE_DIR}/include")
      set(LIBXSMM_COMPILE_DEFINITIONS LIBXSMM_DEFAULT_CONFIG)
      get_target_property(_xsmm_srcs xsmm SOURCES)
      list(FILTER _xsmm_srcs EXCLUDE REGEX
           "(binaryexport_generator|gemm_driver)")
      set_target_properties(xsmm PROPERTIES SOURCES "${_xsmm_srcs}")
    endif ()
  endif ()
endif ()

# =================================== BLAS & LAPACK, PkgConfig
find_package(LAPACK REQUIRED) # needed for some of the integrated test routines,
                              # also calls find_package(BLAS)

# =================================== Python this module looks preferably for
# version 3 of Python. If not found, version 2 is searched. In CMake 3.15, if a
# python virtual environment is activated, it will search the virtual
# environment for a python interpreter before searching elsewhere in the system.
# In CMake <3.15, the system is searched before the virtual environment.
if (NOT Python_EXECUTABLE)
  # If the python interpreter is not specified (command line), try finding it:
  find_package(
    Python
    COMPONENTS Interpreter
    REQUIRED)
endif ()

# =================================== MPI
if (USE_MPI)
  get_property(REQUIRED_MPI_COMPONENTS GLOBAL PROPERTY ENABLED_LANGUAGES)
  if (NOT CMAKE_CROSSCOMPILING) # when cross compiling, assume the users know
                                # what they are doing
    set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
  endif ()
  find_package(
    MPI
    COMPONENTS ${REQUIRED_MPI_COMPONENTS}
    REQUIRED)

  if (NOT MPI_Fortran_HAVE_F90_MODULE)
    message(
      FATAL_ERROR
        "\
The listed MPI implementation does not provide the required mpi.mod interface. \
When using the GNU compiler in combination with Intel MPI, please use the \
Intel MPI compiler wrappers. Check the INSTALL.md for more information.")
  endif ()
  if (USE_MPI_F08)
    if (NOT MPI_Fortran_HAVE_F08_MODULE)
      message(
        FATAL_ERROR
          "The listed MPI implementation does not provide the required mpi_f08.mod interface."
      )
    endif ()
  endif ()
  if ("${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI v2.1"
      OR "${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI v3.1")
    message(
      WARNING
        "RMA with ${MPI_Fortran_LIBRARY_VERSION_STRING} is not supported due to issues with its implementation."
        " Please use a newer version of OpenMPI or switch to MPICH if you plan on using MPI-RMA."
    )
  endif ()
endif ()

# =================================== GPU backends

if (NOT USE_ACCEL MATCHES "none")
  set(DBCSR_ACC_HEADER acc/acc.h acc/acc_bench.h acc/acc_libsmm.h)
endif ()

if (USE_ACCEL MATCHES "opencl")
  # Hint additional OpenCL locations (Intel oneAPI, CUDA toolkit)
  if (DEFINED ENV{CMPLR_ROOT})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CMPLR_ROOT}")
  endif ()
  if (DEFINED ENV{CUDA_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CUDA_PATH}")
  endif ()
  find_package(OpenCL REQUIRED)

  if (NOT USE_LIBXS OR NOT TARGET DBCSR::LIBXS)
    message(
      FATAL_ERROR
        "LIBXS not found. OpenCL backend requires LIBXS (USE_LIBXS=ON).")
  endif ()

  set(_dbcsr_libxstream_pkg_names libxstream)
  if (BUILD_SHARED_LIBS)
    list(PREPEND _dbcsr_libxstream_pkg_names libxstream-shared)
  else ()
    list(PREPEND _dbcsr_libxstream_pkg_names libxstream-static)
  endif ()

  if (PkgConfig_FOUND)
    foreach (_dbcsr_libxstream_pkg IN LISTS _dbcsr_libxstream_pkg_names)
      pkg_check_modules(DBCSR_LIBXSTREAM QUIET IMPORTED_TARGET GLOBAL
                        ${_dbcsr_libxstream_pkg})
      if (DBCSR_LIBXSTREAM_FOUND)
        break()
      endif ()
    endforeach ()
  endif ()

  if (DBCSR_LIBXSTREAM_FOUND)
    set(LIBXSTREAM_FOUND TRUE)
    set(LIBXSTREAM_INCLUDE_DIRS ${DBCSR_LIBXSTREAM_INCLUDE_DIRS})
    set(LIBXSTREAM_LINK_LIBRARIES PkgConfig::DBCSR_LIBXSTREAM)
    message(STATUS "Using prebuilt LIBXSTREAM from pkg-config")
  else ()
    find_path(
      LIBXSTREAM_INCLUDE_DIRS
      NAMES libxstream.h
      PATH_SUFFIXES include)
    find_library(LIBXSTREAM_LIBRARY NAMES xstream)
    if (LIBXSTREAM_INCLUDE_DIRS AND LIBXSTREAM_LIBRARY)
      set(LIBXSTREAM_FOUND TRUE)
      set(LIBXSTREAM_LINK_LIBRARIES ${LIBXSTREAM_LIBRARY})
      message(STATUS "Using prebuilt LIBXSTREAM")
    endif ()
  endif ()

  if (LIBXSTREAM_FOUND)
    set(_dbcsr_libxstream_prefix_hints)
    if (DBCSR_LIBXSTREAM_PREFIX)
      list(APPEND _dbcsr_libxstream_prefix_hints "${DBCSR_LIBXSTREAM_PREFIX}")
    endif ()
    foreach (_dbcsr_libxstream_include_dir IN LISTS LIBXSTREAM_INCLUDE_DIRS)
      get_filename_component(_dbcsr_libxstream_prefix
                             "${_dbcsr_libxstream_include_dir}/.." ABSOLUTE)
      list(APPEND _dbcsr_libxstream_prefix_hints "${_dbcsr_libxstream_prefix}")
    endforeach ()

    find_program(
      LIBXSTREAM_OPENCL_SCRIPT
      NAMES tool_opencl.sh
      HINTS ${_dbcsr_libxstream_prefix_hints}
      PATH_SUFFIXES scripts bin)
    find_path(
      LIBXSTREAM_SMM_DIR
      NAMES smm_acc.c
      HINTS ${_dbcsr_libxstream_prefix_hints}
      PATH_SUFFIXES samples/smm share/libxstream/samples/smm)
    if (NOT LIBXSTREAM_OPENCL_SCRIPT OR NOT LIBXSTREAM_SMM_DIR)
      message(
        STATUS
          "Prebuilt LIBXSTREAM was found, but its OpenCL helper script or SMM samples were not found -- downloading via FetchContent"
      )
      set(LIBXSTREAM_FOUND FALSE)
    elseif (NOT TARGET DBCSR::LIBXSTREAM)
      add_library(DBCSR::LIBXSTREAM INTERFACE IMPORTED GLOBAL)
      target_include_directories(DBCSR::LIBXSTREAM
                                 INTERFACE "${LIBXSTREAM_INCLUDE_DIRS}")
      target_link_libraries(DBCSR::LIBXSTREAM
                            INTERFACE ${LIBXSTREAM_LINK_LIBRARIES})
    endif ()
  else ()
    message(
      STATUS "LIBXSTREAM not found locally -- downloading via FetchContent")
    FetchContent_Declare(
      libxstream
      GIT_REPOSITORY ${LIBXSTREAM_GIT_REPOSITORY}
      GIT_TAG ${LIBXSTREAM_GIT_TAG})
    set(LIBXSTREAM_SHARED
        ${BUILD_SHARED_LIBS}
        CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(libxstream)
    set(LIBXSTREAM_FETCHED TRUE)
    set(LIBXSTREAM_FOUND TRUE)
    set(LIBXSTREAM_INCLUDE_DIRS "${libxstream_SOURCE_DIR}/include")
    set(LIBXSTREAM_LINK_LIBRARIES libxstream::libxstream)
    set(LIBXSTREAM_OPENCL_SCRIPT
        "${libxstream_SOURCE_DIR}/scripts/tool_opencl.sh")
    set(LIBXSTREAM_SMM_DIR "${libxstream_SOURCE_DIR}/samples/smm")
    if (NOT TARGET DBCSR::LIBXSTREAM)
      add_library(DBCSR::LIBXSTREAM INTERFACE IMPORTED GLOBAL)
      target_link_libraries(DBCSR::LIBXSTREAM INTERFACE libxstream::libxstream)
      target_include_directories(DBCSR::LIBXSTREAM
                                 INTERFACE "${LIBXSTREAM_INCLUDE_DIRS}")
    endif ()
  endif ()

  message(STATUS "Using LIBXSTREAM")
  set(DBCSR_OPENCL_SCRIPT "${LIBXSTREAM_OPENCL_SCRIPT}")
  set(DBCSR_LIBXSTREAM_SMM_DIR "${LIBXSTREAM_SMM_DIR}")
  list(GET LIBXSTREAM_INCLUDE_DIRS 0 DBCSR_LIBXSTREAM_INCLUDE_DIR)
endif ()

if (USE_ACCEL MATCHES "cuda|hip")
  set(GPU_ARCH_NUMBER_K20X 35)
  set(GPU_ARCH_NUMBER_K40 35)
  set(GPU_ARCH_NUMBER_K80 37)
  set(GPU_ARCH_NUMBER_P100 60)
  set(GPU_ARCH_NUMBER_V100 70)
  set(GPU_ARCH_NUMBER_A100 80)
  set(GPU_ARCH_NUMBER_H100 90)
  set(GPU_ARCH_NUMBER_Mi50 gfx906)
  set(GPU_ARCH_NUMBER_Mi100 gfx908)
  set(GPU_ARCH_NUMBER_Mi250 gfx90a)
  set(GPU_ARCH_NUMBER_Mi300 gfx942)
  set(GPU_ARCH_NUMBER_Mi350 gfx950)
endif ()

if (USE_ACCEL MATCHES "cuda")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif ()

  if (CUDAToolkit_VERSION LESS 5.5)
    message(FATAL_ERROR "CUDA version >= 5.5 is required.")
  endif ()

  # Make sure the GPU required is supported
  list(FIND SUPPORTED_CUDA_ARCHITECTURES ${WITH_GPU} GPU_SUPPORTED)
  if (GPU_SUPPORTED EQUAL -1)
    message(
      FATAL_ERROR "GPU architecture requested (${WITH_GPU}) is not supported. "
                  "Please choose from: ${SUPPORTED_CUDA_ARCHITECTURES}")
  endif ()

  # set cuda architecture number and compilation flags
  set(ACC_ARCH_NUMBER ${GPU_ARCH_NUMBER_${WITH_GPU}})

  message(STATUS "GPU target architecture: " ${WITH_GPU})
  message(STATUS "Kernel parameters: " ${WITH_GPU_PARAMS})
  message(STATUS "GPU architecture number: " ${ACC_ARCH_NUMBER})
  message(STATUS "GPU profiling enabled: " ${WITH_CUDA_PROFILING})
endif ()

if (USE_ACCEL MATCHES "hip")
  if (NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES OFF)
  endif ()
  enable_language(HIP)

  if (NOT DEFINED CMAKE_HIP_STANDARD)
    set(CMAKE_HIP_STANDARD 14)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
  endif ()

  # Make sure the GPU required is supported
  list(FIND SUPPORTED_HIP_ARCHITECTURES ${WITH_GPU} GPU_SUPPORTED)
  if (GPU_SUPPORTED EQUAL -1)
    message(
      FATAL_ERROR "GPU architecture requested (${WITH_GPU}) is not supported. "
                  "Please choose from: ${SUPPORTED_HIP_ARCHITECTURES}")
  endif ()

  # ROCm is typically installed in /opt/rocm; otherwise let the user set
  # ROCM_PATH as an environment variable or define.
  if (NOT DEFINED ROCM_PATH)
    if (NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH
          "/opt/rocm"
          CACHE PATH "Path to ROCm installation")
    else ()
      set(ROCM_PATH
          $ENV{ROCM_PATH}
          CACHE PATH "Path to ROCm installation")
    endif ()
  endif ()

  # Notice: this is not FindHIP.cmake for hip language support, but
  # hip-config.cmake which contains targets like hip::host for jitting.
  find_package(hip CONFIG REQUIRED HINTS ${ROCM_PATH})

  message(STATUS "Build with HIP ${hip_VERSION}")
  if (hip_VERSION LESS 4.4.0)
    message(FATAL_ERROR "HIP version >= 4.4.0 is required.")
  endif ()

  set(ACC_ARCH_NUMBER ${GPU_ARCH_NUMBER_${WITH_GPU}})
  message(STATUS "GPU target architecture: " ${WITH_GPU})
  message(STATUS "Kernel parameters: " ${WITH_GPU_PARAMS})
  message(STATUS "GPU architecture number: " ${ACC_ARCH_NUMBER})
  message(STATUS "GPU profiling enabled: " ${WITH_HIP_PROFILING})

  # =================================== BLAS on GPU backend
  find_package(hipblas CONFIG REQUIRED HINTS ${ROCM_PATH})

  # =================================== HIPRTC
  find_package(hiprtc REQUIRED)
endif ()

# =================================================================================================
# OPTION HANDLING

# make sure that the default build type is RELEASE
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS
      "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${default_build_type}"
      CACHE STRING
            "Choose the type of build, options are: Debug Release Coverage."
            FORCE)
  # set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "Coverage")
endif ()

# compiler configuration could have impacted package discovery (above)
include(CompilerConfiguration)
include(CheckCompilerSupport)

# subdirectories
add_subdirectory(${DBCSR_SOURCE_DIR})

if (BUILD_TESTING)
  include(CTest)
  add_subdirectory(tests)
endif ()

if (WITH_EXAMPLES)
  add_subdirectory(examples)
endif ()

add_subdirectory(docs)

include(CustomTargets)

# Disable LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE FORCE)
