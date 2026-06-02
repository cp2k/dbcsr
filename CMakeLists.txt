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

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(USE_OPENMP "Build with OpenMP support" ON)
option(USE_MPI "Build with MPI support" ON)
option(USE_MPI_F08 "Build with the mpi_f08 module support" OFF)
option(BUILD_TESTING "Build dbcsr unit tests" OFF)
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

option(WITH_CUDA_PROFILING "Enable profiling within CUDA" OFF)
option(WITH_HIP_PROFILING "Enable profiling within HIP" OFF)

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
# kernels within LIBXS (optional, auto-detected). Set LIBXSROOT to use a
# prebuilt installation, or let FetchContent download it. Set USE_LIBXS=OFF to
# disable entirely.
include(FetchContent)
include(cmake/dependencies.cmake)
option(USE_LIBXS "Use LIBXS for host-side Small Matrix Multiplication" ON)
find_package(PkgConfig QUIET)
set(LIBXSROOT
    ""
    CACHE PATH "Root directory of a prebuilt LIBXS (skips FetchContent)")
if (USE_LIBXS)
  # Try prebuilt: pkg-config > explicit root > path probing
  if (NOT LIBXSROOT AND PkgConfig_FOUND)
    pkg_check_modules(_LIBXS QUIET libxs)
    if (_LIBXS_FOUND)
      pkg_get_variable(LIBXSROOT libxs prefix)
    endif ()
  endif ()
  if (NOT LIBXSROOT)
    foreach (_dir "${CMAKE_SOURCE_DIR}/../libxs" "$ENV{HOME}/libxs"
                  "/opt/libxs")
      if (EXISTS "${_dir}/include/libxs.f")
        set(LIBXSROOT "${_dir}")
        break()
      endif ()
    endforeach ()
  endif ()
  if (LIBXSROOT AND EXISTS "${LIBXSROOT}/include/libxs.f")
    find_library(
      LIBXS_LIBRARY
      NAMES xs
      PATHS "${LIBXSROOT}/lib"
      NO_DEFAULT_PATH)
    if (LIBXS_LIBRARY)
      message(STATUS "Using prebuilt LIBXS from ${LIBXSROOT}")
    else ()
      message(
        STATUS "LIBXS source found at ${LIBXSROOT} -- building as subdirectory")
      set(LIBXS_FORTRAN
          ON
          CACHE BOOL "" FORCE)
      set(LIBXS_SHARED
          ${BUILD_SHARED_LIBS}
          CACHE BOOL "" FORCE)
      add_subdirectory("${LIBXSROOT}" "${CMAKE_BINARY_DIR}/_deps/libxs-build")
      set(LIBXS_FETCHED TRUE)
    endif ()
  else ()
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
    set(LIBXSROOT "${libxs_SOURCE_DIR}")
  endif ()
  # Optional LIBXSMM for JIT-compiled GEMM kernels
  option(USE_LIBXSMM "Use LIBXSMM for JIT-compiled GEMM kernels" ON)
  if (USE_LIBXSMM)
    set(LIBXSMMROOT
        ""
        CACHE PATH "Root directory of LIBXSMM (optional)")
    if (NOT LIBXSMMROOT)
      foreach (_dir "${LIBXSROOT}/../libxsmm" "$ENV{HOME}/libxsmm")
        if (EXISTS "${_dir}/include/libxsmm.h")
          set(LIBXSMMROOT "${_dir}")
          break()
        endif ()
      endforeach ()
    endif ()
    if (LIBXSMMROOT AND EXISTS "${LIBXSMMROOT}/include/libxsmm.h")
      find_library(
        LIBXSMM_LIBRARY
        NAMES xsmm
        PATHS "${LIBXSMMROOT}/lib"
        NO_DEFAULT_PATH)
      if (LIBXSMM_LIBRARY)
        message(STATUS "Using prebuilt LIBXSMM from ${LIBXSMMROOT}")
      else ()
        message(
          STATUS
            "LIBXSMM source found at ${LIBXSMMROOT} -- building as subdirectory"
        )
        set(XSMM_STATIC
            ON
            CACHE BOOL "" FORCE)
        add_subdirectory("${LIBXSMMROOT}"
                         "${CMAKE_BINARY_DIR}/_deps/libxsmm-build")
        set(LIBXSMM_FETCHED TRUE)
        get_target_property(_xsmm_srcs xsmm SOURCES)
        list(FILTER _xsmm_srcs EXCLUDE REGEX "binaryexport_generator")
        set_target_properties(xsmm PROPERTIES SOURCES "${_xsmm_srcs}")
      endif ()
    else ()
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
      set(LIBXSMMROOT "${libxsmm_SOURCE_DIR}")
      set(LIBXSMM_FETCHED TRUE)
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
  # LIBXS is required for the OpenCL backend
  if (NOT LIBXS_LIBRARY AND NOT LIBXS_FETCHED)
    message(
      FATAL_ERROR
        "LIBXS not found. OpenCL backend requires LIBXS (USE_LIBXS=ON).")
  endif ()
  # Locate LIBXSTREAM: prebuilt > pkg-config > path probing > FetchContent
  set(LIBXSTREAMROOT
      ""
      CACHE PATH "Root directory of a prebuilt LIBXSTREAM (skips FetchContent)")
  if (NOT LIBXSTREAMROOT AND PkgConfig_FOUND)
    pkg_check_modules(_LIBXSTREAM QUIET libxstream)
    if (_LIBXSTREAM_FOUND)
      pkg_get_variable(LIBXSTREAMROOT libxstream prefix)
    endif ()
  endif ()
  if (NOT LIBXSTREAMROOT)
    foreach (_dir "${CMAKE_SOURCE_DIR}/../libxstream" "$ENV{HOME}/libxstream"
                  "/opt/libxstream")
      if (EXISTS "${_dir}/include/libxstream.h")
        set(LIBXSTREAMROOT "${_dir}")
        break()
      endif ()
    endforeach ()
  endif ()
  if (LIBXSTREAMROOT AND EXISTS "${LIBXSTREAMROOT}/include/libxstream.h")
    find_library(
      LIBXSTREAM_LIBRARY xstream
      PATHS "${LIBXSTREAMROOT}/lib"
      NO_DEFAULT_PATH)
    if (LIBXSTREAM_LIBRARY)
      message(STATUS "Using prebuilt LIBXSTREAM from ${LIBXSTREAMROOT}")
    else ()
      message(
        STATUS
          "LIBXSTREAM source found at ${LIBXSTREAMROOT} -- building as subdirectory"
      )
      set(LIBXSROOT
          "${LIBXSROOT}"
          CACHE PATH "" FORCE)
      set(LIBXSTREAM_SHARED
          ${BUILD_SHARED_LIBS}
          CACHE BOOL "" FORCE)
      add_subdirectory("${LIBXSTREAMROOT}"
                       "${CMAKE_BINARY_DIR}/_deps/libxstream-build")
      set(LIBXSTREAM_FETCHED TRUE)
    endif ()
  else ()
    message(
      STATUS "LIBXSTREAM not found locally -- downloading via FetchContent")
    FetchContent_Declare(
      libxstream
      GIT_REPOSITORY ${LIBXSTREAM_GIT_REPOSITORY}
      GIT_TAG ${LIBXSTREAM_GIT_TAG})
    set(LIBXSROOT
        "${LIBXSROOT}"
        CACHE PATH "" FORCE)
    set(LIBXSTREAM_SHARED
        ${BUILD_SHARED_LIBS}
        CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(libxstream)
    set(LIBXSTREAMROOT "${libxstream_SOURCE_DIR}")
    set(LIBXSTREAM_FETCHED TRUE)
  endif ()
  message(STATUS "Using LIBXSTREAM from ${LIBXSTREAMROOT}")
  # Kernel build script and common OpenCL headers
  set(DBCSR_OPENCL_SCRIPT "${LIBXSTREAMROOT}/scripts/tool_opencl.sh")
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
