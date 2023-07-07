cmake_minimum_required(VERSION 3.22)

# include our cmake snippets
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

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
option(USE_MPI "Build with MPI support" ON)
option(USE_MPI_F08 "Build with the mpi_f08 module support" OFF)
cmake_dependent_option(
  WITH_C_API "Build the C API (ISO_C_BINDINGS)" ON "USE_MPI" OFF
)# the ISO_C_BINDINGS require MPI unconditionally
cmake_dependent_option(WITH_EXAMPLES "Build the examples" ON "USE_MPI" OFF
)# all examples require MPI

set(TEST_MPI_RANKS
    "auto"
    CACHE STRING "Number of MPI ranks for testing")
set(TEST_OMP_THREADS
    2
    CACHE STRING "Number of OpenMP threads for testing")

set(USE_SMM
    "blas"
    CACHE STRING
          "Small Matrix Multiplication implementation to use (default: blas)")
set_property(CACHE USE_SMM PROPERTY STRINGS blas libxsmm)

set(USE_ACCEL
    ""
    CACHE STRING "Build with acceleration support (default: none)")
set_property(CACHE USE_ACCEL PROPERTY STRINGS "" opencl cuda hip)

set(SUPPORTED_CUDA_ARCHITECTURES K20X K40 K80 P100 V100 A100)
set(SUPPORTED_HIP_ARCHITECTURES Mi50 Mi100 Mi250)
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

if (WITH_C_API AND WITH_EXAMPLES)
  enable_language(CXX)
  enable_language(C)
endif ()

# always use at least C++11
set(CMAKE_CXX_STANDARD 11)

# =================================== OpenMP
if (USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif ()

if ((USE_ACCEL MATCHES "opencl") AND (NOT USE_SMM MATCHES "libxsmm"))
  message(FATAL_ERROR "OpenCL requires USE_SMM=libxsmm")
endif ()

# =================================== SMM (Small Matrix-Matrix multiplication)
if (USE_SMM MATCHES "blas")
  message(STATUS "Using BLAS for Small Matrix Multiplication")
elseif (USE_SMM MATCHES "libxsmm")
  message(STATUS "Using libxsmm for Small Matrix Multiplication")
else ()
  message(FATAL_ERROR "Unknown SMM library specified")
endif ()

# =================================== LIBXSMM (rely on pkg-config)
if (USE_SMM MATCHES "libxsmm")
  find_package(PkgConfig REQUIRED)
  if (USE_OPENMP)
    if (NOT USE_SMM MATCHES "libxsmm-shared")
      pkg_check_modules(LIBXSMMEXT IMPORTED_TARGET GLOBAL libxsmmext-static)
    endif ()
    if (NOT LIBXSMMEXT_FOUND)
      pkg_check_modules(LIBXSMMEXT REQUIRED IMPORTED_TARGET GLOBAL libxsmmext)
    endif ()
  endif ()
  if (NOT USE_SMM MATCHES "libxsmm-shared")
    pkg_check_modules(LIBXSMM IMPORTED_TARGET GLOBAL libxsmmf-static)
  endif ()
  if (NOT LIBXSMM_FOUND)
    pkg_check_modules(LIBXSMM REQUIRED IMPORTED_TARGET GLOBAL libxsmmf)
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
if (USE_ACCEL MATCHES "opencl")
  find_package(OpenCL REQUIRED)
  enable_language(C)
endif ()

if (USE_ACCEL MATCHES "cuda|hip")
  enable_language(CXX)
  set(GPU_ARCH_NUMBER_K20X 35)
  set(GPU_ARCH_NUMBER_K40 35)
  set(GPU_ARCH_NUMBER_K80 37)
  set(GPU_ARCH_NUMBER_P100 60)
  set(GPU_ARCH_NUMBER_V100 70)
  set(GPU_ARCH_NUMBER_A100 80)
  set(GPU_ARCH_NUMBER_Mi50 gfx906)
  set(GPU_ARCH_NUMBER_Mi100 gfx908)
  set(GPU_ARCH_NUMBER_Mi250 gfx90a)
endif ()

if (USE_ACCEL MATCHES "cuda")
  find_package(CUDAToolkit REQUIRED)

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
add_subdirectory(src)

include(CTest)
if (BUILD_TESTING)
  add_subdirectory(tests)
endif ()

if (WITH_EXAMPLES)
  add_subdirectory(examples)
endif ()

add_subdirectory(docs)

include(CustomTargets)
