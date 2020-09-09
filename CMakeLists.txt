cmake_minimum_required (VERSION 3.12)

# include our cmake snippets
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# =================================================================================================
# REQUIRE OUT-OF-SOURCE BUILDS
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory.")
endif()

# =================================================================================================
# PROJECT AND VERSION
INCLUDE(GetGitRevisionDescription)

git_describe(GIT_DESC)

if (GIT_DESC)
  string(REGEX REPLACE "^v([0-9]+)\\..*" "\\1" VERSION_MAJOR "${GIT_DESC}")
  string(REGEX REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR "${GIT_DESC}")
  string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH "${GIT_DESC}")
  string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.[0-9]+(.*)" "\\1" VERSION_GIT "${GIT_DESC}")

  git_local_changes(GIT_STATE)
  if ("${GIT_STATE}" STREQUAL "DIRTY")
    set(VERSION_GIT "${VERSION_GIT}-dirty")
  endif ()

  execute_process(COMMAND git log -1 --format=%ai
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                  OUTPUT_VARIABLE GIT_COMMIT_DATE
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  # take only the date from the git timestamp:
  string(REGEX REPLACE "^([0-9\\-]+) .*" "\\1" VERSION_DATE "${GIT_COMMIT_DATE}")
else ()
  file(STRINGS VERSION VERSION_INFO)
  foreach(line ${VERSION_INFO})
    if (${line} MATCHES "^([^#].*)=[ \t]*(.*)$")
      set(key ${CMAKE_MATCH_1})
      set(value ${CMAKE_MATCH_2})
      string(REGEX REPLACE "[ \t\n]+$" "" key "${key}")
      string(REGEX REPLACE "[ \t\n]+$" "" value "${value}")
      set(VERSION_${key} "${value}")
      continue ()
    endif ()
  endforeach()
endif ()

project(dbcsr DESCRIPTION "DBCSR: Distributed Block Compressed Sparse Row matrix library (https://dbcsr.cp2k.org)")
set(dbcsr_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}${VERSION_GIT})
set(dbcsr_APIVERSION ${VERSION_MAJOR}.${VERSION_MINOR})

# =================================================================================================
# OPTIONS
include(CMakeDependentOption)

option(USE_OPENMP "Build with OpenMP support" ON)
option(USE_MPI "Build with MPI support" ON)
cmake_dependent_option(WITH_C_API "Build the C API (ISO_C_BINDINGS)" ON "USE_MPI" OFF) # the ISO_C_BINDINGS require MPI unconditionally
cmake_dependent_option(WITH_EXAMPLES "Build the examples" ON "USE_MPI" OFF) # all examples require MPI

set(TEST_MPI_RANKS "auto" CACHE STRING "Number of MPI ranks for testing")
set(TEST_OMP_THREADS 2 CACHE STRING "Number of OpenMP threads for testing")

set(USE_SMM "blas" CACHE STRING "Small Matrix Multiplication implementation to use (default: blas)")
set_property(CACHE USE_SMM PROPERTY STRINGS blas libxsmm)

option(USE_CUDA "Build with CUDA support" OFF)
set(SUPPORTED_CUDA_ARCHITECTURES
    K20X K40 K80 P100 V100)
set(WITH_GPU "P100" CACHE STRING "Set the CUDA GPU architecture if CUDA is enabled (default: P100)")
set_property(CACHE WITH_GPU PROPERTY STRINGS ${SUPPORTED_CUDA_ARCHITECTURES})

option(USE_HIP "Build with HIP support" OFF)
set(SUPPORTED_HIP_ARCHITECTURES
    ${SUPPORTED_CUDA_ARCHITECTURES} Mi50)
set(WITH_GPU "P100" CACHE STRING "Set the CUDA GPU architecture if HIP is enabled (default: P100)")
set_property(CACHE WITH_GPU PROPERTY STRINGS ${SUPPORTED_HIP_ARCHITECTURES})

# USE_CUDA and USE_HIP are mutually exclusive options: we either compile with nvcc OR with hipcc
if (USE_CUDA AND USE_HIP)
  message(FATAL_ERROR "USE_CUDA and USE_HIP options are mutually exclusive. Please choose one.")
endif()

# =================================================================================================
# LANGUAGES AND TESTING
enable_language(Fortran)

if (WITH_C_API AND WITH_EXAMPLES)
  enable_language(CXX)
endif ()

# we're always using at least C++11
set(CMAKE_CXX_STANDARD 11)

# =================================================================================================
# PACKAGE DISCOVERY (compiler configuration can impact package discovery)

# =================================== OpenMP and OpenMP/offload backend
if (USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif ()

# =================================== BLAS & LAPACK, PkgConfig
find_package(PkgConfig)
find_package(LAPACK REQUIRED)  # needed for some of the integrated test routines, also calls find_package(BLAS)

# =================================== Python
# this module looks preferably for version 3 of Python. If not found, version 2 is searched
# In CMake 3.15, if a python virtual environment is activated, it will search the virtual
# environment for a python interpreter before searching elsewhere in the system.
# In CMake <3.15, the system is searched before the virtual environment.
if (NOT Python_EXECUTABLE)
    # If the python interpreter isn't specified as a command line option, look for it:
    find_package(Python COMPONENTS Interpreter REQUIRED)
endif ()

# =================================== MPI
if (USE_MPI)
  get_property(REQUIRED_MPI_COMPONENTS GLOBAL PROPERTY ENABLED_LANGUAGES)
  list(REMOVE_ITEM REQUIRED_MPI_COMPONENTS CUDA)  # CUDA does not have an MPI component
  if (NOT CMAKE_CROSSCOMPILING)  # when cross compiling, assume the users know what they are doing
     set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
  endif ()
  find_package(MPI COMPONENTS ${REQUIRED_MPI_COMPONENTS} REQUIRED)

  if (NOT MPI_Fortran_HAVE_F90_MODULE)
    message(FATAL_ERROR "\
The listed MPI implementation does not provide the required mpi.mod interface. \
When using the GNU compiler in combination with Intel MPI, please use the \
Intel MPI compiler wrappers. Check the INSTALL.md for more information.")
  endif ()
  if ("${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI v2.1" OR "${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI v3.1")
     message(WARNING
            "RMA with ${MPI_Fortran_LIBRARY_VERSION_STRING} is not supported due to issues with its implementation."
            " Please use a newer version of OpenMPI or switch to MPICH if you plan on using MPI-RMA.")
  endif ()
endif ()

# =================================== SMM (Small Matrix-Matrix multiplication)
if (USE_SMM MATCHES "blas")
  message("-- Using BLAS for Small Matrix Multiplication")
elseif (USE_SMM MATCHES "libxsmm")
  # rely on pkg-config in order to link against libxsmm
  pkg_check_modules(deps REQUIRED IMPORTED_TARGET GLOBAL libxsmmf)
  message("-- Using libxsmm for Small Matrix Multiplication")
else()
  message(FATAL_ERROR "Unknown SMM library specified")
endif ()

# =================================== GPU backend
if (USE_CUDA OR USE_HIP)
  enable_language(CXX)
  set(GPU_ARCH_NUMBER_K20X 35)
  set(GPU_ARCH_NUMBER_K40  35)
  set(GPU_ARCH_NUMBER_K80  37)
  set(GPU_ARCH_NUMBER_P100 60)
  set(GPU_ARCH_NUMBER_V100 70)
  set(GPU_ARCH_NUMBER_Mi50 gfx906)
endif()

if (USE_CUDA)

  enable_language(CUDA)
  if (CMAKE_CUDA_COMPILER_VERSION LESS 5.5)
      message(FATAL_ERROR "CUDA version >= 5.5 is required.")
  endif()

  # Make sure the GPU required is supported
  list(FIND SUPPORTED_CUDA_ARCHITECTURES ${WITH_GPU} GPU_SUPPORTED)
  if (GPU_SUPPORTED EQUAL -1)
      message(FATAL_ERROR "GPU architecture requested (${WITH_GPU}) is not supported. Please choose from: ${SUPPORTED_CUDA_ARCHITECTURES}")
  endif()

  # assume that the backend compiler for nvcc understands the -std=c++11
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # set cuda architecture number and compilation flags
  set(ACC_ARCH_NUMBER ${GPU_ARCH_NUMBER_${WITH_GPU}})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${ACC_ARCH_NUMBER} --cudart static -D__CUDA")
  message(STATUS "GPU target architecture: " ${WITH_GPU})
  message(STATUS "GPU architecture number: " ${ACC_ARCH_NUMBER})

  # =================================== BLAS on GPU backend
  find_library(CUBLAS cublas HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  if (NOT CUBLAS)
      message(FATAL_ERROR "cuBLAS library not found but support required for DBCSR's CUDA backend")
  else ()
    message(STATUS "Found cuBLAS: ${CUBLAS}")
  endif ()

endif ()

# inspired from https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/2_Cookbook/12_cmake_hip_add_executable
if (USE_HIP)

  # Make sure the GPU required is supported
  list(FIND SUPPORTED_HIP_ARCHITECTURES ${WITH_GPU} GPU_SUPPORTED)
  if (GPU_SUPPORTED EQUAL -1)
      message(FATAL_ERROR "GPU architecture requested (${WITH_GPU}) is not supported. Please choose from: ${SUPPORTED_HIP_ARCHITECTURES}")
  endif()

  # Set path to HIP installation, include HIP cmake utilities
  if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
      set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to HIP installation")
    else()
      set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to HIP installation")
    endif()
  endif()
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HIP_PATH}/cmake")

  # Find HIP package
  find_package(HIP)
  if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
  else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
  endif()

  # Find hiprtc library (adds support for JIT-ing in HIP)
  find_library(ROCM_HIPRTC_LIB amdhip64 HINTS ${HIP_PATH}/lib)
  if(NOT ROCM_HIPRTC_LIB)
    message(FATAL_ERROR "HIPRTC (HIP library for just-in-time compilation) not found")
  endif()
  set(ENV{HIP_PATH} /opt/rocm/hip)  # workaround bug in hiprtc.cpp

  # Set platform to compile for (NVIDIA-nvcc or ROCm-hcc) as well as corresponding architecture and flags
  # adapted from: https://github.com/ROCmSoftwarePlatform/hipDNN/blob/master/CMakeLists.txt
  execute_process(COMMAND ${HIP_PATH}/bin/hipconfig -P OUTPUT_VARIABLE HIP_PLATFORM)
  message(STATUS "Compiling for platform: " ${HIP_PLATFORM})

  # set appropriate compilation flags depending on platform
  set(ACC_ARCH_NUMBER ${GPU_ARCH_NUMBER_${WITH_GPU}})
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -D__HIP -O3")
  if (${HIP_PLATFORM} STREQUAL "nvcc")
    set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -std=c++11 -arch=sm_${ACC_ARCH_NUMBER} --cudart static")
  else()
    set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} --amdgpu-target=${ACC_ARCH_NUMBER}")
    set(HIP_ARCH_FLAGS "--amdgpu-target=${ACC_ARCH_NUMBER}")
  endif()
  message(STATUS "GPU target architecture: " ${WITH_GPU})
  message(STATUS "GPU architecture number: " ${ACC_ARCH_NUMBER})
  message(STATUS "HIPCC flags: " ${HIP_HIPCC_FLAGS})

  # =================================== BLAS on GPU backend
  find_library(HIPBLAS hipblas HINTS ${HIP_PATH}/../lib)  # /opt/rocm/lib
  if (NOT HIPBLAS)
      message(FATAL_ERROR "hipBLAS library not found but support required for DBCSR's HIP backend")
  else ()
    message(STATUS "Found hipBLAS: ${HIPBLAS}")
  endif ()
endif()

# =================================================================================================
# OPTION HANDLING

# make sure that the default build type is RELEASE
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING
    "Choose the type of build, options are: Debug Release Coverage."
    FORCE)
  # set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "Coverage")
endif ()

# compiler configuration could have impacted package discovery (above)
include(CompilerConfiguration)
include(CheckCompilerSupport)

add_subdirectory(src)

include(CTest)
if (BUILD_TESTING)
  add_subdirectory(tests)
endif ()

if (WITH_EXAMPLES)
  add_subdirectory(examples)
endif ()

include(CustomTargets)
