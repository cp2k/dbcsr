cmake_minimum_required (VERSION 3.10)

# include our cmake snippets
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(dbcsr)

include(CMakeDependentOption)

option(USE_MPI "Build with MPI support" ON)
option(USE_OPENMP "Build with OpenMP support" ON)
cmake_dependent_option(WITH_C_API "Build the C API (ISO_C_BINDINGS)" ON "USE_MPI" OFF) # the ISO_C_BINDINGS require MPI unconditionally
cmake_dependent_option(WITH_EXAMPLES "Build the examples" ON "USE_MPI" OFF) # all examples require MPI

set(TEST_MPI_RANKS "auto" CACHE STRING "Number of MPI ranks for testing")
set(TEST_OMP_THREADS 2 CACHE STRING "Number of OpenMP threads for testing")

set(USE_SMM "blas" CACHE STRING "Small Matrix Multiplication implementation to use (default: blas)")
set_property(CACHE USE_SMM PROPERTY STRINGS blas libxsmm)

option(USE_CUDA "Build with CUDA support" OFF)
cmake_dependent_option(USE_CUBLAS "Build with CUBLAS support" OFF "USE_CUDA" OFF)
set(WITH_GPU "P100" CACHE STRING "Set the CUDA GPU architecture if CUDA is enabled (default: P100)")
set_property(CACHE WITH_GPU PROPERTY STRINGS K20X K40 K80 P100 V100)

enable_language(Fortran)
enable_testing() # enables the `make test` target

if (WITH_C_API AND WITH_EXAMPLES)
  enable_language(CXX)
endif ()

if (USE_CUDA)
  enable_language(CXX CUDA)

  set(CUDA_ARCH_NUMBER_K20X 35)
  set(CUDA_ARCH_NUMBER_K40  35)
  set(CUDA_ARCH_NUMBER_K80  37)
  set(CUDA_ARCH_NUMBER_P100 60)
  set(CUDA_ARCH_NUMBER_V100 70)
  set(CUDA_ARCH_NUMBER ${CUDA_ARCH_NUMBER_${WITH_GPU}})

  # assume that the backend compiler for nvcc understands the -std=c++11
  set(CMAKE_CUDA_FLAGS "-std=c++11 -arch=sm_${CUDA_ARCH_NUMBER} --cudart static")
endif ()


# PACKAGE DISCOVERY:

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(PkgConfig)

if (USE_MPI)
  get_property(REQUIRED_MPI_COMPONENTS GLOBAL PROPERTY ENABLED_LANGUAGES)
  list(REMOVE_ITEM REQUIRED_MPI_COMPONENTS CUDA)  # CUDA does not have a MPI component
  find_package(MPI COMPONENTS ${REQUIRED_MPI_COMPONENTS} REQUIRED)
endif ()

if (USE_SMM MATCHES "blas")
  message("-- Using BLAS for Small Matrix Multiplication")
elseif (USE_SMM MATCHES "libxsmm")
  # rely on pkg-config since it's quiet hard to link against libxsmm properly
  pkg_check_modules(deps REQUIRED IMPORTED_TARGET GLOBAL libxsmmf)
  message("-- Using libxsmm for Small Matrix Multiplication")
else()
  message(FATAL_ERROR "Unknown SMM library specified" )
endif ()

if (USE_CUBLAS)
  # the rest of CUDA is detected by enabling the language
  find_library(CUBLAS cublas HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  if (NOT CUBLAS)
    message(FATAL_ERROR "cuBLAS library not found but support requested")
  endif ()
endif ()

if (USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif ()

# make sure that the default is a RELEASE
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE CACHE STRING
    "Choose the type of build, options are: Debug Release Coverage."
    FORCE)
endif ()

include(CompilerConfiguration)
include(CheckCompilerSupport)

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

add_subdirectory(src)
add_subdirectory(tests)

if (WITH_EXAMPLES)
  add_subdirectory(examples)
endif ()

set(ARCHIVE_NAME "${CMAKE_PROJECT_NAME}-${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
add_custom_target(dist
  COMMENT "Building distribution: ${ARCHIVE_NAME}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/dist"
  COMMAND git archive-all "${CMAKE_BINARY_DIR}/dist/${ARCHIVE_NAME}.tar.gz"
  COMMAND ${CMAKE_COMMAND} -E echo "SHA512 Digests:"
  COMMAND ${CMAKE_COMMAND} -E sha512sum "${CMAKE_BINARY_DIR}/dist/${ARCHIVE_NAME}.tar.gz"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

find_program(FORD_EXE ford
  DOC "path to the ford executable (required to generate the documentation)"
  )

# copy the FORD project-file into the build directory
configure_file(project-file.md.in project-file.md)

add_custom_target(doc
  COMMENT "Generating API documentation"
  COMMAND "${FORD_EXE}" project-file.md
  VERBATIM
  )
add_dependencies(doc fypp)  # only depend on the fypp step to avoid building everything just for the docs
