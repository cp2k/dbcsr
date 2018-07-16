cmake_minimum_required (VERSION 3.0)

# If available, enable developer warnings for the build system
if (POLICY CMP0005)
    cmake_policy(SET CMP0005 NEW)
endif ()

# include our cmake snippets
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

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
  set(CMAKE_Fortran_FLAGS         "-ffree-form -ffree-line-length-none -std=f2003 -fopenmp")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -funroll-loops")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -ggdb")
elseif (Fortran_COMPILER_NAME MATCHES "ifort.*")
  set(CMAKE_Fortran_FLAGS "-free -stand f03 -fpp")
  # Disable the line-length-extension warning #5268
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -diag-disable=5268")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -debug")
else ()
  message("CMAKE_Fortran_COMPILER full path: " ${CMAKE_Fortran_COMPILER})
  message("Fortran compiler: " ${Fortran_COMPILER_NAME})
  message("No optimized Fortran compiler flags are known, we just try -O2...")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O2")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g")
endif ()


add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
