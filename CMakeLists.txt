cmake_minimum_required (VERSION 3.0)

# If available, enable developer warnings for the build system
if (POLICY CMP0005)
    cmake_policy(SET CMP0005 NEW)
endif ()

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
  set(CMAKE_Fortran_FLAGS "-ffree-form -ffree-line-length-none -std=f2003 -fopenmp")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -funroll-loops -ffast-math")
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

# Use a system-provided fypp if available, otherwise the bundled one
find_program(FYPP_EXECUTABLE fypp DOC "The FYPP preprocessor" PATHS tools/build_utils/)
if (NOT FYPP_EXECUTABLE)
  message(FATAL_ERROR "Failed to find the FYPP preprocessor.")
else ()
  message(STATUS "FYPP preprocessor found.")
endif ()

function (ADD_FYPP_SOURCES OUTVAR)
  set(outfiles)
  foreach (f ${ARGN})
    # first we might need to make the input file absolute
    get_filename_component(f "${f}" ABSOLUTE)
    # get the relative path of the file to the current source dir
    file(RELATIVE_PATH rf "${CMAKE_CURRENT_SOURCE_DIR}" "${f}")
    # set the output filename of fypped sources
    set(of "${CMAKE_CURRENT_BINARY_DIR}/${rf}")

    # create the output directory if it doesn't exist
    get_filename_component(d "${of}" PATH)
    if (NOT IS_DIRECTORY "${d}")
      file(MAKE_DIRECTORY "${d}")
    endif ()

    if ("${f}" MATCHES ".F$")
      # append the output file to the list of outputs
      list(APPEND outfiles "${of}")
      # now add the custom command to generate the output file
      add_custom_command(OUTPUT "${of}" COMMAND ${FYPP_EXECUTABLE} ARGS "${f}" "${of}" DEPENDS "${f}")
    else ()
      configure_file("${f}" "${of}" COPYONLY)
    endif ()
  endforeach ()
  # set the output list in the calling scope
  set(${OUTVAR} ${outfiles} PARENT_SCOPE)
endfunction ()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
