# Locate LIBXSTREAM for DBCSR's OpenCL backend.
#
# A usable LIBXSTREAM dependency must provide the library/header as well as the
# OpenCL helper script and the SMM sample sources used by DBCSR.  This module
# first searches already installed packages through pkg-config and standard
# CMake search paths.  When it is used from the DBCSR build tree and no complete
# local installation is found, it falls back to FetchContent.
#
# Result variables: LIBXSTREAM_FOUND LIBXSTREAM_INCLUDE_DIRS
# LIBXSTREAM_LINK_LIBRARIES LIBXSTREAM_OPENCL_SCRIPT LIBXSTREAM_SMM_DIR
# LIBXSTREAM_FETCHED
#
# Imported target: DBCSR::LIBXSTREAM

include(FindPackageHandleStandardArgs)
include("${CMAKE_CURRENT_LIST_DIR}/dependencies.cmake" OPTIONAL)

if (NOT DEFINED DBCSR_FETCH_MISSING_DEPS)
  if (PROJECT_NAME STREQUAL "dbcsr")
    set(DBCSR_FETCH_MISSING_DEPS ON)
  else ()
    set(DBCSR_FETCH_MISSING_DEPS OFF)
  endif ()
endif ()

set(LIBXSTREAM_FETCHED FALSE)

set(_dbcsr_libxstream_prefix_hints)
if (DEFINED PACKAGE_PREFIX_DIR)
  list(APPEND _dbcsr_libxstream_prefix_hints "${PACKAGE_PREFIX_DIR}")
endif ()
find_package(PkgConfig QUIET)

set(_dbcsr_libxstream_pkg_names libxstream)
if (DEFINED BUILD_SHARED_LIBS)
  if (BUILD_SHARED_LIBS)
    list(PREPEND _dbcsr_libxstream_pkg_names libxstream-shared)
  else ()
    list(PREPEND _dbcsr_libxstream_pkg_names libxstream-static)
  endif ()
else ()
  list(PREPEND _dbcsr_libxstream_pkg_names libxstream-static libxstream-shared)
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
  set(LIBXSTREAM_INCLUDE_DIRS ${DBCSR_LIBXSTREAM_INCLUDE_DIRS})
  set(LIBXSTREAM_LINK_LIBRARIES PkgConfig::DBCSR_LIBXSTREAM)
  if (DBCSR_LIBXSTREAM_PREFIX)
    list(APPEND _dbcsr_libxstream_prefix_hints "${DBCSR_LIBXSTREAM_PREFIX}")
  endif ()
  message(STATUS "Using prebuilt LIBXSTREAM from pkg-config")
else ()
  find_path(
    LIBXSTREAM_INCLUDE_DIRS
    NAMES libxstream.h
    HINTS ${_dbcsr_libxstream_prefix_hints}
    PATH_SUFFIXES include)
  find_library(
    LIBXSTREAM_LIBRARY
    NAMES xstream
    HINTS ${_dbcsr_libxstream_prefix_hints})
  if (LIBXSTREAM_INCLUDE_DIRS AND LIBXSTREAM_LIBRARY)
    set(LIBXSTREAM_LINK_LIBRARIES ${LIBXSTREAM_LIBRARY})
    message(STATUS "Using prebuilt LIBXSTREAM")
  endif ()
endif ()

foreach (_dbcsr_libxstream_include_dir IN LISTS LIBXSTREAM_INCLUDE_DIRS)
  get_filename_component(_dbcsr_libxstream_prefix
                         "${_dbcsr_libxstream_include_dir}/.." ABSOLUTE)
  list(APPEND _dbcsr_libxstream_prefix_hints "${_dbcsr_libxstream_prefix}")
endforeach ()

if (LIBXSTREAM_INCLUDE_DIRS AND LIBXSTREAM_LINK_LIBRARIES)
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
endif ()

if ((NOT LIBXSTREAM_INCLUDE_DIRS
     OR NOT LIBXSTREAM_LINK_LIBRARIES
     OR NOT LIBXSTREAM_OPENCL_SCRIPT
     OR NOT LIBXSTREAM_SMM_DIR
    )
    AND DBCSR_FETCH_MISSING_DEPS)
  include(FetchContent)
  if (LIBXSTREAM_INCLUDE_DIRS AND LIBXSTREAM_LINK_LIBRARIES)
    message(
      STATUS
        "Prebuilt LIBXSTREAM was found, but its OpenCL helper script or SMM samples were not found -- downloading via FetchContent"
    )
  else ()
    message(
      STATUS "LIBXSTREAM not found locally -- downloading via FetchContent")
  endif ()
  FetchContent_Declare(
    libxstream
    GIT_REPOSITORY ${LIBXSTREAM_GIT_REPOSITORY}
    GIT_TAG ${LIBXSTREAM_GIT_TAG})
  set(LIBXSTREAM_SHARED
      ${BUILD_SHARED_LIBS}
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(libxstream)
  set(LIBXSTREAM_FETCHED TRUE)
  set(LIBXSTREAM_INCLUDE_DIRS "${libxstream_SOURCE_DIR}/include")
  set(LIBXSTREAM_LINK_LIBRARIES libxstream::libxstream)
  set(LIBXSTREAM_OPENCL_SCRIPT
      "${libxstream_SOURCE_DIR}/scripts/tool_opencl.sh")
  set(LIBXSTREAM_SMM_DIR "${libxstream_SOURCE_DIR}/samples/smm")
endif ()

find_package_handle_standard_args(
  LIBXSTREAM DEFAULT_MSG LIBXSTREAM_INCLUDE_DIRS LIBXSTREAM_LINK_LIBRARIES
  LIBXSTREAM_OPENCL_SCRIPT LIBXSTREAM_SMM_DIR)

if (LIBXSTREAM_FOUND AND NOT TARGET DBCSR::LIBXSTREAM)
  add_library(DBCSR::LIBXSTREAM INTERFACE IMPORTED GLOBAL)
  target_include_directories(DBCSR::LIBXSTREAM
                             INTERFACE "${LIBXSTREAM_INCLUDE_DIRS}")
  target_link_libraries(DBCSR::LIBXSTREAM
                        INTERFACE ${LIBXSTREAM_LINK_LIBRARIES})
endif ()

unset(_dbcsr_libxstream_pkg)
unset(_dbcsr_libxstream_pkg_names)
unset(_dbcsr_libxstream_prefix_hints)
unset(_dbcsr_libxstream_prefix)
unset(_dbcsr_libxstream_include_dir)
