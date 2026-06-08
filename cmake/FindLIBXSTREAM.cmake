# Locate LIBXSTREAM for DBCSR's OpenCL backend.
#
# A usable LIBXSTREAM dependency must provide the CMake package target as well
# as the OpenCL helper script and the SMM sample sources used by DBCSR.  This
# module first searches a standard CMake package.  When it is used from the
# DBCSR build tree and no local installation is found, it falls back to
# FetchContent.
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

find_package(libxstream CONFIG QUIET)

if (TARGET libxstream::libxstream)
  set(LIBXSTREAM_LINK_LIBRARIES libxstream::libxstream)
  get_target_property(LIBXSTREAM_INCLUDE_DIRS libxstream::libxstream
                      INTERFACE_INCLUDE_DIRECTORIES)
  if (NOT LIBXSTREAM_INCLUDE_DIRS OR LIBXSTREAM_INCLUDE_DIRS MATCHES "-NOTFOUND")
    set(LIBXSTREAM_INCLUDE_DIRS "")
  endif ()
  message(STATUS "Using LIBXSTREAM from CMake package")
endif ()

foreach (_dbcsr_libxstream_include_dir IN LISTS LIBXSTREAM_INCLUDE_DIRS)
  if (IS_ABSOLUTE "${_dbcsr_libxstream_include_dir}")
    get_filename_component(_dbcsr_libxstream_prefix
                           "${_dbcsr_libxstream_include_dir}/.." ABSOLUTE)
    list(APPEND _dbcsr_libxstream_prefix_hints
         "${_dbcsr_libxstream_prefix}")

    if (_dbcsr_libxstream_prefix MATCHES "/include$")
      get_filename_component(_dbcsr_libxstream_prefix
                             "${_dbcsr_libxstream_prefix}/.." ABSOLUTE)
      list(APPEND _dbcsr_libxstream_prefix_hints
           "${_dbcsr_libxstream_prefix}")
    endif ()
  endif ()
endforeach ()

if (LIBXSTREAM_LINK_LIBRARIES)
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

if (LIBXSTREAM_LINK_LIBRARIES
    AND (NOT LIBXSTREAM_OPENCL_SCRIPT OR NOT LIBXSTREAM_SMM_DIR))
  message(
    FATAL_ERROR
      "LIBXSTREAM CMake package was found, but its OpenCL helper script or SMM samples were not found"
  )
endif ()

if (NOT LIBXSTREAM_LINK_LIBRARIES AND DBCSR_FETCH_MISSING_DEPS)
  include(FetchContent)
  message(STATUS "LIBXSTREAM not found locally -- downloading via FetchContent")
  FetchContent_Declare(
    libxstream
    GIT_REPOSITORY ${LIBXSTREAM_GIT_REPOSITORY}
    GIT_TAG ${LIBXSTREAM_GIT_TAG})
  set(LIBXSTREAM_SHARED
      ${BUILD_SHARED_LIBS}
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(libxstream)
  set(LIBXSTREAM_FETCHED TRUE)

  if (TARGET libxstream::libxstream)
    set(LIBXSTREAM_LINK_LIBRARIES libxstream::libxstream)
    get_target_property(LIBXSTREAM_INCLUDE_DIRS libxstream::libxstream
                        INTERFACE_INCLUDE_DIRECTORIES)
    if (NOT LIBXSTREAM_INCLUDE_DIRS
        OR LIBXSTREAM_INCLUDE_DIRS MATCHES "-NOTFOUND")
      set(LIBXSTREAM_INCLUDE_DIRS "${libxstream_SOURCE_DIR}/include")
    endif ()
    set(LIBXSTREAM_OPENCL_SCRIPT
        "${libxstream_SOURCE_DIR}/scripts/tool_opencl.sh")
    set(LIBXSTREAM_SMM_DIR "${libxstream_SOURCE_DIR}/samples/smm")
  endif ()
endif ()

find_package_handle_standard_args(
  LIBXSTREAM DEFAULT_MSG LIBXSTREAM_LINK_LIBRARIES LIBXSTREAM_OPENCL_SCRIPT
  LIBXSTREAM_SMM_DIR)

if (LIBXSTREAM_FOUND AND NOT TARGET DBCSR::LIBXSTREAM)
  add_library(DBCSR::LIBXSTREAM INTERFACE IMPORTED GLOBAL)
  if (LIBXSTREAM_INCLUDE_DIRS)
    target_include_directories(DBCSR::LIBXSTREAM
                               INTERFACE ${LIBXSTREAM_INCLUDE_DIRS})
  endif ()
  target_link_libraries(DBCSR::LIBXSTREAM
                        INTERFACE ${LIBXSTREAM_LINK_LIBRARIES})
endif ()

unset(_dbcsr_libxstream_prefix_hints)
unset(_dbcsr_libxstream_prefix)
unset(_dbcsr_libxstream_include_dir)
