# Locate LIBXSMM for DBCSR.
#
# This module first searches already installed packages through pkg-config and
# standard CMake search paths.  When it is used from the DBCSR build tree and no
# local installation is found, it falls back to FetchContent.
#
# Result variables:
#   LIBXSMM_FOUND
#   LIBXSMM_INCLUDE_DIRS
#   LIBXSMM_LINK_LIBRARIES
#   LIBXSMM_COMPILE_DEFINITIONS
#   LIBXSMM_FETCHED
#   DBCSR_USE_LIBXSMM
#
# Imported targets:
#   DBCSR::LIBXSMM
#   xsmm

include(FindPackageHandleStandardArgs)
include("${CMAKE_CURRENT_LIST_DIR}/dependencies.cmake" OPTIONAL)

if (NOT DEFINED DBCSR_FETCH_MISSING_DEPS)
  if (PROJECT_NAME STREQUAL "dbcsr")
    set(DBCSR_FETCH_MISSING_DEPS ON)
  else ()
    set(DBCSR_FETCH_MISSING_DEPS OFF)
  endif ()
endif ()

set(LIBXSMM_FETCHED FALSE)
set(LIBXSMM_COMPILE_DEFINITIONS "")
set(DBCSR_USE_LIBXSMM FALSE)

set(_dbcsr_libxsmm_prefix_hints)
if (DEFINED PACKAGE_PREFIX_DIR)
  list(APPEND _dbcsr_libxsmm_prefix_hints "${PACKAGE_PREFIX_DIR}")
endif ()
find_package(PkgConfig QUIET)

set(_dbcsr_libxsmm_pkg_names libxsmm)
if (DEFINED BUILD_SHARED_LIBS)
  if (BUILD_SHARED_LIBS)
    list(PREPEND _dbcsr_libxsmm_pkg_names libxsmm-shared)
  else ()
    list(PREPEND _dbcsr_libxsmm_pkg_names libxsmm-static)
  endif ()
else ()
  list(PREPEND _dbcsr_libxsmm_pkg_names libxsmm-static libxsmm-shared)
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
  set(LIBXSMM_INCLUDE_DIRS ${DBCSR_LIBXSMM_INCLUDE_DIRS})
  set(LIBXSMM_LINK_LIBRARIES PkgConfig::DBCSR_LIBXSMM)
  message(STATUS "Using prebuilt LIBXSMM from pkg-config")
else ()
  find_path(
    LIBXSMM_INCLUDE_DIRS
    NAMES libxsmm.h
    HINTS ${_dbcsr_libxsmm_prefix_hints}
    PATH_SUFFIXES include)
  find_library(
    LIBXSMM_LIBRARY
    NAMES xsmm
    HINTS ${_dbcsr_libxsmm_prefix_hints})
  if (LIBXSMM_INCLUDE_DIRS AND LIBXSMM_LIBRARY)
    set(LIBXSMM_LINK_LIBRARIES ${LIBXSMM_LIBRARY})
    message(STATUS "Using prebuilt LIBXSMM")
  endif ()
endif ()

if ((NOT LIBXSMM_INCLUDE_DIRS OR NOT LIBXSMM_LINK_LIBRARIES)
    AND DBCSR_FETCH_MISSING_DEPS)
  include(FetchContent)
  message(STATUS "LIBXSMM not found locally -- downloading via FetchContent")
  FetchContent_Declare(
    libxsmm
    GIT_REPOSITORY "https://github.com/libxsmm/libxsmm.git"
    GIT_TAG ${LIBXSMM_GIT_TAG})
  set(XSMM_STATIC
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(libxsmm)
  set(LIBXSMM_FETCHED TRUE)
  set(LIBXSMM_INCLUDE_DIRS "${libxsmm_SOURCE_DIR}/include")
  set(LIBXSMM_LINK_LIBRARIES xsmm)
  set(LIBXSMM_COMPILE_DEFINITIONS LIBXSMM_DEFAULT_CONFIG)
  get_target_property(_xsmm_srcs xsmm SOURCES)
  list(FILTER _xsmm_srcs EXCLUDE REGEX "(binaryexport_generator|gemm_driver)")
  set_target_properties(xsmm PROPERTIES SOURCES "${_xsmm_srcs}")
endif ()

find_package_handle_standard_args(LIBXSMM DEFAULT_MSG LIBXSMM_INCLUDE_DIRS
                                  LIBXSMM_LINK_LIBRARIES)

if (LIBXSMM_FOUND)
  set(DBCSR_USE_LIBXSMM TRUE)

  if (NOT TARGET DBCSR::LIBXSMM)
    add_library(DBCSR::LIBXSMM INTERFACE IMPORTED GLOBAL)
    target_include_directories(DBCSR::LIBXSMM
                               INTERFACE "${LIBXSMM_INCLUDE_DIRS}")
    target_link_libraries(DBCSR::LIBXSMM INTERFACE ${LIBXSMM_LINK_LIBRARIES})
    if (LIBXSMM_COMPILE_DEFINITIONS)
      target_compile_definitions(DBCSR::LIBXSMM
                                 INTERFACE ${LIBXSMM_COMPILE_DEFINITIONS})
    endif ()
  endif ()

  if (NOT TARGET xsmm)
    if (DBCSR_LIBXSMM_FOUND)
      add_library(xsmm INTERFACE IMPORTED GLOBAL)
      target_link_libraries(xsmm INTERFACE PkgConfig::DBCSR_LIBXSMM)
    else ()
      add_library(xsmm UNKNOWN IMPORTED GLOBAL)
      set_target_properties(
        xsmm PROPERTIES IMPORTED_LOCATION "${LIBXSMM_LIBRARY}"
                        INTERFACE_INCLUDE_DIRECTORIES
                        "${LIBXSMM_INCLUDE_DIRS}")
    endif ()
  endif ()
endif ()

unset(_dbcsr_libxsmm_pkg)
unset(_dbcsr_libxsmm_pkg_names)
unset(_dbcsr_libxsmm_prefix_hints)
unset(_xsmm_srcs)
