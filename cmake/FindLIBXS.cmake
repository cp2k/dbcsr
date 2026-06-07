# Locate LIBXS for DBCSR.
#
# This module first searches already installed packages through pkg-config and
# standard CMake search paths.  When it is used from the DBCSR build tree and no
# local installation is found, it falls back to FetchContent.
#
# Result variables:
#   LIBXS_FOUND
#   LIBXS_INCLUDE_DIRS
#   LIBXS_LINK_LIBRARIES
#   LIBXS_FETCHED
#
# Imported target:
#   DBCSR::LIBXS

include(FindPackageHandleStandardArgs)
include("${CMAKE_CURRENT_LIST_DIR}/dependencies.cmake" OPTIONAL)

if (NOT DEFINED DBCSR_FETCH_MISSING_DEPS)
  if (PROJECT_NAME STREQUAL "dbcsr")
    set(DBCSR_FETCH_MISSING_DEPS ON)
  else ()
    set(DBCSR_FETCH_MISSING_DEPS OFF)
  endif ()
endif ()

set(LIBXS_FETCHED FALSE)
set(_dbcsr_libxs_prefix_hints)
if (DEFINED PACKAGE_PREFIX_DIR)
  list(APPEND _dbcsr_libxs_prefix_hints "${PACKAGE_PREFIX_DIR}")
endif ()
find_package(PkgConfig QUIET)

set(_dbcsr_libxs_pkg_names libxs)
if (DEFINED BUILD_SHARED_LIBS)
  if (BUILD_SHARED_LIBS)
    list(PREPEND _dbcsr_libxs_pkg_names libxs-shared)
  else ()
    list(PREPEND _dbcsr_libxs_pkg_names libxs-static)
  endif ()
else ()
  list(PREPEND _dbcsr_libxs_pkg_names libxs-static libxs-shared)
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
  set(LIBXS_INCLUDE_DIRS ${DBCSR_LIBXS_INCLUDE_DIRS})
  set(LIBXS_LINK_LIBRARIES PkgConfig::DBCSR_LIBXS)
  message(STATUS "Using prebuilt LIBXS from pkg-config")
else ()
  find_path(
    LIBXS_INCLUDE_DIRS
    NAMES libxs.h
    HINTS ${_dbcsr_libxs_prefix_hints}
    PATH_SUFFIXES include)
  find_library(
    LIBXS_LIBRARY
    NAMES xs
    HINTS ${_dbcsr_libxs_prefix_hints})
  if (LIBXS_INCLUDE_DIRS AND LIBXS_LIBRARY)
    set(LIBXS_LINK_LIBRARIES ${LIBXS_LIBRARY})
    message(STATUS "Using prebuilt LIBXS")
  endif ()
endif ()

if ((NOT LIBXS_INCLUDE_DIRS OR NOT LIBXS_LINK_LIBRARIES)
    AND DBCSR_FETCH_MISSING_DEPS)
  include(FetchContent)
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
  set(LIBXS_INCLUDE_DIRS "${libxs_SOURCE_DIR}/include")
  set(LIBXS_LINK_LIBRARIES libxs::libxs)
endif ()

find_package_handle_standard_args(LIBXS DEFAULT_MSG LIBXS_INCLUDE_DIRS
                                  LIBXS_LINK_LIBRARIES)

if (LIBXS_FOUND AND NOT TARGET DBCSR::LIBXS)
  add_library(DBCSR::LIBXS INTERFACE IMPORTED GLOBAL)
  target_include_directories(DBCSR::LIBXS INTERFACE "${LIBXS_INCLUDE_DIRS}")
  target_link_libraries(DBCSR::LIBXS INTERFACE ${LIBXS_LINK_LIBRARIES})
endif ()

unset(_dbcsr_libxs_pkg)
unset(_dbcsr_libxs_pkg_names)
unset(_dbcsr_libxs_prefix_hints)
