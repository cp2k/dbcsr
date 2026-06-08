# Locate LIBXS for DBCSR.
#
# This module first searches a standard CMake package.  When it is used from
# the DBCSR build tree and no local installation is found, it falls back to
# FetchContent.
#
# Result variables: LIBXS_FOUND LIBXS_INCLUDE_DIRS LIBXS_LINK_LIBRARIES
# LIBXS_FETCHED
#
# Imported target: DBCSR::LIBXS

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

find_package(libxs CONFIG QUIET)

if (TARGET libxs::libxs)
  set(LIBXS_LINK_LIBRARIES libxs::libxs)
  get_target_property(LIBXS_INCLUDE_DIRS libxs::libxs
                      INTERFACE_INCLUDE_DIRECTORIES)
  if (NOT LIBXS_INCLUDE_DIRS OR LIBXS_INCLUDE_DIRS MATCHES "-NOTFOUND")
    set(LIBXS_INCLUDE_DIRS "")
  endif ()
  message(STATUS "Using LIBXS from CMake package")
endif ()

if (NOT LIBXS_LINK_LIBRARIES AND DBCSR_FETCH_MISSING_DEPS)
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

  if (TARGET libxs::libxs)
    set(LIBXS_LINK_LIBRARIES libxs::libxs)
    get_target_property(LIBXS_INCLUDE_DIRS libxs::libxs
                        INTERFACE_INCLUDE_DIRECTORIES)
    if (NOT LIBXS_INCLUDE_DIRS OR LIBXS_INCLUDE_DIRS MATCHES "-NOTFOUND")
      set(LIBXS_INCLUDE_DIRS "${libxs_SOURCE_DIR}/include")
    endif ()
  endif ()
endif ()

find_package_handle_standard_args(LIBXS DEFAULT_MSG LIBXS_LINK_LIBRARIES)

if (LIBXS_FOUND AND NOT TARGET DBCSR::LIBXS)
  add_library(DBCSR::LIBXS INTERFACE IMPORTED GLOBAL)
  if (LIBXS_INCLUDE_DIRS)
    target_include_directories(DBCSR::LIBXS INTERFACE ${LIBXS_INCLUDE_DIRS})
  endif ()
  target_link_libraries(DBCSR::LIBXS INTERFACE ${LIBXS_LINK_LIBRARIES})
endif ()
