# Locate LIBXSMM for DBCSR.
#
# This module first searches a standard CMake package.  When it is used from
# the DBCSR build tree and no local installation is found, it falls back to
# FetchContent.
#
# Result variables: LIBXSMM_FOUND LIBXSMM_INCLUDE_DIRS LIBXSMM_LINK_LIBRARIES
# LIBXSMM_FETCHED DBCSR_USE_LIBXSMM
#
# Imported target: DBCSR::LIBXSMM

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
set(DBCSR_USE_LIBXSMM FALSE)

find_package(libxsmm CONFIG QUIET)

if (TARGET libxsmm::libxsmm)
  set(LIBXSMM_LINK_LIBRARIES libxsmm::libxsmm)
  get_target_property(LIBXSMM_INCLUDE_DIRS libxsmm::libxsmm
                      INTERFACE_INCLUDE_DIRECTORIES)
  if (NOT LIBXSMM_INCLUDE_DIRS OR LIBXSMM_INCLUDE_DIRS MATCHES "-NOTFOUND")
    set(LIBXSMM_INCLUDE_DIRS "")
  endif ()
  message(STATUS "Using LIBXSMM from CMake package")
endif ()

if (NOT LIBXSMM_LINK_LIBRARIES AND DBCSR_FETCH_MISSING_DEPS)
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

  if (TARGET libxsmm::libxsmm)
    set(LIBXSMM_LINK_LIBRARIES libxsmm::libxsmm)
    get_target_property(LIBXSMM_INCLUDE_DIRS libxsmm::libxsmm
                        INTERFACE_INCLUDE_DIRECTORIES)
    if (NOT LIBXSMM_INCLUDE_DIRS OR LIBXSMM_INCLUDE_DIRS MATCHES "-NOTFOUND")
      set(LIBXSMM_INCLUDE_DIRS "${libxsmm_SOURCE_DIR}/include")
    endif ()
  endif ()
endif ()

find_package_handle_standard_args(LIBXSMM DEFAULT_MSG LIBXSMM_LINK_LIBRARIES)

if (LIBXSMM_FOUND)
  set(DBCSR_USE_LIBXSMM TRUE)

  if (NOT TARGET DBCSR::LIBXSMM)
    add_library(DBCSR::LIBXSMM INTERFACE IMPORTED GLOBAL)
    if (LIBXSMM_INCLUDE_DIRS)
      target_include_directories(DBCSR::LIBXSMM
                                 INTERFACE ${LIBXSMM_INCLUDE_DIRS})
    endif ()
    target_link_libraries(DBCSR::LIBXSMM INTERFACE ${LIBXSMM_LINK_LIBRARIES})
  endif ()
endif ()
