#!-------------------------------------------------------------------------------------------------!
#!   DBCSR: A general program to perform molecular dynamics simulations                             !
#!   Copyright 2000-2026 DBCSR developers group <https://cp2k.org>                                  !
#!                                                                                                 !
#!   SPDX-License-Identifier: GPL-2.0-or-later                                                     !
#!-------------------------------------------------------------------------------------------------!

include(FindPackageHandleStandardArgs)
find_package(PkgConfig QUIET)

# Prefer pkg-config when available.
if (PKG_CONFIG_FOUND)
  pkg_check_modules(DBCSR_LIBXSMM QUIET IMPORTED_TARGET GLOBAL libxsmm)
endif ()

# Optional explicit prefix hint.
set(_dbcsr_libxsmm_hints)
if (DBCSR_LIBXSMM_ROOT)
  list(APPEND _dbcsr_libxsmm_hints "${DBCSR_LIBXSMM_ROOT}")
endif ()

# Fallback discovery when pkg-config did not populate usable values.
if (NOT DBCSR_LIBXSMM_INCLUDE_DIR)
  find_path(
    DBCSR_LIBXSMM_INCLUDE_DIR
    NAMES libxsmm.h
    HINTS ${_dbcsr_libxsmm_hints}
    PATH_SUFFIXES include)
endif ()

if (NOT DBCSR_LIBXSMM_LIBRARY)
  find_library(
    DBCSR_LIBXSMM_LIBRARY
    NAMES xsmm
    HINTS ${_dbcsr_libxsmm_hints}
    PATH_SUFFIXES lib lib64)
endif ()

# Reconcile pkg-config variables with the fallback naming used below.
if (NOT DBCSR_LIBXSMM_INCLUDE_DIR AND DBCSR_LIBXSMM_INCLUDE_DIRS)
  list(GET DBCSR_LIBXSMM_INCLUDE_DIRS 0 DBCSR_LIBXSMM_INCLUDE_DIR)
endif ()

if (NOT DBCSR_LIBXSMM_LIBRARY AND DBCSR_LIBXSMM_LINK_LIBRARIES)
  list(GET DBCSR_LIBXSMM_LINK_LIBRARIES 0 DBCSR_LIBXSMM_LIBRARY)
endif ()

find_package_handle_standard_args(
  LIBXSMM REQUIRED_VARS DBCSR_LIBXSMM_INCLUDE_DIR DBCSR_LIBXSMM_LIBRARY)

if (NOT TARGET dbcsr::libxsmm)
  if (TARGET PkgConfig::DBCSR_LIBXSMM)
    add_library(dbcsr::libxsmm ALIAS PkgConfig::DBCSR_LIBXSMM)
  else ()
    add_library(dbcsr::libxsmm INTERFACE IMPORTED)
    set_target_properties(
      dbcsr::libxsmm
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${DBCSR_LIBXSMM_INCLUDE_DIR}"
                 INTERFACE_LINK_LIBRARIES "${DBCSR_LIBXSMM_LIBRARY}")
  endif ()
endif ()

mark_as_advanced(DBCSR_LIBXSMM_INCLUDE_DIR DBCSR_LIBXSMM_LIBRARY)
