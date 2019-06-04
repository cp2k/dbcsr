#*****************************************************************************
# FindNumpy
#
# Check if numpy is installed
# Based on the FindNUMPY.cmake distributed with Kitware's SENSEI project (BSD-style license)
#
# This module defines
#  NUMPY_FOUND, set TRUE if numpy and c-api are available
#  NUMPY_VERSION, numpy release version
set(_TMP_PY_OUTPUT)
set(_TMP_PY_RETURN)
exec_program("${Python_EXECUTABLE}"
  ARGS "-c 'import numpy; print(numpy.version.version)'"
  OUTPUT_VARIABLE _TMP_PY_OUTPUT
  RETURN_VALUE _TMP_PY_RETURN)
set(NUMPY_VERSION_FOUND FALSE)
if(NOT _TMP_PY_RETURN)
  set(NUMPY_VERSION_FOUND TRUE)
else()
  set(_TMP_PY_OUTPUT)
endif()
set(NUMPY_VERSION "${_TMP_PY_OUTPUT}" CACHE STRING
  "numpy version string")

if (NOT ${QUIET})
  message(STATUS "NUMPY_VERSION=${NUMPY_VERSION}")
endif()

mark_as_advanced(NUMPY_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMPY DEFAULT_MSG
  NUMPY_VERSION_FOUND)
