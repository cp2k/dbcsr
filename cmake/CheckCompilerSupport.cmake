
include(CheckFortranSourceCompiles)

set(CHECK_PROGRAMS
  f2008-norm2.f90
  f2008-block_construct.f90
  f2008-contiguous.f90
  )

if (CMAKE_Fortran_COMPILER_ID STREQUAL "PGI")
  set(CHECK_PROGRAMS ${CHECK_PROGRAMS} f95-reshape-order-allocatable.f90)
endif ()

foreach (prog ${CHECK_PROGRAMS})
  get_filename_component(prog_ext ${prog} EXT)  # get the src extension to pass along
  get_filename_component(prog_name ${prog} NAME_WE)

  file(READ "${CMAKE_CURRENT_LIST_DIR}/compiler-tests/${prog}" prog_src)
  check_fortran_source_compiles("${prog_src}" "${prog_name}" SRC_EXT "${prog_ext}")

  if (NOT ${prog_name})
    message(FATAL_ERROR "Your compiler does not support all required F2008 features")
  endif ()
endforeach ()
