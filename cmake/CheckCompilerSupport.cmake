include(CheckFortranSourceCompiles)

set(CHECK_PROGRAMS f2008-norm2.f90 f2008-block_construct.f90
                   f2008-contiguous.f90 f95-reshape-order-allocatable.f90)

set(_saved_fortran_flags "${CMAKE_Fortran_FLAGS}")
set(CMAKE_Fortran_FLAGS "")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
foreach (prog ${CHECK_PROGRAMS})
  get_filename_component(prog_ext ${prog} EXT) # get the src extension to pass
                                               # along
  get_filename_component(prog_name ${prog} NAME_WE)

  file(READ "${CMAKE_CURRENT_LIST_DIR}/compiler-tests/${prog}" prog_src)
  CHECK_Fortran_SOURCE_COMPILES("${prog_src}" "${prog_name}" SRC_EXT
                                "${prog_ext}")

  if (NOT ${prog_name})
    message(
      FATAL_ERROR "Your compiler does not support all required F2008 features")
  endif ()
endforeach ()
unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
set(CMAKE_Fortran_FLAGS "${_saved_fortran_flags}")
