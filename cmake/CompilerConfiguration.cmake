# Sanitizers in Debug builds (default ON, disable with -DUSE_ASAN=OFF)
option(USE_ASAN "Enable sanitizers (ASan/UBSan/LSan) in Debug builds" ON)

# ==============================================================================
# Fortran
# ==============================================================================

if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} -ffree-form -std=f2008ts -fimplicit-none")
  set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} -Wno-maybe-uninitialized -Wno-function-elimination")
  set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} -Werror=aliasing -Werror=ampersand")
  set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} -Werror=c-binding-type -Werror=intrinsic-shadow")
  set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} -Werror=intrinsics-std -Werror=line-truncation")
  set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} -Werror=tabs -Werror=target-lifetime")
  set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} -Werror=underflow -Werror=unused-but-set-parameter")
  set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} -Werror=unused-but-set-variable -Werror=unused-variable")
  set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} -Werror=unused-dummy-argument -Werror=conversion")
  set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} -Werror=zerotrip -Werror=unused-parameter")

  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch")
  else ()
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Werror=argument-mismatch")
  endif ()
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wno-error=uninitialized")
  endif ()

  include(CheckFortranCompilerFlag)
  check_fortran_compiler_flag("-Wdeprecated-openmp" _fc_has_deprecated_openmp)
  if (_fc_has_deprecated_openmp)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wno-deprecated-openmp")
  endif ()

  set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -g -funroll-loops")
  set(CMAKE_Fortran_FLAGS_COVERAGE
      "-O0 -g --coverage -fno-omit-frame-pointer -fcheck=all,no-array-temps -ffpe-trap=invalid,zero,overflow -fbacktrace -finit-real=snan -finit-integer=-42 -finit-derived -Werror=realloc-lhs -finline-matmul-limit=0 -Werror")
  if (USE_ASAN)
    set(CMAKE_Fortran_FLAGS_DEBUG
        "-O2 -ggdb -fno-omit-frame-pointer -fcheck=all -ffpe-trap=invalid,zero,overflow -fbacktrace -finit-real=snan -finit-integer=-42 -finit-derived -finline-matmul-limit=0 -Wall -Wextra -Werror -Werror=realloc-lhs -Wno-error=array-temporaries -Wno-error=compare-reals -Wno-error=surprising -fsanitize=undefined -fsanitize=address -fsanitize-recover=all")
    if ((NOT USE_MPI) OR (NOT "${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI"))
      set(CMAKE_Fortran_FLAGS_DEBUG
          "${CMAKE_Fortran_FLAGS_DEBUG} -fsanitize=leak")
    endif ()
  else ()
    set(CMAKE_Fortran_FLAGS_DEBUG
        "-O0 -ggdb -fno-omit-frame-pointer -fcheck=all -ffpe-trap=invalid,zero,overflow -fbacktrace -finit-real=snan -finit-integer=-42 -finit-derived -finline-matmul-limit=0 -Wall -Wextra -Werror -Werror=realloc-lhs -Wno-error=array-temporaries -Wno-error=compare-reals -Wno-error=surprising")
  endif ()

elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -free -stand=f18 -fpp -heap-arrays")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -g")
  set(CMAKE_Fortran_FLAGS_DEBUG "-O2 -debug")

elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -free -stand f18 -fpp -heap-arrays")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -g")
  set(CMAKE_Fortran_FLAGS_DEBUG "-O2 -debug")

elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "PGI")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Mfreeform -Mextend -Mallocatable=03")
  set(CMAKE_Fortran_FLAGS_RELEASE "-fast")
  set(CMAKE_Fortran_FLAGS_DEBUG "-g")

elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "NAG")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -f2008 -free -Warn=reallocation -Warn=subnormal")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O2")
  set(CMAKE_Fortran_FLAGS_DEBUG "-g -C")
  if (NOT OpenMP_FOUND)
    set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -gline")
    set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -C=all")
  endif ()

elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "Cray")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -f free -M3105 -ME7212")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O2 -G2")
  set(CMAKE_Fortran_FLAGS_DEBUG "-G2")
  set(CMAKE_Fortran_MODOUT_FLAG "-ef")

elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-form -std=f2018 -cpp")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3")
  set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g")

elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "Flang")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-form -cpp")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3")
  set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g")

else ()
  message(WARNING
    "Unknown Fortran compiler, trying without any additional flags.\n"
    "-- CMAKE_Fortran_COMPILER_ID: ${CMAKE_Fortran_COMPILER_ID}\n"
    "-- CMAKE_Fortran_COMPILER: ${CMAKE_Fortran_COMPILER}")
endif ()

# ==============================================================================
# C / C++
# ==============================================================================

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_RELEASE
      "-O3 -g -funroll-loops -Wall -Wextra -Werror -Wno-missing-field-initializers")
  set(CMAKE_CXX_FLAGS_COVERAGE
      "-O0 -g --coverage -Wall -Wextra -Werror -Wno-missing-field-initializers")
  if (USE_ASAN)
    set(CMAKE_CXX_FLAGS_DEBUG
        "-O2 -ggdb -Wall -Wextra -Werror -Wno-missing-field-initializers -fsanitize=undefined -fsanitize=address -fsanitize-recover=all")
    if ((NOT USE_MPI) OR (NOT "${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI"))
      set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=leak")
    endif ()
  else ()
    set(CMAKE_CXX_FLAGS_DEBUG
        "-O0 -ggdb -Wall -Wextra -Werror -Wno-missing-field-initializers")
  endif ()

elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
  set(CMAKE_CXX_FLAGS_COVERAGE "-O0 -g --coverage")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "-lgcov")
  endif ()

elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel|IntelLLVM")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -debug")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
  set(CMAKE_CXX_FLAGS_RELEASE "-fast")
  set(CMAKE_CXX_FLAGS_DEBUG "-g")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Cray")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  set(CMAKE_CXX_FLAGS_DEBUG "-G2")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -h system_alloc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -h system_alloc")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -h system_alloc")
    list(FILTER CMAKE_C_IMPLICIT_LINK_LIBRARIES EXCLUDE REGEX "tcmalloc")
    list(FILTER CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES EXCLUDE REGEX "tcmalloc")
  endif ()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -hnoacc -h nomessage=1234")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hnoacc -h nomessage=1234")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -hnoacc -M1234")

else ()
  message(WARNING
    "Unknown C++ compiler, trying without any additional flags.\n"
    "-- CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}\n"
    "-- CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
endif ()

# C inherits from CXX
set(CMAKE_C_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
set(CMAKE_C_FLAGS_COVERAGE ${CMAKE_CXX_FLAGS_COVERAGE})
set(CMAKE_C_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})

# Suppress GFortran runtime warnings when LIBXS provides the wrapper
if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU"
    AND NOT USE_ASAN AND USE_LIBXS AND TARGET libxs::libxs)
  add_link_options("-Wl,--wrap=_gfortran_runtime_warning_at")
endif ()
