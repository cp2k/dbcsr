
set(DBCSR_PROGRAM_SRCS
  dbcsr_example_1.F
  dbcsr_example_2.F
  dbcsr_example_3.F
  )

# register each program source file as executable
foreach (dbcsr_program_src ${DBCSR_PROGRAM_SRCS})
  get_filename_component(dbcsr_program_name ${dbcsr_program_src} NAME_WE)
  if (USE_HIP)
    hip_add_executable(${dbcsr_program_name} ${dbcsr_program_src})
  else ()
    add_executable(${dbcsr_program_name} ${dbcsr_program_src})
  endif ()
  target_link_libraries(${dbcsr_program_name} dbcsr)

  # with the Intel compiler CMake 3.12 seems to forget that the source is actually Fortran
  # and needs to be told explicitly:
  set_target_properties(${dbcsr_program_name} PROPERTIES LINKER_LANGUAGE Fortran)
endforeach ()

if (WITH_C_API)
  if (USE_HIP)
    hip_add_executable(dbcsr_example_3_cpp dbcsr_example_3.cpp)
  else ()
    add_executable(dbcsr_example_3_cpp dbcsr_example_3.cpp)
  endif ()
  target_link_libraries(dbcsr_example_3_cpp dbcsr_c MPI::MPI_CXX)

  if (CMAKE_CXX_COMPILER_ID STREQUAL "Cray")
    # for recent Cray compiler versions CMake doesn't know
    target_compile_options(dbcsr_example_3_cpp PRIVATE "-hstd=c++14")
  else ()
    target_compile_features(dbcsr_example_3_cpp PRIVATE cxx_std_14)
  endif ()
endif ()
