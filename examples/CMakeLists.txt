
set(DBCSR_PROGRAM_SRCS
  dbcsr_example_1.F
  dbcsr_example_2.F
  dbcsr_example_3.F
  )

# register each program source file as executable
foreach (dbcsr_program_src ${DBCSR_PROGRAM_SRCS})
  get_filename_component(dbcsr_program_name ${dbcsr_program_src} NAME_WE)
  add_executable(${dbcsr_program_name} ${dbcsr_program_src})
  target_link_libraries(${dbcsr_program_name} dbcsr)
endforeach ()
