add_library(vecLibFort STATIC vecLibFort.c)

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
  target_compile_options(vecLibFort PRIVATE -flax-vector-conversions)
endif ()

install(
  TARGETS vecLibFort
  EXPORT DBCSRTargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
