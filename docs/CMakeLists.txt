# =================================================================================================
# FORD - DOCUMENTATION GENERATION
find_program(
  FORD_EXE ford
  DOC "path to the ford executable (required to generate the documentation)")

# Copy the FORD project-file into the build directory
set(FORD_PROJECT_FILE "${CMAKE_BINARY_DIR}/DBCSR.md")
configure_file(${CMAKE_SOURCE_DIR}/DBCSR.md "${FORD_PROJECT_FILE}")

# Copy the FORD project-file into the build directory
add_custom_target(
  doc
  COMMENT "Generating API documentation and doc pages"
  COMMAND "${FORD_EXE}" "${FORD_PROJECT_FILE}"
  VERBATIM)

if (BUILD_TESTING)
  add_dependencies(doc doc_copy_tests)
endif ()

if (WITH_C_API AND WITH_EXAMPLES)
  add_dependencies(doc doc_copy_examples)
endif ()

add_dependencies(doc fypp) # only depend on the fypp step to avoid building
                           # everything just for the docs
