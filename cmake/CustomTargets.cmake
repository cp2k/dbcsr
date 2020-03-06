set(ARCHIVE_NAME "${CMAKE_PROJECT_NAME}-${dbcsr_VERSION}")
add_custom_target(dist
  COMMENT "Building distribution: ${ARCHIVE_NAME}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/dist"
  COMMAND git archive-all "${CMAKE_BINARY_DIR}/dist/${ARCHIVE_NAME}.tar.gz"
  COMMAND ${CMAKE_COMMAND} -E echo "SHA512 Digests:"
  COMMAND ${CMAKE_COMMAND} -E sha512sum "${CMAKE_BINARY_DIR}/dist/${ARCHIVE_NAME}.tar.gz"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

find_program(FORD_EXE ford
  DOC "path to the ford executable (required to generate the documentation)"
  )

# Copy the FORD project-file into the build directory
set (FORD_PROJECT_FILE "${CMAKE_BINARY_DIR}/DBCSR.md")
configure_file(DBCSR.md "${FORD_PROJECT_FILE}")

add_custom_target(doc
  COMMENT "Generating API documentation"
  COMMAND "${FORD_EXE}" "${FORD_PROJECT_FILE}"
  VERBATIM
  )
add_dependencies(doc fypp)  # only depend on the fypp step to avoid building everything just for the docs


find_program(LCOV_EXE lcov
  DOC "path to the lcov executable (required to generate coverage reports)"
  )

find_program(GENHTML_EXE genhtml
  DOC "path to the genhtml executable (required to generate HTML coverage reports)"
  )

add_custom_target(cov-info
  COMMAND "${LCOV_EXE}" --directory "${CMAKE_BINARY_DIR}" --base-dir "${CMAKE_SOURCE_DIR}" -r "${CMAKE_SOURCE_DIR}/tests/*" --no-external --capture --output-file coverage.info
  COMMAND "${LCOV_EXE}" --list coverage.info
  VERBATIM
  BYPRODUCTS coverage.info
  COMMENT "Generate coverage.info using lcov"
  )

add_custom_target(cov-genhtml
  COMMAND "${GENHTML_EXE}" coverage.info --output-directory cov-html
  COMMENT "Generate a HTML-based coverage report using lcov in ${CMAKE_BINARY_DIR}/cov-html"
  VERBATIM
  )  # Note: this directory will not be cleaned by `cmake --build .. -- clean`
add_dependencies(cov-genhtml cov-info)
