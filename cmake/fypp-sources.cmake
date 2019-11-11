
add_custom_target(fypp)  # common target for all fypp calls

# Use a system-provided fypp if available, otherwise the bundled one
find_program(FYPP_EXECUTABLE fypp DOC "The FYPP preprocessor" PATHS ../tools/build_utils/fypp/bin)
if (NOT FYPP_EXECUTABLE)
  message(FATAL_ERROR "Failed to find the FYPP preprocessor.")
else ()
  message(STATUS "FYPP preprocessor found.")
endif ()

function (ADD_FYPP_SOURCES OUTVAR)
  set(outfiles)
  foreach (f ${ARGN})
    # first we might need to make the input file absolute
    get_filename_component(f "${f}" ABSOLUTE)
    # get the relative path of the file to the current source dir
    file(RELATIVE_PATH rf "${CMAKE_CURRENT_SOURCE_DIR}" "${f}")
    # set the output filename of fypped sources
    set(of "${CMAKE_CURRENT_BINARY_DIR}/${rf}")

    # create the output directory if it doesn't exist
    get_filename_component(d "${of}" PATH)
    if (NOT IS_DIRECTORY "${d}")
      file(MAKE_DIRECTORY "${d}")
    endif ()

    if ("${f}" MATCHES ".F$")
      # append the output file to the list of outputs
      list(APPEND outfiles "${of}")
      # now add the custom command to generate the output file
      add_custom_command(OUTPUT "${of}" COMMAND ${FYPP_EXECUTABLE} ARGS "${f}" "${of}" DEPENDS "${f}")
    else ()
      configure_file("${f}" "${of}" COPYONLY)
    endif ()
  endforeach ()

  # build a custom target to fypp seperately (required for example by the doc target)
  add_custom_target("fypp_${OUTVAR}" DEPENDS ${outfiles} )
  add_dependencies(fypp "fypp_${OUTVAR}")

  # set the output list in the calling scope
  set(${OUTVAR} ${outfiles} PARENT_SCOPE)
endfunction ()
