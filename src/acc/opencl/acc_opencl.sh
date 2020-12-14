#!/usr/bin/env bash

for OFILE in $@; do :; done
if [ "$#" -gt 1 ] && [ "${OFILE##*.}" = "h" ]; then
  truncate -s0 ${OFILE}
  for IFILE in $@; do
    if [ "${IFILE}" != "${OFILE}" ]; then
      if [ -e ${IFILE} ] && [ "${IFILE##*.}" = "cl" ]; then
        BASENAME=$(basename "${IFILE}" .cl)
        VNAME=opencl_source_${BASENAME}
        MNAME=$(echo ${VNAME} | tr '[:lower:]' '[:upper:]')
        echo "#define ${MNAME} ${VNAME}" >>${OFILE}
        echo "const char ${VNAME}[] =" >>${OFILE}
        echo "  \"#pragma OPENCL EXTENSION all: enable\\n\"" >>${OFILE}
        sed \
          -e 's/\/\*.[^/*]*\*\///g' -e '/\/\*/,/\*\//d' \
          -e '/^[[:space:]]*$/d' -e 's/[[:space:]]*$//' \
          -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/  "/' -e 's/$/\\n"/' \
          ${IFILE} >>${OFILE}
        echo ";" >>${OFILE}
      else
        echo "ERROR: ${IFILE} does not exist or is no OpenCL file!"
        rm -f ${OFILE}
        exit 1
      fi
    fi
  done
else
  echo "Usage: $0 infile1.cl infile2.cl .. infilen.cl outfile.h"
fi
