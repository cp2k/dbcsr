#!/usr/bin/env bash

BASENAME=$(command -v basename)
SED=$(command -v gsed)
CPP=$(command -v cpp)
RM=$(command -v rm)

# GNU sed is desired (macOS)
if [ "" = "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "${BASENAME}" ] && [ "${SED}" ] && [ "${RM}" ]; then
  for OFILE in $@; do :; done
  if [ "$#" -gt 1 ]; then
    # allow for instance /dev/stdout
    if [ "${OFILE##*.}" = "h" ]; then truncate -s0 ${OFILE}; fi
    for IFILE in $@; do
      if [ "${IFILE}" != "${OFILE}" ]; then
        if [ -e ${IFILE} ] && [ "${IFILE##*.}" = "cl" ]; then
          BNAME=$(${BASENAME} "${IFILE}" .cl)
          VNAME=opencl_source_${BNAME}
          MNAME=$(echo ${VNAME} | tr '[:lower:]' '[:upper:]')
          echo "#define ${MNAME} ${VNAME}" >>${OFILE}
          echo "const char ${VNAME}[] =" >>${OFILE}
          echo "  \"#pragma OPENCL EXTENSION all: enable\\n\"" >>${OFILE}
          if [ "${CPP}" ] && \
             [ "$(${CPP} -P -fpreprocessed ${IFILE} 2>/dev/null >/dev/null && echo OK)" ];
          then
            ${CPP} -P -fpreprocessed ${IFILE}
          else # fallback to sed
            ${SED} -r ':a;s%(.*)/\*.*\*/%\1%;ta;/\/\*/!b;N;ba' ${IFILE}
          fi | \
          ${SED} \
            -e '/^[[:space:]]*$/d' -e 's/[[:space:]]*$//' \
            -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/  "/' -e 's/$/\\n"/' \
            >>${OFILE}
          echo ";" >>${OFILE}
        else
          >&2 echo "ERROR: ${IFILE} does not exist or is no OpenCL file!"
          rm -f ${OFILE}
          exit 1
        fi
      fi
    done
  else
    echo "Usage: $0 infile1.cl infile2.cl .. infilen.cl outfile.h"
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
