#!/usr/bin/env bash
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

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
    if [ "${OFILE##*.}" = "h" ]; then
      truncate -s0 ${OFILE}
    elif [ "${OFILE##*.}" = "cl" ] || [ "${OFILE##*.}" = "csv" ]; then
      >&2 echo "ERROR: no output/header file given!"
      exit 1
    fi
    NFILES_OCL=0
    NFILES_CSV=0
    for IFILE in $@; do
      if [ "${IFILE}" != "${OFILE}" ]; then
        if [ "${IFILE##*.}" = "cl" ]; then
          if [ -e ${IFILE} ]; then
            BNAME=$(${BASENAME} "${IFILE}" .cl)
            VNAME=opencl_libsmm_source_${BNAME}
            MNAME=$(echo ${VNAME} | tr '[:lower:]' '[:upper:]')
            echo "#define ${MNAME} ${VNAME}" >>${OFILE}
            echo "const char ${VNAME}[] =" >>${OFILE}
            echo "  \"#pragma OPENCL EXTENSION all: enable\\n\"" >>${OFILE}
            if [ "${CPP}" ] && \
               [ "$(${CPP} -dD -P -fpreprocessed ${IFILE} 2>/dev/null >/dev/null && echo OK)" ];
            then
              ${CPP} -dD -P -fpreprocessed ${IFILE}
            else # fallback to sed
              ${SED} -r ':a;s%(.*)/\*.*\*/%\1%;ta;/\/\*/!b;N;ba' ${IFILE}
            fi | \
            ${SED} \
              -e '/^[[:space:]]*$/d' -e 's/[[:space:]]*$//' \
              -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/  "/' -e 's/$/\\n"/' \
              >>${OFILE}
            echo ";" >>${OFILE}
            NFILES_OCL=$((NFILES_OCL+1))
          else
            >&2 echo "ERROR: ${IFILE} does not exist!"
            rm -f ${OFILE}
            exit 1
          fi
        elif [ "${IFILE##*.}" = "csv" ]; then
          # non-existence does not trigger an error
          if [ -e ${IFILE} ]; then
            VNAME=opencl_libsmm_params_smm
            MNAME=$(echo ${VNAME} | tr '[:lower:]' '[:upper:]')
            echo "#define ${MNAME} ${VNAME}" >>${OFILE}
            echo "const char ${VNAME}[] =" >>${OFILE}
            ${SED} 's/^/  "/;s/$/\\n"/;1d' ${IFILE} >>${OFILE}
            echo ";" >>${OFILE}
            NFILES_CSV=$((NFILES_CSV+1))
          fi
        else
          >&2 echo "ERROR: ${IFILE} is not an OpenCL or CSV file!"
          rm -f ${OFILE}
          exit 1
        fi
      fi
    done
    if [ "0" = "${NFILES_OCL}" ]; then
      >&2 echo "ERROR: no OpenCL file was given!"
      rm -f ${OFILE}
      exit 1
    elif [ "0" != "$((1<NFILES_CSV))" ]; then
      >&2 echo "ERROR: more than one CSV file was given!"
      rm -f ${OFILE}
      exit 1
    fi
  else
    echo "Usage: $0 infile.cl [infile2.cl .. infileN.cl] [infile.csv] outfile.h"
    echo "       At least one OpenCL file must be supplied."
    echo "       Parameters per CSV file are optional."
    echo "       The CSV file can be at any position."
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
