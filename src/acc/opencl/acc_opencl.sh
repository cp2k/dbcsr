#!/usr/bin/env bash
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################
# shellcheck disable=SC2129

BASENAME=$(command -v basename)
SED=$(command -v gsed)
CPP=$(command -v cpp)
RM=$(command -v rm)

# flags used to control preprocessor
CPPBASEFLAGS="-dD -P -fpreprocessed"

# delimiters allowed in CSV-file
DELIMS=";,\t|/"

# GNU sed is desired (macOS)
if [ "" = "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "${BASENAME}" ] && [ "${SED}" ] && [ "${RM}" ]; then
  for OFILE in "$@"; do :; done
  while test $# -gt 0; do
    case "$1" in
    -h|--help)
      shift $#;;
    -c|-d|--debug|--comments)
      CPPFLAGS+=" -C"
      shift;;
    *) break;;
    esac
  done
  if [ "$#" -gt 1 ]; then
    # allow for instance /dev/stdout
    if [ "${OFILE##*.}" = "h" ]; then
      truncate -s0 "${OFILE}"
      HFILE=${OFILE}
    elif [ "${OFILE##*.}" = "cl" ] || [ "${OFILE##*.}" = "csv" ]; then
      >&2 echo "ERROR: no output/header file given!"
      exit 1
    fi
    NFILES_OCL=0
    NFILES_CSV=0
    for IFILE in "$@"; do
      if [ "${IFILE}" != "${OFILE}" ]; then
        if [ "0" != "$((0<(NFILES_OCL+NFILES_CSV)))" ]; then
          echo >>"${OFILE}"
        fi
        if [ "${IFILE##*.}" = "cl" ]; then
          if [ -e "${IFILE}" ]; then
            BNAME=$(${BASENAME} "${IFILE}" .cl)
            UNAME=$(echo "${BNAME}" | tr '[:lower:]' '[:upper:]')
            SNAME=OPENCL_LIBSMM_STRING_${UNAME}
            VNAME=opencl_libsmm_source_${BNAME}
            MNAME=OPENCL_LIBSMM_SOURCE_${UNAME}
            echo "#define ${MNAME} ${VNAME}" >>"${OFILE}"
            echo "#define ${SNAME} \\" >>"${OFILE}"
            if [ "${CPP}" ] && \
               [ "$(eval "${CPP} ${CPPBASEFLAGS} ${IFILE}" 2>/dev/null >/dev/null && echo "OK")" ];
            then
              if [ "" != "${CPPFLAGS}" ] && \
                 [ "$(eval "${CPP} ${CPPFLAGS} ${CPPBASEFLAGS} ${IFILE}" 2>/dev/null >/dev/null && echo "OK")" ];
              then
                eval "${CPP} ${CPPFLAGS} ${CPPBASEFLAGS} ${IFILE}" 2>/dev/null
              else
                eval "${CPP} ${CPPBASEFLAGS} ${IFILE}" 2>/dev/null
              fi
            else # fallback to sed
              ${SED} -r ':a;s%(.*)/\*.*\*/%\1%;ta;/\/\*/!b;N;ba' "${IFILE}"
            fi | \
            ${SED} \
              -e '/^[[:space:]]*$/d' -e 's/[[:space:]]*$//' \
              -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/  "/' -e 's/$/\\n" \\/' \
              >>"${OFILE}"
            echo "  \"\"" >>"${OFILE}"
            echo "const char ${VNAME}[] = ${SNAME};" >>"${OFILE}"
            NFILES_OCL=$((NFILES_OCL+1))
          else
            >&2 echo "ERROR: ${IFILE} does not exist!"
            if [ "${HFILE}" ]; then ${RM} -f "${OFILE}"; fi
            exit 1
          fi
        elif [ "${IFILE##*.}" = "csv" ]; then
          # non-existence does not trigger an error
          if [ -e "${IFILE}" ]; then
            DELIM=$(tr -cd "${DELIMS}" < "${IFILE}")
            SEPAR=${DELIM:0:1}
            SNAME=OPENCL_LIBSMM_STRING_PARAMS_SMM
            VNAME=opencl_libsmm_params_smm
            MNAME=$(echo "${VNAME}" | tr '[:lower:]' '[:upper:]')
            DEVCOL=0
            if [ "$(command -v tail)" ] && [ "$(command -v cut)" ] && \
               [ "$(command -v sort)" ] && [ "$(command -v wc)" ];
            then
              DEVICE=$(tail -n+2 "${IFILE}" | cut -d"${SEPAR}" -f1 | sort -u)
              if [ "$(echo "${DEVICE}" | ${SED} "s/[0-9]//g")" ]; then DEVCOL=1; fi
              if [ "0" = "${DEVCOL}" ] || [ "1" = "$(echo "${DEVICE}" | wc -l | ${SED} "s/[[:space:]]//g")" ]; then
                if [ "0" != "${DEVCOL}" ] && [ "${DEVICE}" ]; then
                  echo "#define OPENCL_LIBSMM_PARAMS_DEVICE \"${DEVICE}\"" >>"${OFILE}"
                else
                  echo "#define OPENCL_LIBSMM_PARAMS_DEVICE NULL" >>"${OFILE}"
                fi
              else
                >&2 echo "ERROR: ${IFILE} contains parameters for different devices!"
                if [ "${HFILE}" ]; then ${RM} -f "${OFILE}"; fi
                exit 1
              fi
            fi
            echo "#define ${MNAME} ${VNAME}" >>"${OFILE}"
            echo "#define ${SNAME} \\" >>"${OFILE}"
            if [ "0" != "${DEVCOL}" ]; then
              ${SED} "1d;s/^[^${SEPAR}]*${SEPAR}/  \"/;s/[\r]*$/\\\n\" \\\/" "${IFILE}" >>"${OFILE}"
            else
              ${SED} "1d;s/^/  \"/;s/[\r]*$/\\\n\" \\\/" "${IFILE}" >>"${OFILE}"
            fi
            echo "  \"\"" >>"${OFILE}"
            echo "const char ${VNAME}[] = ${SNAME};" >>"${OFILE}"
            NFILES_CSV=$((NFILES_CSV+1))
          fi
        else
          >&2 echo "ERROR: ${IFILE} is not an OpenCL or CSV file!"
          if [ "${HFILE}" ]; then ${RM} -f "${OFILE}"; fi
          exit 1
        fi
      fi
    done
    if [ "0" = "${NFILES_OCL}" ]; then
      >&2 echo "ERROR: no OpenCL file was given!"
      if [ "${HFILE}" ]; then ${RM} -f "${OFILE}"; fi
      exit 1
    elif [ "0" != "$((1<NFILES_CSV))" ]; then
      >&2 echo "ERROR: more than one CSV file was given!"
      if [ "${HFILE}" ]; then ${RM} -f "${OFILE}"; fi
      exit 1
    fi
  else
    echo "Usage: $0 infile.cl [infile2.cl .. infileN.cl] [infile.csv] outfile.h"
    echo "       At least one OpenCL file must be supplied."
    echo "       Parameters per CSV file are optional."
    echo "       The CSV file can be at any position."
    echo "       -c|-d|--debug|--comments: keep comments"
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
