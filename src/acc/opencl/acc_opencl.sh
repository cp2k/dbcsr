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
SORT=$(command -v sort)
SED=$(command -v gsed)
CPP=$(command -v cpp)
TR=$(command -v tr)
RM=$(command -v rm)

# flags used to control preprocessor
CPPBASEFLAGS="-dD -P -fpreprocessed"

# delimiters allowed in CSV-file
DELIMS=";,\t|/"
IFS=$'\n'

# GNU sed is desired (macOS)
if [ "" = "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "${BASENAME}" ] && [ "${SORT}" ] && [ "${SED}" ] && [ "${TR}" ] && [ "${RM}" ]; then
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
            UNAME=$(echo "${BNAME}" | ${TR} '[:lower:]' '[:upper:]')
            SNAME=OPENCL_LIBSMM_STRING_${UNAME}
            VNAME=opencl_libsmm_source_${BNAME}
            MNAME=OPENCL_LIBSMM_SOURCE_${UNAME}
            echo "#define ${MNAME} ${VNAME}" >>"${OFILE}"
            echo "#define ${SNAME} \\" >>"${OFILE}"
            if [ "${CPP}" ] && \
               [ "$(eval "${CPP} ${CPPBASEFLAGS} ${IFILE}" 2>/dev/null >/dev/null && echo "YES")" ];
            then
              if [ "" != "${CPPFLAGS}" ] && \
                 [ "$(eval "${CPP} ${CPPFLAGS} ${CPPBASEFLAGS} ${IFILE}" 2>/dev/null >/dev/null && echo "YES")" ];
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
            echo "static const char ${VNAME}[] = ${SNAME};" >>"${OFILE}"
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
            DNAME=opencl_libsmm_params_dev
            MNAME=$(echo "${VNAME}" | ${TR} '[:lower:]' '[:upper:]')
            DEVCOL=$(${TR} '[:lower:]' '[:upper:]' <"${IFILE}" | ${SED} -n "1 s/^[[:space:]]*DEVICE[[:space:]]*${SEPAR}..*$/YES/p")
            DEVPAT="s/${SEPAR}..*//"
            echo "#define ${MNAME} ${VNAME}" >>"${OFILE}"
            echo "#define ${SNAME} \\" >>"${OFILE}"
            if [ "${DEVCOL}" ]; then
              DEVICES=$(${SED} "1d;${DEVPAT}" "${IFILE}" | ${SORT} -u)
            else
              DNAME=NULL
            fi
            while read -r LINE; do
              if [ "${DEVCOL}" ]; then
                I=0; IDEVICE=$(echo "${LINE}" | ${SED} "${DEVPAT}")
                for DEVICE in ${DEVICES}; do
                  if [ "${DEVICE}" = "${IDEVICE}" ]; then break; fi
                  I=$((I+1));
                done
                echo "${LINE}" | ${SED} "s/[^${SEPAR}]*//;s/^/  \"${I}/" >>"${OFILE}"
              else
                echo "${LINE}" | ${SED} "s/^/  \"-1${SEPAR}/" >>"${OFILE}"
              fi
            done < <(${SED} "1d;s/[\r]*$/\\\n\" \\\/" "${IFILE}")
            echo "  \"\"" >>"${OFILE}"
            echo "static const char ${VNAME}[] = ${SNAME};" >>"${OFILE}"
            echo >>"${OFILE}"
            echo "#define OPENCL_LIBSMM_PARAMS_DEVICES ${DNAME}" >>"${OFILE}"
            if [ "${DEVCOL}" ]; then
              echo "static const char *const ${DNAME}[] = {" >>"${OFILE}"
              NDEVICES=$(echo "${DEVICES}" | wc -l); I=0; S=","
              for DEVICE in ${DEVICES}; do
                I=$((I+1)); if [ "0" != "$((NDEVICES==I))" ]; then S=""; fi
                echo "  \"${DEVICE}\"${S}" >>"${OFILE}"
              done
              echo "};" >>"${OFILE}"
            fi
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
