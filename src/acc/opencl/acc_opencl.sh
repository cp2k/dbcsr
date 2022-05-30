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
WC=$(command -v wc)

# flags used to control preprocessor
CPPBASEFLAGS="-dD -P -fpreprocessed"

# delimiters allowed in CSV-file
DELIMS=";,\t|/"

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "${BASENAME}" ] && [ "${SORT}" ] && [ "${SED}" ] && \
   [ "${TR}" ] && [ "${RM}" ] && [ "${WC}" ];
then
  for OFILE in "$@"; do :; done
  while test $# -gt 0; do
    case "$1" in
    -h|--help)
      shift $#;;
    -p|--params)
      PARAMPATH=yes
      PARAMS=$2
      shift 2;;
    -c|-d|--debug|--comments)
      CPPFLAGS+=" -C"
      shift;;
    *) break;;
    esac
  done
  HERE="$(cd "$(dirname "$0")" && pwd -P)"
  PARAMDIR=$(if [ ! "${PARAMDIR}" ]; then echo "${HERE}/smm/params"; fi)
  if [ "${PARAMPATH}" ]; then
    PARAMPATH=${PARAMS}
  else
    HERE="$(cd "$(dirname "$0")" && pwd -P)"
    PARAMPATH=${PARAMDIR}
  fi
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
    for CLFILE in ${*:1:${#@}-1}; do
      if [ "${CLFILE##*.}" = "cl" ]; then
        if [ -e "${CLFILE}" ]; then
          BNAME=$(${BASENAME} "${CLFILE}" .cl)
          UNAME=$(echo "${BNAME}" | ${TR} '[:lower:]' '[:upper:]')
          SNAME=OPENCL_LIBSMM_STRING_${UNAME}
          VNAME=opencl_libsmm_source_${BNAME}
          MNAME=OPENCL_LIBSMM_SOURCE_${UNAME}
          if [ "0" != "$((0<(NFILES_OCL)))" ]; then
            echo >>"${OFILE}"
          fi
          echo "#define ${MNAME} ${VNAME}" >>"${OFILE}"
          echo "#define ${SNAME} \\" >>"${OFILE}"
          if [ "${CPP}" ] && \
             [ "$(eval "${CPP} ${CPPBASEFLAGS} ${CLFILE}" 2>/dev/null >/dev/null && echo "YES")" ];
          then
            if [ "" != "${CPPFLAGS}" ] && \
               [ "$(eval "${CPP} ${CPPFLAGS} ${CPPBASEFLAGS} ${CLFILE}" 2>/dev/null >/dev/null && echo "YES")" ];
            then
              eval "${CPP} ${CPPFLAGS} ${CPPBASEFLAGS} ${CLFILE}" 2>/dev/null
            else
              eval "${CPP} ${CPPBASEFLAGS} ${CLFILE}" 2>/dev/null
            fi
          else # fallback to sed
            ${SED} -r ':a;s%(.*)/\*.*\*/%\1%;ta;/\/\*/!b;N;ba' "${CLFILE}"
          fi | \
          ${SED} \
            -e '/^[[:space:]]*$/d' -e 's/[[:space:]]*$//' \
            -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/  "/' -e 's/$/\\n" \\/' \
            >>"${OFILE}"
          echo "  \"\"" >>"${OFILE}"
          echo "static const char ${VNAME}[] = ${SNAME};" >>"${OFILE}"
          NFILES_OCL=$((NFILES_OCL+1))
        else
          >&2 echo "ERROR: ${CLFILE} does not exist!"
          if [ "${HFILE}" ]; then ${RM} -f "${OFILE}"; fi
          exit 1
        fi
      else
        CSVFILES=("${*:NFILES_OCL+1:${#@}-NFILES_OCL-1}")
        break
      fi
    done
    if [ "0" = "${NFILES_OCL}" ]; then
      >&2 echo "ERROR: no OpenCL file was given!"
      if [ "${HFILE}" ]; then ${RM} -f "${OFILE}"; fi
      exit 1
    fi
    NFILES_CSV=0
    for CSVFILE in "${CSVFILES[@]}"; do
      if [ "${CSVFILE##*.}" = "csv" ]; then
        if [ -e "${CSVFILE}" ]; then
          NFILES_CSV=$((NFILES_CSV+1))
        fi
      else
        >&2 echo "ERROR: ${CSVFILE} is not a CSV file!"
        if [ "${HFILE}" ]; then ${RM} -f "${OFILE}"; fi
        exit 1
      fi
    done
    if [ "0" = "${NFILES_CSV}" ] && [ "${PARAMPATH}" ]; then
      CSVFILES=("${PARAMPATH}"/*.csv)
      NFILES_CSV=${#CSVFILES[@]}
    fi
    for CSVFILE in "${CSVFILES[@]}"; do
      if [ ! "${DELIM}" ]; then
        SEPAR=$(${SED} -n "1s/[^${DELIMS}]//gp" "${CSVFILE}")
        DELIM=${SEPAR:0:1}
        MATCH=$(${SED} -n "1s/[^${DELIM}]//gp" "${CSVFILE}")
      fi
      if [ "${DELIM}" ]; then
        CHECK=$(${SED} "/^[[:space:]]*$/d;s/[^${DELIM}]//g" "${CSVFILE}" | ${SORT} -u | ${SED} -n "0,/./p")
        if [ "0" != "$((${#MATCH}<${#CHECK}))" ]; then
          ERRFILE=${CSVFILES[0]}
        elif [ "${MATCH}" != "${CHECK}" ]; then
          ERRFILE=${CSVFILE}
        fi
      else
        ERRFILE=${CSVFILE}
      fi
      if [ "${ERRFILE}" ]; then
        >&2 echo "ERROR: ${ERRFILE} is malformed!"
        if [ "${HFILE}" ]; then ${RM} -f "${OFILE}"; fi
        exit 1
      fi
    done
    DEVPAT="s/${DELIM}..*//"
    DEVICES=$(for CSVFILE in "${CSVFILES[@]}"; do ${SED} "1d;/^[[:space:]]*$/d;${DEVPAT}" "${CSVFILE}"; done | ${SORT} -u)
    SNAME=OPENCL_LIBSMM_STRING_PARAMS_SMM
    VNAME=opencl_libsmm_params_smm
    DNAME=opencl_libsmm_devices
    MNAME=$(echo "${VNAME}" | ${TR} '[:lower:]' '[:upper:]')
    NNAME=$(echo "${DNAME}" | ${TR} '[:lower:]' '[:upper:]')
    if [ "${DEVICES}" ]; then
      echo >>"${OFILE}"
      echo "#define ${MNAME} ${VNAME}" >>"${OFILE}"
      echo "#define ${SNAME} \\" >>"${OFILE}"
      CSVLINES=$(for CSVFILE in "${CSVFILES[@]}"; do ${SED} "1d;/^[[:space:]]*$/d;s/[\r]*$/\\\n\" \\\/" "${CSVFILE}"; done)
      IFS=$'\n'
      for LINE in ${CSVLINES}; do
        I=0; IDEVICE=$(echo "${LINE}" | ${SED} "${DEVPAT}")
        for DEVICE in ${DEVICES}; do
          if [ "${DEVICE}" = "${IDEVICE}" ]; then break; fi
          I=$((I+1));
        done
        echo "${LINE}" | ${SED} "s/[^${DELIM}]*//;s/^/  \"${I}/" >>"${OFILE}"
      done
      echo "  \"\"" >>"${OFILE}"
      echo "static const char ${VNAME}[] = ${SNAME};" >>"${OFILE}"
      echo >>"${OFILE}"
      echo "#define ${NNAME} ${DNAME}" >>"${OFILE}"
      echo "static const char *const ${DNAME}[] = {" >>"${OFILE}"
      I=0; S=","; NDEVICES=$(echo "${DEVICES}" | ${WC} -l)
      for DEVICE in ${DEVICES}; do
        I=$((I+1)); if [ "0" != "$((NDEVICES==I))" ]; then S=""; fi
        echo "  \"${DEVICE}\"${S}" >>"${OFILE}"
      done
      unset IFS
      echo "};" >>"${OFILE}"
    fi
  else
    echo "Usage: $0 infile.cl [infile2.cl .. infileN.cl] [infile.csv [.. infileN.csv]] outfile.h"
    echo "       At least one OpenCL file and one header file must be supplied."
    echo "       -p|--params P: directory-path to CSV-files (can be \"\")"
    echo "             default: ${PARAMDIR}"
    echo "       -c|-d|--debug|--comments: keep comments in source-code"
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
