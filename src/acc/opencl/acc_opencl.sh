#!/usr/bin/env bash
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################
# shellcheck disable=SC2048,SC2129

BASENAME=$(command -v basename)
DIRNAME=$(command -v dirname)
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

trap_exit() {
  if [ "0" != "$?" ] && [ "${HFILE}" ]; then ${RM} -f "${OFILE}"; fi
}

process_pre() {
  if [ "${CPP}" ] && \
     [ "$(eval "${CPP} ${CPPBASEFLAGS} $1" 2>/dev/null >/dev/null && echo "YES")" ];
  then
    if [ "${CPPFLAGS}" ] && \
       [ "$(eval "${CPP} ${CPPFLAGS} ${CPPBASEFLAGS} $1" 2>/dev/null >/dev/null && echo "YES")" ];
    then
      eval "${CPP} ${CPPFLAGS} ${CPPBASEFLAGS} $1" 2>/dev/null
    else
      eval "${CPP} ${CPPBASEFLAGS} $1" 2>/dev/null
    fi
  else # fallback to sed
    ${SED} -r ':a;s%(.*)/\*.*\*/%\1%;ta;/\/\*/!b;N;ba' "$1"
  fi
}

process() {
  IFS=$'\n'
  while read -r LINE; do
    INCLUDE=$(${SED} -n "s/#[[:space:]]*include[[:space:]][[:space:]]*\"/\"/p" <<<"${LINE}")
    if [ "${INCLUDE}" ] && [ "$1" ] && [ -e "$1" ]; then
      CLINC=$(${SED} "s/\"//g" <<<"${INCLUDE}")
      CLPATH=$(${DIRNAME} "$1")
      FILE=${CLPATH}/${CLINC}
      if [ "${FILE}" ] && [ -e "${FILE}" ]; then
        process_pre "${FILE}" | process "${FILE}"
      else
        >&2 echo "ERROR: header file ${FILE} not found!"
        exit 1
      fi
    else
      ${SED} <<<"${LINE}" \
        -e '/^[[:space:]]*$/d' -e 's/[[:space:]]*$//' \
        -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/  "/' -e 's/$/\\n" \\/'
    fi
  done
  unset IFS
}

if [ "${BASENAME}" ] && [ "${DIRNAME}" ] && [ "${SORT}" ] && \
   [ "${SED}" ] && [ "${TR}" ] && [ "${RM}" ] && [ "${WC}" ];
then
  for OFILE in "$@"; do :; done
  while test $# -gt 0; do
    case "$1" in
    -h|--help)
      shift $#;;
    -p|--params)
      PARAMS="$2\t"
      shift 2;;
    -c|-d|--debug|--comments)
      CPPFLAGS+=" -C"
      shift;;
    -v|--verbose)
      VERBOSE=1
      shift;;
    *) break;;
    esac
  done
  HERE="$(cd "$(${DIRNAME} "$0")" && pwd -P)"
  PARAMDIR=${PARAMDIR:-${PARAMS}}
  PARAMDIR=${PARAMDIR:-${HERE}/smm/params}
  PARAMDIR=$(echo -e "${PARAMDIR}" | ${TR} -d '\t')
  if [ "$#" -gt 1 ]; then
    # allow for instance /dev/stdout
    if [ "${OFILE##*.}" = "h" ]; then
      if [ "${VERBOSE}" ] && [ "0" != "${VERBOSE}" ]; then
        echo "$0 $*" # stdout
      fi
      truncate -s0 "${OFILE}"
      HFILE=${OFILE}
    elif [ "${OFILE##*.}" = "cl" ] || [ "${OFILE##*.}" = "csv" ]; then
      >&2 echo "ERROR: no output/header file given!"
      exit 1
    elif [ "${VERBOSE}" ] && [ "0" != "${VERBOSE}" ]; then
      if [[ ${OFILE} != /dev/stderr ]]; then
        >&2 echo "$0 $*"
      else # stdout
        echo "$0 $*"
      fi
    fi
    trap 'trap_exit' EXIT
    NFILES_OCL=0
    for CLFILE in ${*:1:${#@}-1}; do
      if [ "${CLFILE##*.}" = "cl" ]; then
        if [ -e "${CLFILE}" ]; then
          BNAME=$(${BASENAME} "${CLFILE}" .cl)
          UNAME=$(${TR} '[:lower:]' '[:upper:]' <<<"${BNAME}")
          SNAME=OPENCL_LIBSMM_STRING_${UNAME}
          VNAME=opencl_libsmm_source_${BNAME}
          MNAME=OPENCL_LIBSMM_SOURCE_${UNAME}
          if [ "0" != "$((0<(NFILES_OCL)))" ]; then
            echo >>"${OFILE}"
          fi
          echo "#define ${MNAME} ${VNAME}" >>"${OFILE}"
          echo "#define ${SNAME} \\" >>"${OFILE}"
          process_pre "${CLFILE}" | process "${CLFILE}" >>"${OFILE}"
          echo "  \"\"" >>"${OFILE}"
          echo "static const char ${VNAME}[] = ${SNAME};" >>"${OFILE}"
          NFILES_OCL=$((NFILES_OCL+1))
        else
          >&2 echo "ERROR: ${CLFILE} does not exist!"
          exit 1
        fi
      else
        CSVFILES=("${*:NFILES_OCL+1:${#@}-NFILES_OCL-1}")
        break
      fi
    done
    if [ "0" = "${NFILES_OCL}" ]; then
      >&2 echo "ERROR: no OpenCL file was given!"
      exit 1
    fi
    NFILES_CSV=0
    for CSVFILE in "${CSVFILES[@]}"; do
      if [ "${CSVFILE##*.}" = "csv" ]; then
        if [ -f "${CSVFILE}" ]; then
          NFILES_CSV=$((NFILES_CSV+1))
        fi
      else
        >&2 echo "ERROR: ${CSVFILE} is not a CSV file!"
        exit 1
      fi
    done
    if [ "0" = "${NFILES_CSV}" ] && [ "${PARAMDIR}" ] && [ -d "${PARAMDIR}" ]; then
      CSVFILES=("${PARAMDIR}"/*.csv)
      NFILES_CSV=${#CSVFILES[@]}
    fi
    for CSVFILE in "${CSVFILES[@]}"; do
      if [ ! "${DELIM}" ]; then
        SEPAR=$(${SED} -n "1s/[^${DELIMS}]//gp" "${CSVFILE}" 2>/dev/null)
        DELIM=${SEPAR:0:1}
        MATCH=$(${SED} -n "1s/[^${DELIM}]//gp" "${CSVFILE}" 2>/dev/null)
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
      if [ "${ERRFILE}" ] && [ -f "${ERRFILE}" ]; then
        >&2 echo "ERROR: ${ERRFILE} is malformed!"
        exit 1
      fi
    done
    DEVPAT="s/${DELIM}..*//"
    DEVICES=$(for CSVFILE in "${CSVFILES[@]}"; do ${SED} "1d;/^[[:space:]]*$/d;${DEVPAT}" "${CSVFILE}"; done | ${SORT} -u)
    SNAME=OPENCL_LIBSMM_STRING_PARAMS_SMM
    VNAME=opencl_libsmm_params_smm
    DNAME=opencl_libsmm_devices
    MNAME=$(${TR} '[:lower:]' '[:upper:]' <<<"${VNAME}")
    NNAME=$(${TR} '[:lower:]' '[:upper:]' <<<"${DNAME}")
    if [ "${DEVICES}" ]; then
      echo >>"${OFILE}"
      echo "#define ${MNAME} ${VNAME}" >>"${OFILE}"
      echo "#define ${SNAME} \\" >>"${OFILE}"
      CSVLINES=$(for CSVFILE in "${CSVFILES[@]}"; do ${SED} "1d;/^[[:space:]]*$/d;s/[\r]*$/\\\n\" \\\/" "${CSVFILE}"; done)
      IFS=$'\n'
      for LINE in ${CSVLINES}; do
        I=0; IDEVICE=$(${SED} "${DEVPAT}" <<<"${LINE}")
        for DEVICE in ${DEVICES}; do
          if [ "${DEVICE}" = "${IDEVICE}" ]; then break; fi
          I=$((I+1));
        done
        ${SED} "s/[^${DELIM}]*//;s/^/  \"${I}/" <<<"${LINE}" >>"${OFILE}"
      done
      echo "  \"\"" >>"${OFILE}"
      echo "static const char ${VNAME}[] = ${SNAME};" >>"${OFILE}"
      echo >>"${OFILE}"
      echo "#define ${NNAME} ${DNAME}" >>"${OFILE}"
      echo "static const char *const ${DNAME}[] = {" >>"${OFILE}"
      I=0; S=","; NDEVICES=$(${WC} -l <<<"${DEVICES}")
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
    echo "       -v|--verbose: repeat command-line arguments"
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
