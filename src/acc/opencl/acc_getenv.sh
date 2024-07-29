#!/usr/bin/env bash
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: BSD-3-Clause                                                            #
####################################################################################################

FIND=$(command -v find)
SORT=$(command -v sort)
SED=$(command -v gsed)

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi

HERE="$(cd "$(dirname "$0")" && pwd -P)"
SRC="${HERE}"
EXT="c"

if [ "${FIND}" ] && [ "${SORT}" ] && [ "${SED}" ] && [ -d "${SRC}" ]; then
  export LC_ALL=C
  ENVARS="$(${FIND} "${SRC}" -type f -name "*.${EXT}" -exec \
    "${SED}" "s/getenv[[:space:]]*([[:space:]]*\".[^\"]*/\n&/g" {} \; | \
    "${SED}" -n "s/.*getenv[[:space:]]*([[:space:]]*\"\(.[^\"]*\)..*/\1/p" | \
     ${SORT} -u)"
  OTHERS=$(echo "${ENVARS}" | ${SED} "/ACC_OPENCL_/d;/OPENCL_LIBSMM_/d")
  if [ "${OTHERS}" ]; then
    echo "===================================="
    echo "Other environment variables"
    echo "===================================="
    echo "${ENVARS}" | ${SED} "/ACC_OPENCL_/d;/OPENCL_LIBSMM_/d"
  fi
  echo "===================================="
  echo "OpenCL Backend environment variables"
  echo "===================================="
  echo "${ENVARS}" | ${SED} -n "/ACC_OPENCL_/p"
  echo "===================================="
  echo "OpenCL LIBSMM environment variables"
  echo "===================================="
  echo "${ENVARS}" | ${SED} -n "/OPENCL_LIBSMM_/p"
else
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi
