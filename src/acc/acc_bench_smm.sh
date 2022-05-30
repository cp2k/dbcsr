#!/usr/bin/env bash
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)
SED=$(command -v gsed)

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "${SED}" ] && [ -x "${HERE}/acc_bench_smm" ]; then
  LIBXSMM_PEXEC=${LIBXSMM_ROOT:-${HOME}/libxsmm}/scripts/tool_pexec.sh
  if [ -x "${LIBXSMM_PEXEC}" ] && [ -e "$1" ]; then
    NDEVICES=$(ACC_OPENCL_VERBOSE=1 CHECK=0 "${HERE}/acc_bench_smm" 1 1 1 2>&1 >/dev/null \
             | ${SED} -n "s/INFO ACC\/OpenCL: ndevices=\([0-9][0-9]*\) ..*$/\1/p")
  fi
  if [ "${NDEVICES}" ] && [ "0" != "$((1<NDEVICES))" ]; then
    NLINES=0
    while read -r LINE; do
      echo "ACC_OPENCL_DEVICE=$((NLINES%NDEVICES)) ${HERE}/acc_bench_smm ${LINE}"
      NLINES=$((NLINES+1))
    done <"$1" | ${LIBXSMM_PEXEC} "${NDEVICES}"
  else
    "${HERE}/acc_bench_smm" "$*"
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
