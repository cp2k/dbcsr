#!/usr/bin/env bash
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: BSD-3-Clause                                                            #
####################################################################################################

HERE="$(cd "$(dirname "$0")" && pwd -P)"
TEST=acc_bench_smm
EXE=${HERE}/../${TEST}

if [ ! -e "$1" ]; then
  >&2 echo "USAGE: $0 logfile"
  exit 1
fi
if [ ! -e "${EXE}" ]; then
  >&2 echo "ERROR: please build ${TEST}!"
  exit 1
fi

sed -n "s/FAILED: \(..*\)/\1/p" "$1" | while read -r LINE; do
  EXPORT=""
  for KEYVAL in ${LINE}; do
    EXPORT="${EXPORT} OPENCL_LIBSMM_SMM_${KEYVAL}"
  done
  if [ "${EXPORT}" ]; then
    eval "${EXPORT} ${EXE}"
  fi
done
