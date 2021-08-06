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
LS=$(command -v ls)
RM=$(command -v rm)
WC=$(command -v wc)
DELAY=12

# GNU sed is desired (macOS)
if [ "" = "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "$1" ]; then
  LIMIT=$1
  shift
fi
if [ "$1" ]; then
  NPARTS=$1
  shift
else
  NPARTS=1
fi
if [ "$1" ]; then
  PART=$1
  shift
else
  PART=1
fi

if [ "0" != "$((NPARTS<PART))" ]; then
  >&2 echo "ERROR: part-number ${PART} is larger than the requested ${NPARTS} parts!"
  exit 1
fi

if [ "${SED}" ] && [ "${LS}" ] && [ "${RM}" ] && [ "${WC}" ]; then
  echo "Usage: $0 [seconds-per-kernel [num-parts [part [<triplet-spec>]]]]"
  echo "       num-parts and part (one-based), e.g., 12 3"
  echo "         for this session being the 3rd of 12 sessions"
  echo "       <triplet-spec>, e.g., 134 kernels"
  echo "         23, 5 32 13 24 26, 4 9"
  echo
  if [ "$1" ]; then
    MNKS=$("${HERE}/../../acc_triplets.sh" "$@")
  else
    if [ "" != "${MAXEXT}" ]; then MAXEXT="-m ${MAXEXT}"; fi
    if [ "" != "${MAXNUM}" ]; then MAXNUM="-n ${MAXNUM}"; fi
    if [ "" == "${SPECID}" ]; then SPECID=10; fi
    MNKS=$("${HERE}/../acc_triplets.sh" -s ${SPECID} "${LIMIT}" "${MAXNUM}")
  fi
  NTRIPLETS=$(echo "${MNKS}" | wc -w)
  PARTSIZE=$(((NTRIPLETS+NPARTS-1)/NPARTS))
  PARTOFFS=$(((PART-1)*PARTSIZE))
  PARTSIZE=$((PARTSIZE<=(NTRIPLETS-PARTOFFS)?PARTSIZE:(NTRIPLETS-PARTOFFS)))
  if [ "0" != "$((NPARTS<=NTRIPLETS))" ]; then
    echo "Session ${PART} of ${NPARTS} part(s)."
  else
    echo "Session ${PART} of ${NPARTS} part(s). The problem is over-decomposed!"
  fi
  if [ "${LIMIT}" ]; then
    HRS=$((LIMIT*PARTSIZE/3600))
    MNS=$(((LIMIT*PARTSIZE-HRS*3600+59)/60))
    echo "Tuning ${PARTSIZE} kernels in this session will take about ${HRS}h${MNS}m."
    LIMIT="--stop-after=${LIMIT}"
  else
    echo "Tuning ${PARTSIZE} kernels will take an unknown time (no limit given)."
  fi
  NJSONS=$(${LS} -1 ./*.json 2>/dev/null | ${WC} -l)
  if [ "0" != "${NJSONS}" ]; then
    echo "Already found ${NJSONS} (unrelated?) JSON-files."
  elif [ -e tune_multiply.csv ]; then
    echo "No JSON file found but (unrelated?) tune_multiply.csv exists."
  fi
  SLEEP=$(command -v sleep)
  if [ "${DELAY}" ] && [ "${SLEEP}" ]; then
    echo
    echo "Tuning will start in ${DELAY} seconds. Hit CTRL-C to abort."
    ${SLEEP} ${DELAY}
  fi
  N=0
  for MNK in ${MNKS}; do
    if [ "0" != "$((PARTOFFS<=N))" ]; then
      TRIPLET=$(echo "${MNK}" | ${SED} "s/x/ /g")
      echo
      echo "Started auto-tuning ${MNK}-kernel..."
      # avoid mixing database of previous results into new session
      ${RM} -rf "${HERE}/opentuner.db"
      eval "${HERE}/tune_multiply.py ${TRIPLET} --no-dups ${LIMIT}"
      RESULT=$?
      # environment var. CONTINUE allows to proceed with next kernel
      # even if tune_multiply.py returned non-zero exit code
      if [[ ("0" != "${RESULT}") && \
            ("${CONTINUE}" = "" \
          || "${CONTINUE}" = "0" \
          || "${CONTINUE}" = "no" \
          || "${CONTINUE}" = "false") ]];
      then
        exit ${RESULT}
      fi
    fi
    N=$((N+1))
  done
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
