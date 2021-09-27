#!/usr/bin/env bash
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################
XARGS=$(command -v xargs)
SORT=$(command -v sort)
HEAD=$(command -v head)
SED=$(command -v gsed)
CUT=$(command -v cut)

# GNU sed is desired (macOS)
if [ "" = "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "${XARGS}" ] &&  [ "${SORT}" ] && [ "${HEAD}" ] && [ "${SED}" ] && [ "${CUT}" ]; then
  LINES=0
  while test $# -gt 0; do
    case "$1" in
    -h|--help)
      shift $#;;
    -l|--lines)
      LINES=1
      shift;;
    -r|--bound)
      BOUNDL=$2
      BOUNDU=$3
      shift 3;;
    -m|--limit)
      LIMIT=$2
      shift 2;;
    -n|--size)
      SIZE=$2
      shift 2;;
    -a|--amat)
      CUTSEL=-f1,3
      shift;;
    -b|--bmat)
      CUTSEL=-f3,2
      shift;;
    -c|--cmat)
      CUTSEL=-f1,2
      shift;;
    -s|--specid)
      case "$2" in
      0) TRIPLETS="23, 6, 14 16 29, 5 16 13 24 26, 9 16 22, 32, 64, 78, 16 29 55";;
      1) TRIPLETS="23, 6, 14 16 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55";;
      2) TRIPLETS="23, 6, 14 16 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 12";;
      3) TRIPLETS="23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 32 29 55, 12";;
      4) TRIPLETS="23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 32 29 55, 12, 13 26 28 32 45";;
      5) TRIPLETS="23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 32 29 55, 12, 13 26 28 32 45, 7 13 25 32";;
      6) TRIPLETS="23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 64, 78, 16 29 55, 32 29 55, 12, 4 5 7 9 13 25 26 28 32 45";;
      7) TRIPLETS="23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 64, 78, 16 29 55, 32 29 55, 12, 4 5 7 9 13 25 26 28 32 45, 4 10";;
      8) TRIPLETS="23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 64, 78, 16 29 55, 32 29 55, 12, 4 5 7 9 13 25 26 28 32 45, 4 10 15";;
      9) TRIPLETS="23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 64, 78, 16 29 55, 32 29 55, 12, 4 5 7 9 13 25 26 28 32 45, 4 10 15, 6 7 8";;
      *) TRIPLETS=" \
          4 5 7 9 13 25 26 28 32 45, \
          13 14 25 26 32, \
          5 32 13 24 26, \
          14 16 29, \
          14 32 29, \
          16 29 55, \
          32 29 55, \
          9 32 22, \
          4 10 15, \
          6 7 8, \
          23, \
          64, \
          78, \
          12, \
          6";;
      esac
      shift 2;;
    *)
      if [ "" = "$(echo "$*" | ${SED} -n "s/[0-9]*[[:space:]]*,*//gp")" ]; then
        TRIPLETS="$*"
      else
        >&2 echo "ERROR: invalid triplet specification!"
      fi
      break;;
    esac
  done
  if [ "${TRIPLETS}" ]; then
    for SPECS in $(echo "${TRIPLETS}" | ${SED} -e "s/[[:space:]][[:space:]]*/x/g" -e "s/,/ /g"); do
      SPEC=$(echo "${SPECS}" | ${SED} -e "s/^x//g" -e "s/x$//g" -e "s/x/,/g")
      if [ "${LIMIT}" ] && [ "0" != "$((0<LIMIT))" ]; then
        for EXT in $(echo "${SPEC}" | ${SED} "s/,/ /g"); do
          if [ "0" != "$((LIMIT<EXT))" ]; then continue 2; fi
        done
      fi
      MNKS="${MNKS} $(eval printf "%s" "{${SPEC}}x{${SPEC}}x{${SPEC}}\" \"" | ${SED} -e "s/{//g" -e "s/}//g")"
    done
    if [ "${MNKS}" ]; then
      if [ "${BOUNDL}" ] && [ "${BOUNDU}" ]; then
        for MNK in $(echo "${MNKS}" | ${SED} "s/x/*/g"); do
          S=$((MNK))
          if [ "0" != "$((BOUNDL**3<S&&S<=BOUNDU**3))" ]; then
            TMP="${TMP} ${MNK}"
          fi
        done
        MNKS=$(echo "${TMP}" | ${SED} "s/*/x/g")
      fi
      if [ "${CUTSEL}" ]; then
        MNK=$(echo "${MNKS}" | ${XARGS} -n1 | ${CUT} -dx ${CUTSEL} | ${SORT} -u -n -tx -k1 -k2 -k3 | \
          if [ "0" != "$((0<SIZE))" ]; then ${HEAD} -n"${SIZE}"; else cat; fi)
      else
        MNK=$(echo "${MNKS}" | ${XARGS} -n1 | ${SORT} -u -n -tx -k1 -k2 -k3 | \
          if [ "0" != "$((0<SIZE))" ]; then ${HEAD} -n"${SIZE}"; else cat; fi)
      fi
      if [ "0" = "${LINES}" ]; then
        echo "${MNK}" | ${XARGS}
      else
        echo "${MNK}"
      fi
    fi
  else
    echo "Usage: $0 [options] [<triplet-spec>]"
    echo "       Options must precede triplet specification"
    echo "       -l|--lines: lines instead of list of words"
    echo "       -r|--bound L U: limit L**3 < MNK <= U**3"
    echo "       -m|--limit N: limit shape extents to N"
    echo "       -n|--size  N: limit number of elements"
    echo "       -a|--amat: select MxK instead of MxNxK"
    echo "       -b|--bmat: select KxN instead of MxNxK"
    echo "       -c|--cmat: select MxN instead of MxNxK"
    echo "       -s|--specid N: predefined triplets"
    echo "        0-10: older to newer (larger), e.g.,"
    echo "       -s  0:  201 kernels"
    echo "       -s 10: 1266 kernels"
    echo "       <triplet-spec>, e.g., 134 kernels"
    echo "         23, 5 32 13 24 26, 4 9"
    echo
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
