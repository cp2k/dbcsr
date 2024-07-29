#!/usr/bin/env bash
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: BSD-3-Clause                                                            #
####################################################################################################
XARGS=$(command -v xargs)
SORT=$(command -v sort)
HEAD=$(command -v head)
SED=$(command -v gsed)
CUT=$(command -v cut)

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "${XARGS}" ] && [ "${SORT}" ] && [ "${HEAD}" ] && [ "${SED}" ] && [ "${CUT}" ]; then
  LINES=0
  while test $# -gt 0; do
    case "$1" in
    -h|--help)
      HELP=1
      shift $#;;
    -l|--lines)
      LINES=1
      shift;;
    -r|--bound)
      BOUNDL=$2
      BOUNDU=$3
      shift 3;;
    -m|--limit)
      MAXEXT=$2
      shift 2;;
    -n|--triplets)
      MAXNUM=$2
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
    -k|--specid)
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
  if [[ "${TRIPLETS}" && (! "${HELP}" || "0" = "${HELP}") ]]; then
    for SPECS in $(echo "${TRIPLETS}" | ${SED} -e "s/[[:space:]][[:space:]]*/x/g" -e "s/,/ /g"); do
      SPEC=$(echo "${SPECS}" | ${SED} -e "s/^x//g" -e "s/x$//g" -e "s/x/,/g")
      if [ "${MAXEXT}" ] && [ "0" != "$((0<MAXEXT))" ]; then
        for EXT in $(echo "${SPEC}" | ${SED} "s/,/ /g"); do
          if [ "0" != "$((MAXEXT<EXT))" ]; then continue 2; fi
        done
      fi
      MNKS="${MNKS} $(eval printf "%s" "{${SPEC}}x{${SPEC}}x{${SPEC}}\" \"" | ${SED} -e "s/{//g" -e "s/}//g")"
    done
    if [ "${MNKS}" ]; then
      if [ "${BOUNDL}" ] || [ "${BOUNDU}" ]; then
        if [ ! "${BOUNDL}" ]; then BOUNDL=0; elif [ ! "${BOUNDU}" ]; then BOUNDU=0; fi
        if [ "0" != "$((0<=BOUNDL))" ]; then
          for MNK in $(echo "${MNKS}" | ${SED} "s/x/*/g"); do
            S=$((MNK))
            if [ "0" != "$((BOUNDL<BOUNDU))" ]; then
              if [ "0" != "$((BOUNDL**3<S&&S<=BOUNDU**3))" ]; then TMP="${TMP} ${MNK}"; fi
            else
              if [ "0" != "$((BOUNDL**3<S))" ]; then TMP="${TMP} ${MNK}"; fi
            fi
          done
          MNKS=$(echo "${TMP}" | ${SED} "s/*/x/g")
        fi
      fi
      if [ "${CUTSEL}" ]; then
        MNK=$(echo "${MNKS}" | ${XARGS} -n1 | ${CUT} -dx ${CUTSEL} | ${SORT} -u -n -tx -k1,1 -k2,2 -k3,3 | \
          if [ "${MAXNUM}" ] && [ "0" != "$((0<MAXNUM))" ]; then ${HEAD} -n"${MAXNUM}"; else cat; fi)
      else
        MNK=$(echo "${MNKS}" | ${XARGS} -n1 | ${SORT} -u -n -tx -k1,1 -k2,2 -k3,3 | \
          if [ "${MAXNUM}" ] && [ "0" != "$((0<MAXNUM))" ]; then ${HEAD} -n"${MAXNUM}"; else cat; fi)
      fi
      if [ "0" = "${LINES}" ]; then
        echo "${MNK}" | ${XARGS}
      else
        echo "${MNK}"
      fi
    fi
  else
    if [ ! "${HELP}" ] || [ "0" = "${HELP}" ]; then
      ECHO=">&2 echo"
    else
      ECHO="echo"
    fi
    eval "${ECHO} \"Usage: $0 [options] [<triplet-spec>]\""
    eval "${ECHO} \"       Options must precede triplet specification\""
    eval "${ECHO} \"       -l|--lines: lines instead of list of words\""
    eval "${ECHO} \"       -r|--bound L U: limit L**3 < MNK <= U**3\""
    eval "${ECHO} \"       -m|--limit N: limit any shape extent to N\""
    eval "${ECHO} \"       -n|--triplets N: limit number of triplet\""
    eval "${ECHO} \"       -a|--amat: select MxK instead of MxNxK\""
    eval "${ECHO} \"       -b|--bmat: select KxN instead of MxNxK\""
    eval "${ECHO} \"       -c|--cmat: select MxN instead of MxNxK\""
    eval "${ECHO} \"       -k|--specid N: predefined triplets\""
    eval "${ECHO} \"        0-10: older to newer (larger), e.g.,\""
    eval "${ECHO} \"       -k  0:  201 kernels\""
    eval "${ECHO} \"       -k 10: 1266 kernels\""
    eval "${ECHO} \"       <triplet-spec>, e.g., 134 kernels\""
    eval "${ECHO} \"         23, 5 32 13 24 26, 4 9\""
    eval "${ECHO}"
    if [ "${HELP}" ] || [ "0" = "${HELP}" ]; then exit 0; fi
    >&2 echo "ERROR: invalid or no <triplet-spec> given!"
    exit 1
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
