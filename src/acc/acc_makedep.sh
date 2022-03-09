#!/usr/bin/env bash
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

FILE=$1
VAL=$2

if [ "${FILE}" ]; then
  if [ ! -e "${FILE}" ] || [ "$(cat "${FILE}")" != "${VAL}" ]; then
    echo "${VAL}" >"${FILE}"
  fi
  echo "${FILE}"
else
  echo "Usage: $0 filename [value]"
  echo "  The content of the file will be updated with the value"
  echo "  if the value is different than the current value."
  echo "  This suitable to form a Makefile dependency."
fi


