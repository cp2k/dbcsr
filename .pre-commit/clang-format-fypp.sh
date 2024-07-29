#!/usr/bin/env bash
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

# clang-format change FYPP directives, need to revert the changes.

function sed_darwin()
{
    sed -i "" "$@"
}

function sed_linux()
{
    sed -i "$@"
}

function main()
{
    local files=""
    for i in "$@"; do
    case $i in
      -*|--*)
      ;;
      *)
          files+="$i "
      ;;
    esac
    done

    clang-format "$@"
    # Fix FYPP directives
    uname="$(uname -s)"
    case "${uname}" in
        Darwin*)
            sed_fcn=sed_darwin
        ;;
        *)
            sed_fcn=sed_linux
        ;;
    esac

    for i in ${files}; do
        ${sed_fcn} -e '/\${$/ { N; s/\${\n[[:space:]]*/\${/; }' "$i"
        ${sed_fcn} -e 's/#[[:space:]]*: /#:/g' -e 's/} \$/}\$/g' "$i"
    done
}

main "$@"
