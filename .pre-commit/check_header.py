#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

import argparse
import re
import mmap
import sys
import pathlib
from collections import defaultdict
from os import path, listdir
from contextlib import contextmanager

TYPES = {
    "c_cpp": [".c", ".h", ".cc", ".hh", ".cxx", ".hxx", ".cpp", ".hpp", ".cu", ".cl"],
    "fortran": [".F", ".f", ".f90", ".f03"],
    "script": [".py", ".sh"],
    "fypp": [".fypp"],
}

# max number of lines allowed between header and top of file
ALLOWED_LINES = 5

# some assumed max line length to terminate early for large files
MAX_LINE_LENGTH = 128


@contextmanager
def mmap_open(name, mode="r"):
    access = mmap.ACCESS_READ if mode == "r" else mmap.ACCESS_WRITE
    with open(name, mode + "b") as fhandle:
        fmapped = mmap.mmap(fhandle.fileno(), 0, access=access)
        yield fmapped
        fmapped.close()


def check_header(header_dir, files, verbose=False):
    retval = 0
    header_re = defaultdict(list)
    header_len = defaultdict(list)

    for headerfile in listdir(header_dir):
        headertype = pathlib.Path(headerfile).stem
        if headertype in TYPES:
            with open(path.join(header_dir, headerfile), "rb") as fhandle:
                header_content = fhandle.read()
                header_re[headertype].append(re.compile(re.escape(header_content)))
                header_len[headertype].append(len(header_content))
        else:
            print("no matching headerfile to file extensions")
            sys.exit(1)

    ext_map = {e: t for t, exts in TYPES.items() for e in exts}

    for fpath in files:
        _, fext = path.splitext(fpath)

        if fext not in ext_map:
            if verbose:
                print("? {} ... unknown file type, ignoring".format(fpath))
            continue

        with mmap_open(fpath) as fmapped:
            header_type = ext_map[fext]
            for h_re, h_len in zip(header_re[header_type], header_len[header_type]):
                match = h_re.search(fmapped, 0, ALLOWED_LINES * MAX_LINE_LENGTH + h_len)
                if match:
                    break

            if not match:
                print("✗ {} ... required header not found".format(fpath))
                retval = 1
                continue

            lines_above = fmapped[0 : match.start()].splitlines()
            if len(lines_above) > ALLOWED_LINES:
                print(
                    "✗ {} ... header not within first {} lines".format(
                        fpath, ALLOWED_LINES
                    )
                )
                retval = 1
                continue

        if verbose:
            print("✓ {}".format(fpath))

    sys.exit(retval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check files for header presence")
    parser.add_argument(
        "files", metavar="FILE", type=str, nargs="+", help="files to check"
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args()

    header_dir = path.join(path.dirname(path.abspath(__file__)), "headers")
    check_header(header_dir, args.files, args.verbose)
