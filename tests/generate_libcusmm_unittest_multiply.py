#!/usr/bin/env python
# -*- coding: utf-8 -*-
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

import sys
import os
import random
import json
from optparse import OptionParser


def format_to_cpp(kernels):
    """Given a list of Kernels represented as dictionaries, return a string representing them as C++ vector of vectors
    using initializer lists"""
    kernels = sorted(kernels, key=lambda k: (k["m"], k["n"], k["k"]))
    out = ""
    init_list_line = "        {{{m:>2}, {n:>2}, {k:>2}}},\n"
    for k in kernels:
        out += init_list_line.format(m=k["m"], n=k["n"], k=k["k"])
    return out


# ===============================================================================
def main():
    """
    Generate a performance test of libcusmm in the form of a CUDA file, using libcusmm_timer_multiply.template
    as a template
    """
    parser = OptionParser()
    parser.add_option("-f", "--base_folder", metavar="DBCSRHOME", default="", help="Default: %default")
    parser.add_option(
        "-g",
        "--gpu_version",
        metavar="GPU_VERSION",
        default="P100",
        help="GPU card version, used to select the appropriate libcusmm parameters file. Default: %(default)s",
    )
    parser.add_option(
        "-n",
        "--nsamples",
        default=1000,
        help="Number of samples from the matrix sizes space 4 <= m,n,k <= 45 (except autotuned kernels)" +
        " to sample from the 'predicted' kernels for unit testing. Default: %default",
    )

    options, args = parser.parse_args(sys.argv)

    # Read parameter file
    print("GPU version: {}".format(options.gpu_version))
    base_dir = os.path.join(options.base_folder, "src/acc/libsmm_acc/libcusmm/")
    param_fn = os.path.join(base_dir, "parameters_{}.json".format(options.gpu_version))
    with open(param_fn, "r") as f:
        all_kernels = json.load(f)

    # Get the autotuned kernels to test
    autotuned_kernels = [k for k in all_kernels if k["source"] == "autotuned"]
    print("Found {:,} autotuned kernels".format(len(autotuned_kernels)))

    # Get the non-autotuned kernels to test
    predicted_kernels = [k for k in all_kernels if k["source"] != "autotuned"]
    print("Found {:,} predicted kernels".format(len(predicted_kernels)))
    kernels_to_test_predicted = random.sample(predicted_kernels, options.nsamples)
    kernels_to_print = format_to_cpp(autotuned_kernels + kernels_to_test_predicted)

    # Print to test file
    test_directory = os.path.join(options.base_folder, "tests")
    file_template = os.path.join(test_directory, "libcusmm_unittest_multiply.template")
    file_generate = os.path.join(test_directory, "libcusmm_unittest_multiply.cu")
    with open(file_template, "r") as f:
        test = f.read()
    test = test.replace("[[UNITTEST_KERNELS_HERE]]", kernels_to_print.lstrip())
    with open(file_generate, "w") as f:
        f.write(test)
    print("Wrote {:,} test kernels to {}".format(len(kernels_to_print), file_generate))


# ===============================================================================
main()
