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

import os
import random
import json
import argparse


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
def main(basedir, gpu_version, nsamples):
    """
    Generate a performance test of libcusmm in the form of a CUDA file, using libcusmm_timer_multiply.template
    as a template
    """

    # Read parameter file
    print("GPU version: {}".format(gpu_version))
    base_dir = os.path.join(basedir, "src/acc/libsmm_acc/libcusmm/")
    param_fn = os.path.join(base_dir, "parameters_{}.json".format(gpu_version))
    with open(param_fn, "r") as f:
        all_kernels = json.load(f)

    # Get the autotuned kernels to test
    autotuned_kernels = [k for k in all_kernels if k["source"] == "autotuned"]
    print("Found {:,} autotuned kernels".format(len(autotuned_kernels)))
    kernels_to_print_autotuned = format_to_cpp(autotuned_kernels)

    # Get the non-autotuned kernels to test
    predicted_kernels = [k for k in all_kernels if k["source"] != "autotuned"]
    print("Found {:,} predicted kernels".format(len(predicted_kernels)))
    kernels_to_test_predicted = random.sample(predicted_kernels, nsamples)
    kernels_to_print_predicted = format_to_cpp(kernels_to_test_predicted)

    # Print to test file
    test_directory = os.path.join(basedir, "tests")
    file_template = os.path.join(test_directory, "libcusmm_timer_multiply.template")
    file_generate = os.path.join(test_directory, "libcusmm_timer_multiply.cu")
    with open(file_template, "r") as f:
        test = f.read()
    test = test.replace("[[AUTOTUNED_KERNELS_HERE]]", kernels_to_print_autotuned.lstrip())
    test = test.replace("[[PREDICTED_KERNELS_HERE]]", kernels_to_print_predicted.lstrip())
    with open(file_generate, "w") as f:
        f.write(test)
    print("Wrote {:,} test kernels to".format(len(autotuned_kernels + kernels_to_test_predicted), file_generate))


# ===============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
        Generate a performance test of libcusmm in the form of a CUDA file, using libcusmm_timer_multiply.template
        as a template
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--base_folder", metavar="DBCSRHOME", default="")
    parser.add_argument(
        "-g",
        "--gpu_version",
        metavar="GPU_VERSION",
        default="P100",
        help="GPU card version, used to select the appropriate libcusmm parameters file",
    )
    parser.add_argument(
        "-n",
        "--nsamples",
        default=1000,
        help="Number of samples from the matrix sizes space 4 <= m,n,k <= 45 (except autotuned kernels)" +
        " to sample for performance testing",
    )

    args = parser.parse_args()
    main(args.base_folder, args.gpu_version, args.nsamples)
