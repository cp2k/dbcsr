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
    """Given a list of kernels represented as dictionaries, return a string representing them as C++ vector of vectors
    using initializer lists"""
    kernels = sorted(kernels, key=lambda k: (k["m"], k["n"], k["k"]))
    out = ""
    init_list_line = "        {{{m:>2}, {n:>2}, {k:>2}}},\n"
    for k in kernels:
        out += init_list_line.format(m=k["m"], n=k["n"], k=k["k"])
    return out


# ===============================================================================
def main(
    dbcsr_base_dir,
    libsmm_acc_base_dir,
    test_template_dir,
    test_output_dir,
    gpu_version,
    nsamples,
):
    """
    Generate a performance test of libsmm_acc in the form of a CUDA or HIP file, using libsmm_acc_unittest_multiply.cpp.template
    as a template
    """

    # Read parameter file
    print("GPU version: {}".format(gpu_version))
    param_fn = os.path.join(
        libsmm_acc_base_dir,
        os.path.join("parameters", "parameters_{}.json".format(gpu_version)),
    )
    with open(param_fn, "r") as f:
        all_kernels = json.load(f)

    # Get the autotuned kernels to test
    autotuned_kernels = [k for k in all_kernels if k["source"] == "autotuned"]
    print("Found {:,} autotuned kernels".format(len(autotuned_kernels)))

    # Get the non-autotuned kernels to test
    predicted_kernels = [k for k in all_kernels if k["source"] != "autotuned"]
    print("Found {:,} predicted kernels".format(len(predicted_kernels)))
    num_predicted_kernels = len(predicted_kernels)
    if num_predicted_kernels > 0:
        if nsamples >= num_predicted_kernels:
            nsamples = num_predicted_kernels
        kernels_to_test_predicted = random.sample(predicted_kernels, nsamples)
    else:
        kernels_to_test_predicted = list()
    kernels_to_print = format_to_cpp(autotuned_kernels + kernels_to_test_predicted)

    # Print to test file
    file_template = os.path.join(
        test_template_dir, "libsmm_acc_unittest_multiply.cpp.template"
    )
    file_generate = os.path.join(test_output_dir, "libsmm_acc_unittest_multiply.cpp")
    with open(file_template, "r") as f:
        test = f.read()
    test = test.replace("[[UNITTEST_KERNELS_HERE]]", kernels_to_print.lstrip())
    with open(file_generate, "w") as f:
        f.write(test)
    print("Wrote {:,} test kernels to {}".format(len(kernels_to_print), file_generate))


# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Generate a performance test of libsmm_acc in the form of a CUDA or HIP file, using
        libsmm_acc_unittest_multiply.cpp.template as a template
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f", "--base_dir", metavar="DBCSRHOME", default="", help="DBCSR base directory"
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        metavar="OUTDIR",
        default="./tests",
        help="Directory in which to write the generated test files",
    )
    parser.add_argument(
        "-g",
        "--gpu_version",
        metavar="GPU_VERSION",
        default="P100",
        help="GPU card version, used to select the appropriate libsmm_acc parameters file",
    )
    parser.add_argument(
        "-n",
        "--nsamples",
        default=1000,
        help=(
            "Number of samples from the matrix sizes space 4 <= m,n,k <= 45 (except autotuned kernels)"
            " to sample for performance testing"
        ),
    )

    args = parser.parse_args()

    # Folders in/to which to read/write files
    libsmm_acc_base_dir = os.path.join(args.base_dir, "src/acc/libsmm_acc/")
    test_template_dir = os.path.join(args.base_dir, "tests")
    test_output_dir = os.path.join(args.base_dir, args.out_dir)

    main(
        args.base_dir,
        libsmm_acc_base_dir,
        test_template_dir,
        test_output_dir,
        args.gpu_version,
        args.nsamples,
    )
