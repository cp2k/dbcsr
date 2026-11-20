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


def sample_kernels(kernels, nsamples, rng):
    """Select a reproducible subset, treating zero as unlimited."""
    if nsamples < 0:
        raise ValueError("Kernel sample limits must be non-negative")
    if nsamples == 0 or nsamples >= len(kernels):
        return kernels
    return rng.sample(kernels, nsamples)


# ===============================================================================
def main(
    dbcsr_base_dir,
    libsmm_acc_base_dir,
    test_template_dir,
    test_output_dir,
    gpu_version,
    nsamples,
    max_autotuned,
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

    rng = random.Random(0)

    # Get the autotuned kernels to test
    autotuned_kernels = [k for k in all_kernels if k["source"] == "autotuned"]
    print("Found {:,} autotuned kernels".format(len(autotuned_kernels)))
    kernels_to_test_autotuned = sample_kernels(autotuned_kernels, max_autotuned, rng)

    # Get the non-autotuned kernels to test
    predicted_kernels = [k for k in all_kernels if k["source"] != "autotuned"]
    print("Found {:,} predicted kernels".format(len(predicted_kernels)))
    kernels_to_test_predicted = sample_kernels(predicted_kernels, nsamples, rng)
    kernels_to_test = kernels_to_test_autotuned + kernels_to_test_predicted
    kernels_to_print = format_to_cpp(kernels_to_test)

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
    print("Wrote {:,} test kernels to {}".format(len(kernels_to_test), file_generate))


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
        type=int,
        default=1000,
        help=(
            "Number of samples from the matrix sizes space 4 <= m,n,k <= 45 (except autotuned kernels)"
            " to sample for performance testing; zero means all"
        ),
    )
    parser.add_argument(
        "--max-autotuned",
        type=int,
        default=0,
        help="Maximum number of autotuned kernels to test; zero means all",
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
        args.max_autotuned,
    )
