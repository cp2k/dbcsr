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

import sys
import os
from glob import glob
import re
import json
import argparse

sys.path.append("../")
from kernels.smm_acc_predict import descr_to_kernel  # noqa: E402

re_mnk = re.compile(r"tune_(\d+)x(\d+)x(\d+)")
re_winner = re.compile(r"\nWINNER: \d+ (.+)\n")
re_gflops = re.compile(r"# ([0-9.]+) GFlop/s")
re_errors = re.compile(r"Number of errors: (\d+)\n")

from dataclasses import dataclass

@dataclass
class awinner:
    value: str = ""
    missing: int = 0
    incomplete: int = 0
    n_errors: int = 0

# ===============================================================================
def main():
    winners = dict()

    n_errors = 0
    for d in glob("tune_*"):
        if not os.path.isdir(d):
            continue

        for exe_fn in glob(d + "/tune_*main.c*"):
            mnk = tuple([int(i) for i in re_mnk.search(exe_fn).groups()])
            if mnk not in winners:
                winners[mnk] = awinner()
            log_fn = exe_fn.replace("_main.cu", ".log").replace("_main.cpp", ".log")
            if not os.path.exists(log_fn):
                winners[mnk] = "log missing: " + log_fn
                print(
                    "WARNINGL: Missing log:",
                    log_fn,
                    ", please re-run (cd tune_mxnxk; sbatch tune_mxnxk.job)",
                )
                n_errors += 1
            else:
                n_errors += process_log(log_fn, mnk, winners)

    if n_errors > 0:
        print("WARNING: Found %d issues, check above messages." % n_errors)

    # Get kernel objects from list of strings
    kernels = [descr_to_kernel(kernel_descr.value) for kernel_descr in winners.values()]
    kernels_dict = dict(zip([(k.m, k.n, k.k) for k in kernels], kernels))
    new_file = "../parameters/parameters.json"
    with open(new_file, "w") as f:
        s = json.dumps(
            [
                kernels_dict[kernel].as_dict_for_parameters_json
                for kernel in sorted(kernels_dict.keys())
            ]
        )
        s = s.replace("}, ", "},\n")
        s = s.replace("[", "[\n")
        s = s.replace("]", "\n]")
        f.write(s)

    print("\n")
    print("Wrote", new_file)


# ===============================================================================
def process_log(log_fn, mnk, winners):
    print("Reading: " + log_fn)

    with open(log_fn) as f:
        content = f.read()

    m = re_errors.search(content)
    if not m:
        winners[mnk].incomplete += 1
        print(
            "WARNING: Found incomplete log:",
            log_fn,
            ", please re-run (cd tune_mxnxk; sbatch tune_mxnxk.job)",
        )
        return 1

    n_errors = int(m.group(1))
    if n_errors != 0:
        winners[mnk].n_errors += n_errors
        return 1

    old_gflops = 0.0
    m = re_gflops.search(winners[mnk].value)
    if m:
        old_gflops = float(m.group(1))

    new_winner = re_winner.search(content).group(1).strip().replace("GFlops", "GFlop/s")
    new_gflops = float(re_gflops.search(new_winner).group(1))

    if new_gflops > old_gflops:
        winners[mnk].value = new_winner
    return 0


# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Collect autotuning results: parse the log files contained in folders tune_*x*x*
        to determine the best kernel for each block size, and store the results in a
        file "parameters.json".

        This script is part of the workflow for autotuning optimal libsmm_acc parameters.
        For more details, see README.md.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    args = parser.parse_args()
    main()
