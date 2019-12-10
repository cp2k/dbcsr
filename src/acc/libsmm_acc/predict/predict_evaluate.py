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

import re
import numpy as np
import argparse
from predict_helpers import (
    performance_gain,
    relative_performance_gain,
    plot_absolute_performance_gain,
    plot_relative_performance_gain,
    plot_performance_gains,
)


# ===============================================================================
def main(file, file_baseline):
    """
    This script is part of the workflow for predictive modelling of optimal libsmm_acc parameters.
    For more details, see predict.md

    Given a file containing the results of the LBSMM_ACC performance test, perform evaluation of the predictive model.
    """
    # ===============================================================================
    # Read optimal-parameter-prediction result file
    with open(file) as f:
        result_file = f.read().splitlines()
    results_predictive_model = read_result_file(result_file)

    # Read baseline result file
    with open(file_baseline) as f:
        result_file = f.read().splitlines()
    results_baseline = read_result_file(result_file)

    # ===============================================================================
    # Performance comparison quantities
    improved_over_baseline = dict(
        zip(
            sorted(results_predictive_model.keys()),
            [
                results_predictive_model[(m, n, k)] > results_baseline[(m, n, k)]
                for m, n, k in sorted(results_predictive_model.keys())
            ],
        )
    )
    perf_gain_over_baseline = performance_gain(
        results_baseline, results_predictive_model
    )
    rel_perf_gain_over_baseline = relative_performance_gain(
        results_baseline, results_predictive_model
    )

    # ===============================================================================
    # Print results
    header = "m, n, k: baseline perf. [Gflops], predictive model perf. [Gflops], performance gain [? ]"
    print(header)
    line = (
        "{m:>2}, {n:>2}, {k:>2}: {baseline_perf:>7.2f}, {predictive_model_perf:>7.2f}, "
        + "{performance_gain:>7.2f}, {better}"
    )
    for m, n, k in sorted(results_predictive_model.keys()):
        print(
            line.format(
                m=m,
                n=n,
                k=k,
                baseline_perf=results_baseline[(m, n, k)],
                predictive_model_perf=results_predictive_model[(m, n, k)],
                performance_gain=perf_gain_over_baseline[(m, n, k)],
                better=improved_over_baseline[(m, n, k)],
            )
        )

    print(
        "\nKernel performances improved by predictive model:",
        list(improved_over_baseline.values()).count(True),
        "/",
        len(results_predictive_model.keys()),
    )
    perf_gain_improved = [pg for pg in perf_gain_over_baseline.values() if pg > 0]
    print(
        "Mean performance gain amongst improved kernels: {:.2f} Gflops".format(
            np.mean(perf_gain_improved)
        )
    )

    print(
        "\nKernel performances reduced by predictive model:",
        list(improved_over_baseline.values()).count(False),
        "/",
        len(results_predictive_model.keys()),
    )
    perf_gain_deteriorated = [pg for pg in perf_gain_over_baseline.values() if pg < 0]
    print(
        "Mean performance loss amongst deteriorated kernels: {:.2f} Gflops".format(
            np.mean(perf_gain_deteriorated)
        )
    )

    print(
        "\nMean performance gain overall: {:.2f} Gflops".format(
            np.mean(list(perf_gain_over_baseline.values()))
        )
    )

    # ===============================================================================
    # Plot results (testing set: predictive modelling VS naïve)
    plot_absolute_performance_gain(
        perf_gain_over_baseline, "non-autotuned", "baseline", "predictive model"
    )
    plot_relative_performance_gain(
        rel_perf_gain_over_baseline, "non-autotuned", "baseline", "predictive model"
    )
    plot_performance_gains(
        results_predictive_model,
        results_baseline,
        "non-autotuned",
        "baseline",
        "predictive model",
    )


# ===============================================================================
def read_result_file(file):
    results = dict()
    result_line = re.compile(r"OK (\d+) x (\d+) x (\d+) GFlop/s (\d+(?:\.\d+)?)")
    for line in file:
        match = result_line.match(line)
        if match is not None:
            m = int(match.group(1))
            n = int(match.group(2))
            k = int(match.group(3))
            perf = float(match.group(4))
            results[(m, n, k)] = perf

    return results


# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Given a file containing the results of the LIBSMM_ACC performance test, perform evaluation of the predictive
        model.
        This script is part of the workflow for predictive modelling of optimal libsmm_acc parameters.
        For more details, see README.md.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--file",
        metavar="filename.out",
        type=str,
        default="",
        help="Result file to evaluate. Output of tests/libsmm_acc_timer_multiply.cpp",
    )
    parser.add_argument(
        "-n",
        "--file_baseline",
        metavar="filename.out",
        type=str,
        default="",
        help="Baseline performance file to compare against.",
    )

    args = parser.parse_args()
    main(args.file, args.file_baseline)
