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
import json
from optparse import OptionParser
from kernels.cusmm_dnt_helper import params_dict_to_kernel
from predict_helpers import *


# ===============================================================================
# Main
def main(argv):
    """
    Sample mnks among predicted and print in a format convenient for copy-pasting into libcusmm_timer_multiply
    """
    del argv  # unused

    parser = OptionParser()
    parser.add_option("-p", "--params", metavar="filename.json",
                      default="parameters_P100.json", help="Default: %default")
    parser.add_option("-f", "--file", metavar="filename.out",
                      #default="libcusmm_timer_multiply.10310667.o",  # 01.11, perf squared
                      #default="libcusmm_timer_multiply.10395288.o", # 07.11, scaled v0, DT (medium-only)
                      #default="libcusmm_timer_multiply.10406377.o",  # 08.11, scaled v0, RF-5 (medium-only)
                      default="libcusmm_timer_multiply_tiny.10408177.o", # 08.11, scaled v0, DT (tiny-only)
                      help="Real result file to evaluate. Default: %default")
    parser.add_option("-n", "--file_naive", metavar="filename.out",
                      #default="libcusmm_timer_multiply.10312765.o", # (testing pars)
                      default="libcusmm_timer_multiply_tiny.10408385.o", # (tiny-pars)
                      help="Naive result file to evaluate (testing mnks). Default: %default")  # naive test
    parser.add_option("-a", "--file_naive_autotuned", metavar="filename.out",
                      default="libcusmm_timer_autotuned_multiply.10314368.o",   # naive autotuned
                      help="Naive result file to evaluate (autotuned mnks). Default: %default")
    parser.add_option("-t", "--file_predicted_autotuned", metavar="filename.out",
                      default="libcusmm_timer_autotuned_multiply.10412158.o",   # predicted optimal on autotuned set
                      help="Naive result file to evaluate (autotuned mnks). Default: %default")
    options, args = parser.parse_args(sys.argv)

    # ===============================================================================
    # Read parameter file
    with open(options.params) as f:
        all_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]
    predicted_mnks = [(k.m, k.n, k.k) for k in all_kernels if not k.autotuned]
    predicted_perfs_ = [k.perf for k in all_kernels if not k.autotuned]
    predicted_perfs = dict(zip(predicted_mnks, predicted_perfs_))
    autotuned_mnks = [(k.m, k.n, k.k) for k in all_kernels if k.autotuned]
    autotuned_perfs_ = [k.perf for k in all_kernels if k.autotuned]
    autotuned_perfs = dict(zip(autotuned_mnks, autotuned_perfs_))
    autotuned_kernels = dict(zip(autotuned_mnks, [k for k in all_kernels if k.autotuned]))

    # Read optimal-parameter-prediction result file
    with open(options.file) as f:
        result_file = f.read().splitlines()
    results = read_result_file(result_file)

    # Read naïve result file
    with open(options.file_naive) as f:
        result_file = f.read().splitlines()
    results_naive = read_result_file(result_file)

    # Read naïve autotuned result file
    with open(options.file_naive_autotuned) as f:
        result_file = f.read().splitlines()
    results_naive_autotuned = read_result_file(result_file)

    # Read predicted optimal on autotuned set result file
    with open(options.file_predicted_autotuned) as f:
        result_file = f.read().splitlines()
    results_predicted_autotuned = read_result_file(result_file)

    # Performance comparison quantities
    improved = dict(zip(sorted(results.keys()),
                        [results[(m, n, k)] > results_naive[(m, n, k)]
                         for m, n, k in sorted(results.keys())]))
    perf_gain_res_over_naive = performance_gain(results_naive, results)
    rel_perf_gain_res_over_naive = relative_performance_gain(results_naive, results)
    perf_gain_autotuning_over_naive = performance_gain(results_naive_autotuned, autotuned_perfs)
    rel_gain_autotuning_over_naive = relative_performance_gain(results_naive_autotuned, autotuned_perfs)

    # ===============================================================================
    # Print results
    header = "m, n, k: predicted perf. [Gflops], naive perf. [Gflops], measured perf. [Gflops]"
    print(header)
    line = "{m:>2}, {n:>2}, {k:>2}, {pred_perf:>7.2f}, {naive_perf:>7.2f}, {measured_perf:>7.2f}, {perf_gain:>7.2f}, {better}"
    for m, n, k in sorted(results.keys()):
        print(line.format(m=m, n=n, k=k,
                          pred_perf=predicted_perfs[(m, n, k)],
                          naive_perf=results_naive[(m, n, k)],
                          measured_perf=results[(m, n, k)],
                          perf_gain=perf_gain_res_over_naive[(m, n, k)],
                          better=improved[(m, n, k)]))

    print("")
    print("Kernel performances improved by predictive model:", list(improved.values()).count(True), "/", len(results.keys()))
    perf_gain_improved = [pg for pg in perf_gain_res_over_naive.values() if pg > 0]
    print("Mean performance gain amongst improved kernels: {:.2f} Gflops".format(np.mean(perf_gain_improved)))
    print("")
    print("Kernel performances reduced by predictive model:", list(improved.values()).count(False), "/", len(results.keys()))
    perf_gain_deteriorated = [pg for pg in perf_gain_res_over_naive.values() if pg < 0]
    print("Mean performance loss amongst deteriorated kernels: {:.2f} Gflops".format(np.mean(perf_gain_deteriorated)))
    print("")
    print("Mean performance gain overall: {:.2f} Gflops".format(np.mean(list(perf_gain_res_over_naive.values()))))

    # ===============================================================================
    # Plot results (training set: autotuned VS naïve)
    plot_absolute_performance_gain(perf_gain_autotuning_over_naive, 'Autotuned', 'naive', 'autotuned')
    plot_relative_performance_gain(rel_gain_autotuning_over_naive, 'Autotuned', 'naive', 'autotuned')
    plot_performance_gains(results_naive_autotuned, autotuned_perfs, 'Autotuned', 'naive', 'autotuned')
    plot_performance_gains(results_predicted_autotuned, autotuned_perfs, 'Autotuned', 'predicted', 'autotuned')
    plot_performance_gains(results_naive_autotuned, results_predicted_autotuned, 'Autotuned', 'naive', 'predicted')

    mnks = [(m, n, k) for m, n, k in sorted(autotuned_perfs.keys(), key=lambda x: x[0]*x[1]*x[2])]
    mnk_products = [m*n*k for m, n, k in sorted(autotuned_perfs.keys(), key=lambda x: x[0]*x[1]*x[2])]
    res1 = [autotuned_perfs[mnk] for mnk in mnks]
    res2 = [results_naive_autotuned[mnk] for mnk in mnks]
    res3 = [results_predicted_autotuned[mnk] for mnk in mnks]

    marker_size = 1
    plt.plot(mnk_products, res1, '.', markersize=marker_size)
    plt.plot(mnk_products, res2, '.', color='#d62728', markersize=marker_size)
    plt.plot(mnk_products, res3, '.', color='black', markersize=marker_size)
    plt.xlabel('Autotuned' + ' (m, n, k) triplets (in order of increasing m*n*k)')
    plt.ylabel('Performance [Gflops]')
    plt.xscale('log')
    plt.legend(['autotuned', 'naïve', 'predicted'])
    plt.title('Performance of autotuned, naive and predicted parameter set')
    plt.show()

    m, n, k = (4, 4, 4)
    path_to_training_data = 'tune_big/'
    plot_choice_goodness(m, n, k, path_to_training_data)

    # ===============================================================================
    # Plot results (testing set: predictive modelling VS naïve)
    plot_absolute_performance_gain(perf_gain_res_over_naive, 'Tested', 'naive', 'predictive model')
    plot_relative_performance_gain(rel_perf_gain_res_over_naive, 'Tested', 'naive', 'predictive model')
    plot_performance_gains(results_naive, results, 'Tested', 'naive', 'predictive model')


# ===============================================================================
main(argv=sys.argv[1:])

#EOF
