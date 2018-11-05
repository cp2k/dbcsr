#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
from optparse import OptionParser
from kernels.cusmm_dnt_helper import params_dict_to_kernel
from predict_helpers import *


########################################################################################################################
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
                      default="libcusmm_timer_multiply.10310667.o",
                      help="Real result file to evaluate. Default: %default")
    parser.add_option("-n", "--file_naive", metavar="filename.out",
                      default="libcusmm_timer_multiply.10312765.o",
                      help="Naive result file to evaluate (testing mnks). Default: %default")  # naive test
    parser.add_option("-a", "--file_naive_autotuned", metavar="filename.out",
                      default="libcusmm_timer_autotuned_multiply.10314368.o",   # naive autotuned
                      help="Naive result file to evaluate (autotuned mnks). Default: %default")
    options, args = parser.parse_args(sys.argv)

    ####################################################################################################################
    # Read parameter file
    with open(options.params) as f:
        all_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]
    predicted_mnks = [(k.m, k.n, k.k) for k in all_kernels if not k.autotuned]
    predicted_perfs_ = [k.perf for k in all_kernels if not k.autotuned]
    predicted_perfs = dict(zip(predicted_mnks, predicted_perfs_))
    autotuned_mnks = [(k.m, k.n, k.k) for k in all_kernels if k.autotuned]
    autotuned_perfs_ = [k.perf for k in all_kernels if k.autotuned]
    autotuned_perfs = dict(zip(autotuned_mnks, autotuned_perfs_))

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

    # Performance comparison quantities
    improved = dict(zip(sorted(results.keys()),
                        [results[(m, n, k)] > results_naive[(m, n, k)]
                         for m, n, k in sorted(results.keys())]))
    perf_gain_res_over_naive = performance_gain(results_naive, results)
    rel_perf_gain_res_over_naive = relative_performance_gain(results_naive, results)
    perf_gain_autotuning_over_naive = performance_gain(results_naive_autotuned, autotuned_perfs)
    rel_gain_autotuning_over_naive = relative_performance_gain(results_naive_autotuned, autotuned_perfs)


    ####################################################################################################################
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

    ####################################################################################################################
    # Plot results (training set: autotuned VS naïve)
    plot_absolute_performance_gain(perf_gain_autotuning_over_naive, 'Autotuned', 'naive', 'autotuned')
    plot_relative_performance_gain(rel_gain_autotuning_over_naive, 'Autotuned', 'naive', 'autotuned')
    plot_performance_gains(results_naive_autotuned, autotuned_perfs, 'Autotuned', 'naive', 'autotuned')

    ####################################################################################################################
    # Plot results (testing set: predictive modelling VS naïve)
    plot_absolute_performance_gain(perf_gain_res_over_naive, 'Tested', 'naive', 'predictive model')
    plot_relative_performance_gain(rel_perf_gain_res_over_naive, 'Tested', 'naive', 'predictive model')
    plot_performance_gains(results_naive, results, 'Tested', 'naive', 'predictive model')


# ===============================================================================
main(argv=sys.argv[1:])

#EOF
