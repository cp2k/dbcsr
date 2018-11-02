#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import re
from optparse import OptionParser
import matplotlib.pyplot as plt
import numpy as np
from kernels.cusmm_dnt_helper import arch_number, params_dict_to_kernel


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
                      default="libcusmm_timer_multiply.10310667.o", help="Real result file to evaluate. Default: %default")
    parser.add_option("-n", "--file_naive", metavar="filename.out",
                      default="libcusmm_timer_multiply.10312765.o", help="Naive result file to evaluate. Default: %default")  # naive test
                      #default="libcusmm_timer_autotuned_multiply.10314368.o", help="Naive result file to evaluate. Default: %default") # naive autotuned
    options, args = parser.parse_args(sys.argv)

    ####################################################################################################################
    # Read and sample
    assert options.params in arch_number.keys(), "Cannot find compute version for file " + str(options.params)
    with open(options.params) as f:
        all_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]
    predicted_mnks = [(k.m, k.n, k.k) for k in all_kernels if not k.autotuned]
    predicted_perfs_ = [k.perf for k in all_kernels if not k.autotuned]
    predicted_perfs = dict(zip(predicted_mnks, predicted_perfs_))
    autotuned_mnks = [(k.m, k.n, k.k) for k in all_kernels if k.autotuned]
    autotuned_perfs_ = [k.perf for k in all_kernels if k.autotuned]
    autouned_perfs = dict(zip(predicted_mnks, predicted_perfs_))

    with open(options.file) as f:
        result_file = f.read().splitlines()
    results = read_result_file(result_file)

    with open(options.file_naive) as f:
        result_file = f.read().splitlines()
    results_naive = read_result_file(result_file)
    perf_gain = dict(zip(sorted(results.keys()),
                         [results[(m, n, k)] - results_naive[(m, n, k)]
                          for m, n, k in sorted(results.keys())]))
    improved = dict(zip(sorted(results.keys()),
                         [results[(m, n, k)] > results_naive[(m, n, k)]
                          for m, n, k in sorted(results.keys())]))
    rel_gain = dict(zip(sorted(results.keys()),
                        [(results[(m, n, k)] - results_naive[(m, n, k)]) / results_naive[(m, n, k)]
                         for m, n, k in sorted(results.keys())]))

    ####################################################################################################################
    # Print
    header = "m, n, k: predicted perf. [Gflops], naive perf. [Gflops], measured perf. [Gflops]"
    print(header)
    line = "{m:>2}, {n:>2}, {k:>2}, {pred_perf:>7.2f}, {naive_perf:>7.2f}, {measured_perf:>7.2f}, {perf_gain:>7.2f}, {better}"
    for m, n, k in sorted(results.keys()):
        print(line.format(m=m, n=n, k=k,
                          pred_perf=predicted_perfs[(m, n, k)],
                          naive_perf=results_naive[(m, n, k)],
                          measured_perf=results[(m, n, k)],
                          perf_gain=perf_gain[(m, n, k)],
                          better=improved[(m, n, k)]))

    print("")
    print("Kernel performances improved by predictive model:", list(improved.values()).count(True), "/", len(results.keys()))
    perf_gain_improved = [pg for pg in perf_gain.values() if pg > 0]
    print("Mean performance gain amongst improved kernels: {:.2f} Gflops".format(np.mean(perf_gain_improved)))
    print("")
    print("Kernel performances reduced by predictive model:", list(improved.values()).count(False), "/", len(results.keys()))
    perf_gain_deteriorated = [pg for pg in perf_gain.values() if pg < 0]
    print("Mean performance loss amongst deteriorated kernels: {:.2f} Gflops".format(np.mean(perf_gain_deteriorated)))
    print("")
    print("Mean performance gain overall: {:.2f} Gflops".format(np.mean(list(perf_gain.values()))))

    mnk_str = "{}x{}x{}"
    mnks = [mnk_str.format(m, n, k) for m, n, k in sorted(results.keys())]
    mnk_string = [mnk_str.format(m, n, k) for m, n, k in sorted(results.keys(), key=lambda x: x[0]*x[1]*x[2])]
    mnks = [(m, n, k) for m, n, k in sorted(results.keys(), key=lambda x: x[0]*x[1]*x[2])]
    ####################################################################################################################
    # Plot
    # plt.bar(mnks, list(perf_gain.values()), align='center', alpha=0.5)
    # plt.xlabel('Tested (m, n, k) triplets')
    # plt.ylabel('Performance Gain [Gflops]')
    # plt.title('Performance gain of predictive model VS naive parameter set')
    # plt.show()
    ####################################################################################################################
    # Plot
    # plt.bar(mnks, list(perf_gain.values()), align='center', alpha=0.5)
    # plt.xlabel('Tested (m, n, k) triplets')
    # plt.ylabel('Performance Gain [Gflops]')
    # plt.title('Performance gain of predictive model VS naive parameter set')
    # plt.show()

    res = [results[mnk] for mnk in mnks]
    res_naive = [results_naive[mnk] for mnk in mnks]
    mnks = [m*n*k for m, n, k in sorted(results.keys(), key=lambda x: x[0]*x[1]*x[2])]
    #mnks = [((m, n, k), m*n*k) for m, n, k in sorted(results.keys(), key=lambda x: x[0]*x[1]*x[2])]
    # for mnk, r in zip(mnks, res):
    #     m, n, k, mnk_ = mnk
    #     print(m, n, k, ":", mnk_, ', ', r)

    num = 150
    plt.plot(mnks, res, '.', color='#d62728')
    plt.plot(mnks, res_naive, '.')
    plt.xlabel('Tested (m, n, k) triplets')
    plt.ylabel('Performance Gain [%]')
    plt.xscale('log')
    #plt.title('Relative performance gain of predictive model VS naive parameter set')
    plt.show()

    ####################################################################################################################
    # Plot
    # bottom = list(results_naive.values())
    # top = list(results.values())
    # p1 = plt.bar(mnks, bottom, color='#d62728')
    # p2 = plt.bar(mnks, top)
    #
    # plt.ylabel('Performance [Gflops]')
    # plt.title('Performances')
    # plt.legend((p1[0], p2[0]), ('Naive', 'Autotuned'))
    # plt.show()


#===============================================================================
main(argv=sys.argv[1:])

#EOF
