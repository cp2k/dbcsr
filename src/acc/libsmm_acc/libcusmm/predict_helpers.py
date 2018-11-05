#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import matplotlib.pyplot as plt
import numpy as np


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


def performance_gain(baseline, current):
    """
    Compute the absolute perfomance gain, in Gflop/s between a baseline and a 'current'
    :param baseline, current: dictionary, keys: (m, n, k), values: performance in Gflop/s
    :return: dictionary, keys: (m, n, k), values: performance difference in Gflop/s
    """
    return dict(zip(sorted(current.keys()),
                    [current[(m, n, k)] - baseline[(m, n, k)]
                     for m, n, k in sorted(current.keys())]))


def relative_performance_gain(baseline, current):
    """
    Compute the relative perfomance gain (no units), between a baseline and a 'current'
    :param baseline, current: dictionary, keys: (m, n, k), values: performance in Gflop/s
    :return: dictionary, keys: (m, n, k), values: relative performance difference (no units)
    """
    return dict(zip(sorted(current.keys()),
                    [(current[(m, n, k)] - baseline[(m, n, k)]) / baseline[(m, n, k)]
                     for m, n, k in sorted(current.keys())]))


def plot_absolute_performance_gain(perf_gain, mnk_names, baseline_name, current_name, file_name=''):
    mnk_products = [m*n*k for m, n, k in sorted(perf_gain.keys(), key=lambda x: x[0]*x[1]*x[2])]
    plt.plot(mnk_products, list(perf_gain.values()), '.', markersize=1)
    plt.xlabel(mnk_names + ' (m, n, k) triplets (in order of increasing m*n*k)')
    plt.ylabel('Performance Gain [Gflops]')
    plt.title('Performance gain of ' + current_name + ' VS ' + baseline_name + ' parameter set')
    if file_name != '':
        plt.savefig(file_name)
    else:
        plt.show()


def plot_relative_performance_gain(rel_perf_gain, mnk_names, baseline_name, current_name, file_name=''):
    mnk_products = [m*n*k for m, n, k in sorted(rel_perf_gain.keys(), key=lambda x: x[0]*x[1]*x[2])]
    plt.plot(mnk_products, 100*np.array(list(rel_perf_gain.values())), '.', markersize=1)
    plt.xlabel(mnk_names + ' (m, n, k) triplets (in order of increasing m*n*k)')
    plt.ylabel('Performance Gain [%]')
    plt.title('Relative performance gain of ' + current_name + ' VS ' + baseline_name + ' parameter set')
    if file_name != '':
        plt.savefig(file_name)
    else:
        plt.show()


def plot_performance_gains(perf_gain1, perf_gain2, mnk_names, perf_gain1_name, perf_gain2_name, file_name=''):
    mnks = [(m, n, k) for m, n, k in sorted(perf_gain1.keys(), key=lambda x: x[0]*x[1]*x[2])]
    mnk_products = [m*n*k for m, n, k in sorted(perf_gain1.keys(), key=lambda x: x[0]*x[1]*x[2])]
    res1 = [perf_gain1[mnk] for mnk in mnks]
    res2 = [perf_gain2[mnk] for mnk in mnks]

    marker_size = 1
    plt.plot(mnk_products, res1, '.', markersize=marker_size)
    plt.plot(mnk_products, res2, '.', color='#d62728', markersize=marker_size)
    plt.xlabel(mnk_names + ' (m, n, k) triplets (in order of increasing m*n*k)')
    plt.ylabel('Performance [Gflops]')
    plt.xscale('log')
    plt.legend([perf_gain1_name, perf_gain2_name])
    plt.title('Performance of ' + perf_gain1_name + ' and ' + perf_gain2_name + ' parameter set')
    if file_name != '':
        plt.savefig(file_name)
    else:
        plt.show()


# Plot
# p1 = plt.bar(mnk_products, res_naive, color='#d62728')
# p2 = plt.bar(mnk_products, res, bottom=res_naive)
# plt.xlabel(x_label)
# plt.ylabel('Performance [Gflops]')
# plt.title('Performances')
# plt.legend((p1[0], p2[0]), ('Naive', 'Autotuned'))
# plt.show()
