#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kernels.cusmm_dnt_helper import kernel_algorithm, compatible_mnk



########################################################################################################################
# I/O helpers
########################################################################################################################
def safe_pickle(data, file):
    """
    Pickle big files safely by processing them in chunks
    :param data: data to be pickled
    :param file: file to pickle it into
    """
    max_bytes = 2**31 - 1  # Maximum number of bytes to pickle in one chunk
    pickle_out = pickle.dumps(data)
    n_bytes = sys.getsizeof(pickle_out)
    with open(file, 'wb') as f:
        count = 0
        for i in range(0, n_bytes, max_bytes):
            f.write(pickle_out[i:min(n_bytes, i + max_bytes)])
            count += 1


def safe_pickle_load(file_path):
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f.read(max_bytes)
    return pickle.loads(bytes_in)


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
# Model evaluation helpers
########################################################################################################################
def performance_gain(baseline, current):
    """
    Compute the absolute perfomance gain, in Gflop/s between a baseline and a 'current'
    :param baseline, current: dictionary, keys: (m, n, k), values: performance in Gflop/s
    :return: dictionary, keys: (m, n, k), values: performance difference in Gflop/s
    """
    return dict(zip(sorted(current.keys()),
                    [current[(m, n, k)] - baseline[(m, n, k)]
                     for m, n, k in sorted(current.keys())]))


def plot_training_data(Y, X_mnk, folder, algo, file_name):
    import re
    import matplotlib.pyplot as plt

    print("Plotting training data...")

    mnks_strings = X_mnk['mnk'].values
    mnks = list()
    mnk_str = re.compile(r"(\d+)x(\d+)x(\d+)")
    for mnk_s in mnks_strings:
        match = mnk_str.match(mnk_s)
        mnks.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))

    perf_scaled = zip(mnks, Y["perf_scaled"])
    mnk_products_perf_sorted = [(mnk[0]*mnk[1]*mnk[2], p) for mnk, p in sorted(perf_scaled, key=lambda x: x[0][0]*x[0][1]*x[0][2])]
    tmp = list(zip(*mnk_products_perf_sorted))
    mnk_products_sorted = tmp[0]
    perf_scaled_sorted = tmp[1]

    # Plot
    plt.plot(mnk_products_sorted, 100*np.array(perf_scaled_sorted), '.', markersize=1)
    plt.xlabel('Training (m, n, k) triplets (in order of increasing m*n*k)')
    plt.ylabel('Scaled performance [%]')
    plt.title('Scaled performance on training data (' + algo + ')')
    if file_name != '':
        plt.savefig(file_name)
    else:
        plt.show()


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


def plot_choice_goodness(m, n, k, path_to_training_data, file_name=''):

    # Read data corresponding to the given mnk
    chunksize = 100000
    data_mnk = pd.DataFrame()
    mnk = "{}x{}x{}".format(m, n, k)
    compatible_algos = [algo for algo in list(kernel_algorithm.keys()) if compatible_mnk(algo, m, n, k)]
    for algo in compatible_algos:

        # I should actually be loading the raw parameters

        data_algo_file_pars = os.path.join(path_to_training_data, 'train_all_' + algo + '_X.csv')
        reader_pars = pd.read_csv(data_algo_file_pars, index_col=0, chunksize=chunksize, iterator=True),
        data_algo_file_mnk = os.path.join(path_to_training_data, 'train_all_' + algo + '_X_mnk.csv')
        reader_mnk = pd.read_csv(data_algo_file_mnk, index_col=0, chunksize=chunksize, iterator=True),
        data_algo_file_pers = os.path.join(path_to_training_data, 'train_all_' + algo + '_Y_perf.csv')
        reader_perf = pd.read_csv(data_algo_file_pers, index_col=0, chunksize=chunksize, iterator=True),

        for chunk_pars, chunk_mnk, chunk_perf in zip(reader_pars, reader_mnk, reader_perf):

            chunk_pars = chunk_pars.read()
            chunk_mnk = chunk_mnk.read()
            chunk_perf = chunk_perf.read()
            idx_mnk = np.where(chunk_mnk == mnk)[0].tolist()
            if len(idx_mnk) > 0:
                df_tmp = chunk_pars.iloc[idx_mnk]
                df_tmp.loc[:, 'mnk'] = chunk_mnk.iloc[idx_mnk]
                df_tmp.loc[:, 'perf'] = chunk_perf.iloc[idx_mnk]
                data_mnk = data_mnk.append(df_tmp, ignore_index=True)

    # Sort in ascending performances
    data_mnk.sort_values(by='perf', inplace=True)

    # Plot
    marker_size = 1
    par_set_ids = range(len(data_mnk.index.values))
    plt.plot(par_set_ids, data_mnk['perf'], '-', markersize=marker_size)
    plt.xlabel('Parameter set id')
    plt.ylabel('Performance [Gflops]')
    plt.title('Performance profile of parameter sets for ' + str((m, n, k)) + '-triplet')

    # Annotate
    # plt.text(, 'max:')
    # plt.text(, 'chosen:')
    # plt.text(, 'naive:')
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
