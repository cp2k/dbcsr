# -*- coding: utf-8 -*-
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

predict_genpars_compute.py
import re
import sys
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kernels.cusmm_dnt_helper import kernel_algorithm


# ===============================================================================
# I/O helpers
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


# ===============================================================================
# Model evaluation helpers
def performance_gain(baseline, current):
    """
    Compute the absolute perfomance gain, in Gflop/s between a baseline and a 'current'
    :param baseline, current: dictionary, keys: (m, n, k), values: performance in Gflop/s
    :return: dictionary, keys: (m, n, k), values: performance difference in Gflop/s
    """
    return dict(zip(sorted(current.keys()),
                    [current[(m, n, k)] - baseline[(m, n, k)]
                     for m, n, k in sorted(current.keys())]))


def plot_training_data(Y, X_mnk, algo, folder=''):
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
    if folder != '':
        file_name = os.path.join(folder, "y_scaled.svg")
        plt.savefig(file_name)
        print(file_name)
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


def plot_absolute_performance_gain(perf_gain, mnk_names, baseline_name, current_name, pp=None):
    mnk_products = [m*n*k for m, n, k in sorted(perf_gain.keys(), key=lambda x: x[0]*x[1]*x[2])]

    plt.figure()
    plt.plot(mnk_products, list(perf_gain.values()), '.', markersize=3)
    plt.plot([mnk_products[0], mnk_products[-1]], [0, 0], '-r')
    plt.xlabel(mnk_names + ' (m, n, k) triplets (in order of increasing m*n*k)')
    plt.ylabel('Performance Gain [Gflops]')
    plt.title('Performance gain of ' + current_name + ' VS ' + baseline_name + ' parameter set')
    if pp is not None:
        pp.savefig()
    else:
        plt.show()


def plot_relative_performance_gain(rel_perf_gain, mnk_names, baseline_name, current_name, pp=None):
    mnk_products = [m*n*k for m, n, k in sorted(rel_perf_gain.keys(), key=lambda x: x[0]*x[1]*x[2])]

    plt.figure()
    plt.plot(mnk_products, 100*np.array(list(rel_perf_gain.values())), '.', markersize=3)
    plt.plot([mnk_products[0], mnk_products[-1]], [0, 0], '-r')
    plt.xlabel(mnk_names + ' (m, n, k) triplets (in order of increasing m*n*k)')
    plt.ylabel('Performance Gain [%]')
    plt.title('Relative performance gain of ' + current_name + ' VS ' + baseline_name + ' parameter set')
    if pp is not None:
        pp.savefig()
    else:
        plt.show()


def plot_performance_gains(perf_gain1, perf_gain2, mnk_names, perf_gain1_name, perf_gain2_name, pp=None):
    mnks = [(m, n, k) for m, n, k in sorted(perf_gain2.keys(), key=lambda x: x[0]*x[1]*x[2])]
    mnk_products = [m*n*k for m, n, k in sorted(perf_gain2.keys(), key=lambda x: x[0]*x[1]*x[2])]
    res1 = [perf_gain1[mnk] for mnk in mnks]
    res2 = [perf_gain2[mnk] for mnk in mnks]

    marker_size = 3
    plt.figure()
    plt.plot(mnk_products, res1, '.', markersize=marker_size)
    plt.plot(mnk_products, res2, '.', color='#d62728', markersize=marker_size)
    plt.xlabel(mnk_names + ' (m, n, k) triplets (in order of increasing m*n*k)')
    plt.ylabel('Performance [Gflops]')
    plt.xscale('log')
    plt.legend([perf_gain1_name, perf_gain2_name])
    plt.title('Performance of ' + perf_gain1_name + ' and ' + perf_gain2_name + ' parameter set')
    if pp is not None:
        pp.savefig()
    else:
        plt.show()


def plot_scaled_performance_gains(perf_gain1, perf_gain2, mnk_names, perf_gain1_name, perf_gain2_name, pp=None):
    mnks = [(m, n, k) for m, n, k in sorted(perf_gain2.keys(), key=lambda x: x[0]*x[1]*x[2])]
    mnk_products = [m*n*k for m, n, k in sorted(perf_gain2.keys(), key=lambda x: x[0]*x[1]*x[2])]
    res1 = np.array([perf_gain1[mnk] for mnk in mnks])
    res2 = np.array([perf_gain2[mnk] for mnk in mnks])

    marker_size = 3
    plt.figure()
    plt.plot(mnk_products, 100*res1, '.', markersize=marker_size)
    plt.plot(mnk_products, 100*res2, '.', color='#d62728', markersize=marker_size)
    plt.xlabel(mnk_names + ' (m, n, k) triplets (in order of increasing m*n*k)')
    plt.ylabel('Scaled performance [%]')
    plt.xscale('log')
    plt.legend([perf_gain1_name, perf_gain2_name])
    plt.title('Performance of ' + perf_gain1_name + ' and ' + perf_gain2_name + ' parameter set')
    if pp is not None:
        pp.savefig()
    else:
        plt.show()


def plot_choice_goodness(m, n, k, baseline_performances, max_performances, y_true, y_pred, train, pp, scaled=True):

    # Sort in ascending performances
    data_mnk = pd.DataFrame()
    if scaled:
        data_mnk['perf_true'] = (100 * y_true.flatten()).tolist()
        data_mnk['perf_pred'] = (100 * y_pred).tolist()
    else:
        data_mnk['perf_true'] = y_true.flatten().tolist()
        data_mnk['perf_pred'] = y_pred.tolist()
    data_mnk.sort_values(by='perf_true', inplace=True)

    # Plot
    plt.figure()
    marker_size = 1
    par_set_ids = range(len(data_mnk.index.values))
    plt.plot(par_set_ids, data_mnk['perf_true'], 'b.', markersize=marker_size, label='measured performances')
    plt.xlabel('Parameter set id')
    plt.ylabel('Performance scaled [%]')
    plt.title('Performance profile of parameter sets for ' + str((m, n, k)) + '-triplet')

    # Annotate
    x = [0, len(y_true)]
    y = np.array([1, 1])
    perf_num = "{:2.2f}"

    # autotuning
    perf_autotuned_algo = data_mnk['perf_true'].max()
    plt.plot(x, perf_autotuned_algo * y, 'k-', label='max (for this algo): ' + perf_num.format(perf_autotuned_algo))

    # chosen
    idx_perf_chosen = data_mnk['perf_pred'].idxmax()
    perf_chosen = data_mnk['perf_true'][idx_perf_chosen]
    plt.plot(x, perf_chosen * y, 'r-', label='chosen: ' + perf_num.format(perf_chosen))

    # baseline
    mnk_string = "{}x{}x{}"
    if scaled:
        # baseline = per algo, scale it to 0-1
        perf_baseline = 100 * baseline_performances[mnk_string.format(m, n, k)] / max_performances["{}x{}x{}".format(m, n, k)]
    else:
        perf_baseline = baseline_performances[mnk_string.format(m, n, k)]
    plt.plot(x, perf_baseline * y, 'g-', label='baseline: ' + perf_num.format(perf_baseline))

    type = 'train' if train else 'test'
    plt.legend(loc="lower right")
    pp.savefig()


def plot_choice_goodness_old(m, n, k, path_to_training_data, perf_type, file_name=''):

    # Read data corresponding to the given mnk
    chunksize = 100000
    data_mnk = pd.DataFrame()
    mnk = "{}x{}x{}".format(m, n, k)
    compatible_algos = [algo for algo in list(kernel_algorithm.keys()) if compatible_mnk(algo, m, n, k)]
    for algo in compatible_algos:

        # I should actually be loading the raw parameters
        data_algo_file_raw = os.path.join(path_to_training_data, 'raw_training_data_' + algo + '.csv')
        reader_raw = pd.read_csv(data_algo_file_raw, index_col=0, chunksize=chunksize, iterator=True),

        data_algo_file_derived = os.path.join(path_to_training_data, 'training_data_' + algo + '.csv')
        reader_derived = pd.read_csv(data_algo_file_derived, index_col=0, chunksize=chunksize, iterator=True),

        for chunk_raw, chunk_derived in zip(reader_raw, reader_derived):

            chunk_raw = chunk_raw.read()
            chunk_derived = chunk_derived.read()

            # ===============================================================================
            # Read 'X'
            to_drop = list()
            if algo in ['tiny', 'small', 'medium']:
                to_drop = ['w', 'v']
                if algo in ['tiny']:
                    to_drop += ['tile_m', 'tile_n']
            chunk_pars = pd.concat([chunk_raw.drop(to_drop + ['perf (Gflop/s)'], axis=1),
                                    chunk_derived.drop(['perf_squared', 'perf_scaled', 'perf_scaled_by_algo'], axis=1)],
                                   axis=1)

            # ===============================================================================
            # Read 'Y'
            chunk_perf = pd.DataFrame()
            if perf_type == 'perf':
                chunk_perf[perf_type] = chunk_raw[perf_type + ' (Gflop/s)']
            else:
                chunk_perf[perf_type] = chunk_derived[perf_type]

            # ===============================================================================
            # Construct 'X_mnk'
            chunk_mnk = pd.DataFrame()
            chunk_mnk['mnk'] = chunk_pars['m'].astype(str) + 'x' + chunk_pars['n'].astype(str) + 'x' + chunk_pars['k'].astype(str)

            # ===============================================================================
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
