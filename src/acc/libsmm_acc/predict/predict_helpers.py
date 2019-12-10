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
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kernels.smm_acc_predict import to_string


# ===============================================================================
# I/O helpers
def safe_pickle(data, file):
    """
    Pickle big files safely by processing them in chunks.
    This wrapper is a workaround for a bug on OSX (https://bugs.python.org/issue24658)

    :param data: data to be pickled
    :param file: file to pickle it into
    """
    max_bytes = 2 ** 31 - 1  # Maximum number of bytes to write in one chunk
    pickle_out = pickle.dumps(data)
    n_bytes = len(pickle_out)
    with open(file, "wb") as f:
        count = 0
        for i in range(0, n_bytes, max_bytes):
            f.write(pickle_out[i : min(n_bytes, i + max_bytes)])
            count += 1


def safe_pickle_load(file_path):
    """
    Load big pickled files safely by processing them in chunks
    This wrapper is a workaround a bug on OSX (https://bugs.python.org/issue24658)

    :param data: data to be loaded through pickle
    :param file: file to read from
    """
    max_bytes = 2 ** 31 - 1  # Maximum number of bytes to read in one chunk
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
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
    return dict(
        zip(
            sorted(current.keys()),
            [
                current[(m, n, k)] - baseline[(m, n, k)]
                for m, n, k in sorted(current.keys())
            ],
        )
    )


def relative_performance_gain(baseline, current):
    """
    Compute the relative perfomance gain (no units), between a baseline and a 'current'
    :param baseline, current: dictionary, keys: (m, n, k), values: performance in Gflop/s
    :return: dictionary, keys: (m, n, k), values: relative performance difference (no units)
    """
    return dict(
        zip(
            sorted(current.keys()),
            [
                (current[(m, n, k)] - baseline[(m, n, k)]) / baseline[(m, n, k)]
                for m, n, k in sorted(current.keys())
            ],
        )
    )


def plot_absolute_performance_gain(
    perf_gain, mnk_names, baseline_name, current_name, pp=None
):
    mnk_products = [
        m * n * k
        for m, n, k in sorted(perf_gain.keys(), key=lambda x: x[0] * x[1] * x[2])
    ]

    plt.figure()
    plt.plot(mnk_products, list(perf_gain.values()), ".", markersize=3)
    plt.plot([mnk_products[0], mnk_products[-1]], [0, 0], "-r")
    plt.xlabel(mnk_names + " (m, n, k) triplets (in order of increasing m*n*k)")
    plt.ylabel("Performance Gain [Gflops]")
    plt.title(
        "Performance gain of "
        + current_name
        + " VS "
        + baseline_name
        + " parameter set"
    )
    if pp is not None:
        pp.savefig()
    else:
        plt.show()


def plot_relative_performance_gain(
    rel_perf_gain, mnk_names, baseline_name, current_name, pp=None
):
    mnk_products = [
        m * n * k
        for m, n, k in sorted(rel_perf_gain.keys(), key=lambda x: x[0] * x[1] * x[2])
    ]

    plt.figure()
    plt.plot(
        mnk_products, 100 * np.array(list(rel_perf_gain.values())), ".", markersize=3
    )
    plt.plot([mnk_products[0], mnk_products[-1]], [0, 0], "-r")
    plt.xlabel(mnk_names + " (m, n, k) triplets (in order of increasing m*n*k)")
    plt.ylabel("Performance Gain [%]")
    plt.title(
        "Relative performance gain of "
        + current_name
        + " VS "
        + baseline_name
        + " parameter set"
    )
    if pp is not None:
        pp.savefig()
    else:
        plt.show()


def plot_performance_gains(
    perf_gain1, perf_gain2, mnk_names, perf_gain1_name, perf_gain2_name, pp=None
):
    mnks = [
        (m, n, k)
        for m, n, k in sorted(perf_gain2.keys(), key=lambda x: x[0] * x[1] * x[2])
    ]
    mnk_products = [
        m * n * k
        for m, n, k in sorted(perf_gain2.keys(), key=lambda x: x[0] * x[1] * x[2])
    ]
    res1 = [perf_gain1[mnk] for mnk in mnks]
    res2 = [perf_gain2[mnk] for mnk in mnks]

    marker_size = 3
    plt.figure()
    plt.plot(mnk_products, res1, ".", markersize=marker_size)
    plt.plot(mnk_products, res2, ".", color="#d62728", markersize=marker_size)
    plt.xlabel(mnk_names + " (m, n, k) triplets (in order of increasing m*n*k)")
    plt.ylabel("Performance [Gflops]")
    plt.xscale("log")
    plt.legend([perf_gain1_name, perf_gain2_name])
    plt.title(
        "Performance of "
        + perf_gain1_name
        + " and "
        + perf_gain2_name
        + " parameter set"
    )
    if pp is not None:
        pp.savefig()
    else:
        plt.show()


def plot_scaled_performance_gains(
    perf_gain1, perf_gain2, mnk_names, perf_gain1_name, perf_gain2_name, pp=None
):
    mnks = [
        (m, n, k)
        for m, n, k in sorted(perf_gain2.keys(), key=lambda x: x[0] * x[1] * x[2])
    ]
    mnk_products = [
        m * n * k
        for m, n, k in sorted(perf_gain2.keys(), key=lambda x: x[0] * x[1] * x[2])
    ]
    res1 = np.array([perf_gain1[mnk] for mnk in mnks])
    res2 = np.array([perf_gain2[mnk] for mnk in mnks])

    marker_size = 3
    plt.figure()
    plt.plot(mnk_products, 100 * res1, ".", markersize=marker_size)
    plt.plot(mnk_products, 100 * res2, ".", color="#d62728", markersize=marker_size)
    plt.xlabel(mnk_names + " (m, n, k) triplets (in order of increasing m*n*k)")
    plt.ylabel("Scaled performance [%]")
    plt.xscale("log")
    plt.legend([perf_gain1_name, perf_gain2_name])
    plt.title(
        "Performance of "
        + perf_gain1_name
        + " and "
        + perf_gain2_name
        + " parameter set"
    )
    if pp is not None:
        pp.savefig()
    else:
        plt.show()


def plot_choice_goodness(
    m,
    n,
    k,
    baseline_performances,
    max_performances,
    y_true,
    y_pred,
    train,
    pp,
    scaled=True,
):

    # Sort in ascending performances
    data_mnk = pd.DataFrame()
    if scaled:
        data_mnk["perf_true"] = (100 * y_true).tolist()
        data_mnk["perf_pred"] = (100 * y_pred).tolist()
    else:
        data_mnk["perf_true"] = y_true.flatten().tolist()
        data_mnk["perf_pred"] = y_pred.tolist()
    data_mnk.sort_values(by="perf_true", inplace=True)

    # Plot
    plt.figure()
    marker_size = 1
    par_set_ids = range(len(data_mnk.index.values))
    plt.plot(
        par_set_ids,
        data_mnk["perf_true"],
        "b.",
        markersize=marker_size,
        label="measured performances",
    )
    plt.xlabel("Parameter set id")
    plt.ylabel("Percentage of autotuned performance achieved [%]")
    type = "train" if train else "test"
    plt.title(
        "Performance profile of parameter sets for "
        + str((m, n, k))
        + "-triplet ("
        + type
        + ")"
    )

    # Annotate
    x = [0, len(y_true)]
    y = np.array([1, 1])
    perf_num = "{:2.2f}"

    # chosen
    idx_perf_chosen = data_mnk["perf_pred"].idxmax()
    perf_chosen = data_mnk["perf_true"][idx_perf_chosen]
    plt.plot(
        x,
        perf_chosen * y,
        "r-",
        label="perf of chosen param set: " + perf_num.format(perf_chosen) + "%",
    )

    # baseline
    if scaled:
        # baseline = per algo, scale it to 0-1
        perf_baseline = (
            100
            * baseline_performances[to_string(m, n, k)]
            / max_performances["{}x{}x{}".format(m, n, k)]
        )
    else:
        perf_baseline = baseline_performances[to_string(m, n, k)]
    plt.plot(
        x,
        perf_baseline * y,
        "g-",
        label="perf of baseline param set: " + perf_num.format(perf_baseline) + "%",
    )

    plt.legend(loc="lower right")
    pp.savefig()
