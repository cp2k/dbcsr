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
import json
import argparse
import numpy as np
import pandas as pd
import dask.dataframe as dd
from joblib import Parallel, delayed
from tqdm import tqdm
from kernels.cusmm_predict import (
    PredictiveParameters,
    derived_parameters,
    kernel_algorithm,
    mnk_pattern,
)


# ===============================================================================
# HELPER FUNCTIONS
# ===============================================================================
def update_maximums(dictionnary_to_update, dictionnary_partial):
    for mnk, new_perf in dictionnary_partial.items():
        if mnk in dictionnary_to_update.keys():
            if new_perf > dictionnary_to_update[mnk]:
                dictionnary_to_update[mnk] = new_perf
        else:
            dictionnary_to_update[mnk] = new_perf
    return dictionnary_to_update


# ===============================================================================
def get_idx_baseline(data_mnk, algorithm, baseline_pars):
    """

    """
    if algorithm in ["tiny"]:
        idx_baseline = data_mnk[
            (data_mnk.m == baseline_pars["m"])
            & (data_mnk.n == baseline_pars["n"])
            & (data_mnk.k == baseline_pars["k"])
            & (data_mnk.threads == baseline_pars["threads"])
            & (data_mnk.grouping == baseline_pars["grouping"])
            & (data_mnk.minblocks == baseline_pars["minblocks"])
        ].index.tolist()
    elif algorithm in ["small", "medium"]:
        idx_baseline = data_mnk[
            (data_mnk.m == baseline_pars["m"])
            & (data_mnk.n == baseline_pars["n"])
            & (data_mnk.k == baseline_pars["k"])
            & (data_mnk.threads == baseline_pars["threads"])
            & (data_mnk.grouping == baseline_pars["grouping"])
            & (data_mnk.minblocks == baseline_pars["minblocks"])
            & (data_mnk.tile_m == baseline_pars["tile_m"])
            & (data_mnk.tile_n == baseline_pars["tile_n"])
        ].index.tolist()
    else:  # algorithm is largeDB1 or largeDB2
        idx_baseline = data_mnk[
            (data_mnk.m == baseline_pars["m"])
            & (data_mnk.n == baseline_pars["n"])
            & (data_mnk.k == baseline_pars["k"])
            & (data_mnk.threads == baseline_pars["threads"])
            & (data_mnk.minblocks == baseline_pars["minblocks"])
            & (data_mnk.tile_m == baseline_pars["tile_m"])
            & (data_mnk.tile_n == baseline_pars["tile_n"])
            & (data_mnk.w == baseline_pars["w"])
            & (data_mnk.v == baseline_pars["v"])
        ].index.tolist()
    return idx_baseline


def get_performance_closest_to_baseline(
    data, algorithm, mnk, gpu_properties, autotuning_properties
):
    """
    Sometimes, the so-called "baseline" parameter set does not appear in the training data.
    This function finds the performance of the parameter set from the training data whose parameters are closest to those of the
    baseline parameter sets.
    """
    m, n, k = mnk_pattern.match(mnk).groups()
    m, n, k = int(m), int(n), int(k)

    data_mnk = data[(data["m"] == m) & (data["n"] == n) & (data["k"] == k)].compute()
    baseline_pars = kernel_algorithm[algorithm].baseline(
        m, n, k, gpu_properties, autotuning_properties
    )

    # Get performance of baseline parameters for this algorithm & this mnk:
    idx_baseline = get_idx_baseline(data_mnk, algorithm, baseline_pars)

    # Get performance of baseline parameters for this algorithm & this mnk:
    if len(idx_baseline) == 0:
        # Generate space of possibilities
        pars_sets = kernel_algorithm[algorithm].promising_parameters(
            m, n, k, gpu_properties, autotuning_properties
        )
        # Sort space by distance to baseline set
        pars_sets.sort(
            key=lambda x: kernel_algorithm[algorithm].parameter_set_distance(
                x, baseline_pars
            )
        )

        for pars_set in pars_sets:
            idx_baseline = get_idx_baseline(data_mnk, algorithm, pars_set)
            if len(idx_baseline) > 0:
                break
        else:
            assert False, (
                'Could not find closest baseline for mnk=({}x{}x{}) and for algorithm "{}".'
                + "\nLast baseline parameters searched:\n{}"
                + "\nParameter sets searched:\n".format(
                    m, n, k, algorithm, baseline_pars
                )
            )

    idx_baseline = idx_baseline[0]
    baseline_perf = data_mnk["perf (Gflop/s)"][idx_baseline]
    return round(baseline_perf, 3)


def process_chunk(data_chunk, algorithm, gpu_properties, autotuning_properties):
    """
    Given a chunk of data, compute the baseline and maximum performance of the (m, n, k)-triplets featured in the chunk of data.
    """
    # Add "mnk" column
    data_chunk["mnk"] = (
        data_chunk["m"].astype(str)
        + "x"
        + data_chunk["n"].astype(str)
        + "x"
        + data_chunk["k"].astype(str)
    )
    # Get mnks
    mnks = data_chunk["mnk"].unique()

    # For each (mnk), ...
    baseline_performances = dict()
    max_performances = dict()
    for i, mnk in enumerate(mnks):

        data_mnk = data_chunk[data_chunk["mnk"] == mnk]
        m, n, k = mnk_pattern.match(mnk).groups()
        m, n, k = int(m), int(n), int(k)

        # Get baseline configuration for this algorithm & this mnk:
        baseline_pars = kernel_algorithm[algorithm].baseline(
            m, n, k, gpu_properties, autotuning_properties
        )

        # Get performance of baseline parameters for this algorithm & this mnk:
        idx_baseline = get_idx_baseline(data_mnk, algorithm, baseline_pars)
        if len(idx_baseline) < 1:
            baseline_perf = 0
        else:
            idx_baseline = idx_baseline[0]
            baseline_perf = data_mnk["perf (Gflop/s)"][idx_baseline]

        baseline_performances[mnk] = round(baseline_perf, 3)

        # Get max performance for this algorithm & this mnk
        max_perf = data_mnk["perf (Gflop/s)"].max()
        max_performances[mnk] = round(max_perf, 3)

    return baseline_performances, max_performances


# ===============================================================================
def write_to_parquet(data_path, algorithm):
    """
    Compress CSV files to parquet
    """
    # Check whether the files corresponding to this algorithm have been compressed to parquet already
    parquet_file = os.path.join(data_path, "training_data_" + algorithm + ".parquet")
    parquet_file_done = os.path.join(
        data_path, "training_data_" + algorithm + ".parquet.done"
    )
    print(
        "\n\n------------------------------------------------------------------------"
    )
    if os.path.exists(parquet_file_done):
        print("Found {:40}, skipping".format(parquet_file_done))

    else:

        print("Didn't find {:40}, generating".format(parquet_file_done))

        # [RAW] Read CSV files into Pandas dataframes
        data_file_raw = os.path.join(
            data_path, "raw_training_data_" + algorithm + ".csv"
        )
        print("\nRead raw data from: {}".format(data_file_raw))
        data_raw = dd.read_csv(data_file_raw)
        raw_data_nrows = len(data_raw)
        #  n_partitions should be > 1 !
        n_partitions = max(1, int(raw_data_nrows // 1e5))
        data_raw = data_raw.repartition(npartitions=n_partitions)
        data_raw = data_raw.reset_index().set_index("index")
        data_raw["idx"] = 1
        data_raw["idx"] = data_raw.idx.cumsum()
        data_raw = data_raw.set_index("idx", sorted=True)
        print("Raw data head:\n", data_raw.head())

        # [DERIVED] Read CSV files into Pandas dataframes
        data_file_derived = os.path.join(
            data_path, "training_data_" + algorithm + ".csv"
        )
        print("\nRead derived data from: {}".format(data_file_derived))
        data_derived = dd.read_csv(data_file_derived)
        derived_data_nrows = len(data_derived)
        data_derived = data_derived.repartition(npartitions=n_partitions)
        data_derived = data_derived.reset_index().set_index("index")
        data_derived["idx"] = 1
        data_derived["idx"] = data_derived.idx.cumsum()
        data_derived = data_derived.set_index("idx", sorted=True)
        print("Derived data head:\n", data_derived.head())

        # Merge raw/derived data together
        print("Merging raw and derived ...")
        data = dd.merge(data_raw, data_derived, left_index=True, right_index=True)

        len_data, len_data_raw, len_data_derived = (
            len(data),
            raw_data_nrows,
            derived_data_nrows,
        )
        nrows_message_temp = """
        Data 1     : {:15,},
        Data 2     : {:15,},
        Merged data: {:15,}"""
        nrows_message = nrows_message_temp.format(
            len_data_raw, len_data_derived, len_data
        )
        assert len_data == len_data_raw, "Mismatch in number of rows\n" + nrows_message
        assert len_data == len_data_derived, (
            "Mismatch in number of rows\n" + nrows_message
        )

        # Add "mnk" column
        data["mnk"] = (
            data["m"].astype(str)
            + "x"
            + data["n"].astype(str)
            + "x"
            + data["k"].astype(str)
        )

        # Print info on merged dataset
        print("\nMerged data head:", data.head())
        data_nrows = len(data)
        nrows_message = """
Data        : {:15,},
Raw data    : {:15,},
Derived data: {:15,}""".format(
            data_nrows, raw_data_nrows, derived_data_nrows
        )
        assert data_nrows == raw_data_nrows, (
            "Mismatch in number of rows\n" + nrows_message
        )
        assert data_nrows == derived_data_nrows, (
            "Mismatch in number of rows\n" + nrows_message
        )
        print(nrows_message)

        # Compress files to Parquet
        print("Compress and write to {}".format(parquet_file))
        data.to_parquet(parquet_file, engine="fastparquet", compression="snappy")
        open(
            parquet_file_done, "w"
        ).close()  # touch a file to mark that parquet is done


# ===============================================================================
def get_non_null(l):
    """
    Given a list "l", return its first non-null element, if existing, otherwise return null.
    """
    for e in l:
        if e > 0:
            return e
    return 0


def get_max(l):
    """
    Return the largest element of a list of numbers
    """
    return np.array(l).max()


def list_of_dics_to_dic_of_lists(list_of_dics):
    """
    Given a list "list_of_dics" of dictionaries "d", with keys "k" and values "v",
    construct a dictionary with keys "k" and values which are lists "[v1, v2, ...]"
    of the values corresponding to "k" in the various dictionaries "d"
    """
    dic_of_lists = dict()
    for dic in list_of_dics:
        for k, v in dic.items():
            if k not in dic_of_lists.keys():
                dic_of_lists[k] = list()
            dic_of_lists[k].append(v)
    return dic_of_lists


def dic_of_dics_to_dic_of_lists(dic_of_dics):
    dic_of_lists = dict()
    for _, dic in dic_of_dics.items():
        for k, v in dic.items():
            if k not in dic_of_lists.keys():
                dic_of_lists[k] = list()
            dic_of_lists[k].append(v)
    return dic_of_lists


def write_baseline_and_max_records_per_algorithm(
    data_path, algorithm, arch, n_jobs, chunk_size
):
    """
    Write records of baseline performances and maximum performances for the training mnks.
    This function reads from the raw data file (`raw_training_data_ALGORITHM.csv`)
    Writes to JSON files.
    """
    # Read GPU properties and autotuning properties
    with open("../kernels/gpu_properties.json") as f:
        gpu_properties = json.load(f)[str(arch)]
    with open("../kernels/autotuning_properties.json") as f:
        autotuning_properties = json.load(f)

    # Check whether record of baseline exists
    baseline_performances_per_algo_file = os.path.join(
        data_path, "baseline_performances_" + algorithm + ".json"
    )
    max_performances_per_algo_file = os.path.join(
        data_path, "max_performances_" + algorithm + ".json"
    )
    print(
        "\n\n------------------------------------------------------------------------"
    )
    if os.path.exists(baseline_performances_per_algo_file) and os.path.exists(
        max_performances_per_algo_file
    ):
        print("Found {:40}, skipping".format(baseline_performances_per_algo_file))
        print("Found {:40}, skipping".format(max_performances_per_algo_file))

    else:

        print("Processing data of algorithm {}".format(algorithm))
        raw_pars_cols = kernel_algorithm[algorithm].launch_parameters
        if algorithm in ["largeDB1", "largeDB2"]:
            raw_pars_cols.remove("grouping")

        data_file_raw = os.path.join(
            data_path, "raw_training_data_" + algorithm + ".csv"
        )
        baseline_and_maximums_performance_dictionaries = Parallel(
            n_jobs=n_jobs, verbose=1
        )(
            delayed(process_chunk, check_pickle=True)(
                data_chunk, algorithm, gpu_properties, autotuning_properties
            )
            for data_chunk in tqdm(pd.read_csv(data_file_raw, chunksize=chunk_size))
        )

        baseline_performance_dictionaries, maximums_performance_dictionaries = zip(
            *baseline_and_maximums_performance_dictionaries
        )
        baseline_performance_dictionary = list_of_dics_to_dic_of_lists(
            baseline_performance_dictionaries
        )
        assert (
            0 not in baseline_performance_dictionary.values()
        ), "Found a max. performance of 0"
        maximums_performance_dictionary = list_of_dics_to_dic_of_lists(
            maximums_performance_dictionaries
        )
        assert (
            0 not in maximums_performance_dictionary.values()
        ), "Found a baseline performance of 0"

        # Write max performances to files
        max_performances = dict()
        print("\nComputing maximum performances ...")
        for mnk, max_list in maximums_performance_dictionary.items():
            perf = get_max(max_list)
            max_performances[mnk] = perf
        with open(max_performances_per_algo_file, "w") as f:
            json.dump(max_performances, f, indent="\t", sort_keys=True)
        print("Wrote maximum performances to:\n", max_performances_per_algo_file)

        # Write baseline performances to files
        baseline_performances = dict()

        def get_baseline_performance(mnk, base_list, raw_pars_cols):
            perf = get_non_null(base_list)
            if perf == 0:
                data_file = os.path.join(
                    data_path, "raw_training_data_" + algorithm + ".csv"
                )
                data = dd.read_csv(data_file)
                perf = get_performance_closest_to_baseline(
                    data, algorithm, mnk, gpu_properties, autotuning_properties
                )
            return perf

        print("\nComputing baseline performances ...")
        baseline_performances_ = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(get_baseline_performance, check_pickle=True)(
                mnk, base_list, raw_pars_cols
            )
            for mnk, base_list in tqdm(baseline_performance_dictionary.items())
        )

        baseline_performances = dict(
            zip(baseline_performance_dictionary.keys(), baseline_performances_)
        )
        with open(baseline_performances_per_algo_file, "w") as f:
            json.dump(baseline_performances, f, indent="\t", sort_keys=True)
        print("Wrote baseline performances to:\n", baseline_performances_per_algo_file)


# ===============================================================================
def plot_baseline(baseline_perfs_by_algo, data_path, algorithms):
    import re
    import matplotlib.pyplot as plt

    print("\nPlotting baseline performances ...")

    # Get all mnks
    mnk_sequences = list()
    for algo, baseline_dic in baseline_perfs_by_algo.items():
        mnk_sequences += list(baseline_dic.keys())
    all_mnks = list(set.union(set(mnk_sequences)))

    # Reduce baseline_perfs_by_algo to baseline_perfs
    baseline_perfs = dict()
    for mnk in all_mnks:
        for algo in [
            "medium",
            "small",
            "largeDB1",
            "largeDB2",
            "tiny",
        ]:  # algorithms in order of baseline-ness
            if mnk in baseline_perfs_by_algo[algo].keys():
                baseline_perfs[mnk] = baseline_perfs_by_algo[algo][mnk]
                break
        else:
            assert (
                False
            ), "NOOOO this is actually impossible by def of all_mnks, isn't it?"

    # Sort
    mnks = list()
    mnk_str = re.compile(r"(\d+)x(\d+)x(\d+)")
    for mnk_s in baseline_perfs.keys():
        match = mnk_str.match(mnk_s)
        mnks.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))

    baseline_performances = zip(mnks, baseline_perfs.values())

    baseline_performances_sorted = [
        (mnk[0] * mnk[1] * mnk[2], p)
        for mnk, p in sorted(
            baseline_performances, key=lambda x: x[0][0] * x[0][1] * x[0][2]
        )
    ]
    mnk_sorted, baseline_perf_sorted = list(zip(*baseline_performances_sorted))

    # Plot
    plt.plot(mnk_sorted, baseline_perf_sorted, ".", markersize=1)
    plt.xlabel("(m, n, k) triplets of training data (in order of increasing m*n*k)")
    plt.ylabel("Baseline performances (Gflop/s)")
    plt.title("Baseline performances on training data")
    algorithm_extension = "_" + algorithms[0] if len(algorithms) == 0 else ""
    file_name = os.path.join(
        data_path, "baseline_performances" + algorithm_extension + ".svg"
    )
    plt.savefig(file_name)
    print("... wrote to", file_name)


def write_baseline_record(data_path, algorithms):
    baseline_performances_by_algo_file = os.path.join(
        data_path, "baseline_performances_by_algo.json"
    )
    if os.path.exists(baseline_performances_by_algo_file):
        print("Found {:40}, skipping".format(baseline_performances_by_algo_file))
        with open(baseline_performances_by_algo_file) as f:
            baseline_performances_by_algo = json.load(f)

    else:
        print(
            "File {:40} not found, generating".format(
                baseline_performances_by_algo_file
            )
        )
        # Get baseline performances by algorithm
        baseline_performances_by_algo = dict()
        for algorithm in algorithms:
            # Read baseline parameters
            baseline_performances_per_algo_file = os.path.join(
                data_path, "baseline_performances_" + algorithm + ".json"
            )
            with open(baseline_performances_per_algo_file, "r") as f:
                baseline_algorithm = json.load(f)
            # Add to dictionary
            baseline_performances_by_algo[algorithm] = baseline_algorithm

        # Write to file
        with open(baseline_performances_by_algo_file, "w") as f:
            json.dump(baseline_performances_by_algo, f, indent="\t", sort_keys=True)
        print("\nWrote baseline performances to:\n", baseline_performances_by_algo_file)

    plot_baseline(baseline_performances_by_algo, data_path, algorithms)


def write_max_by_algo_record(data_path, algorithms):
    max_performances_by_algo_file = os.path.join(
        data_path, "max_performances_by_algo.json"
    )
    if os.path.exists(max_performances_by_algo_file):
        print("Found {:40}, skipping".format(max_performances_by_algo_file))

    else:
        # Get max performances by algorithm
        max_performances_by_algo = dict()
        for algorithm in algorithms:
            # Read max parameters
            max_performances_per_algo_file = os.path.join(
                data_path, "max_performances_" + algorithm + ".json"
            )
            with open(max_performances_per_algo_file, "r") as f:
                max_algorithm = json.load(f)
            # Add to dictionary
            max_performances_by_algo[algorithm] = max_algorithm

        # Write to file
        with open(max_performances_by_algo_file, "w") as f:
            json.dump(max_performances_by_algo, f, indent="\t", sort_keys=True)
        print(
            "\nWrote max performances by algorithm to:\n", max_performances_by_algo_file
        )


def plot_max_performances(max_perfs, data_path, algorithms):
    import re
    import matplotlib.pyplot as plt

    print("\nPlotting max. performances ...")

    mnks = list()
    mnk_str = re.compile(r"(\d+)x(\d+)x(\d+)")
    for mnk_s in max_perfs.keys():
        match = mnk_str.match(mnk_s)
        mnks.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))

    max_performances = zip(mnks, max_perfs.values())
    max_performances_sorted = [
        (mnk[0] * mnk[1] * mnk[2], p)
        for mnk, p in sorted(
            max_performances, key=lambda x: x[0][0] * x[0][1] * x[0][2]
        )
    ]
    mnk_sorted, max_perf_sorted = list(zip(*max_performances_sorted))

    # Plot
    plt.plot(mnk_sorted, max_performances_sorted, ".", markersize=1)
    plt.xlabel("(m, n, k) triplets of training data (in order of increasing m*n*k)")
    plt.ylabel("Max. performances (Gflop/s)")
    plt.title("Maximum performances on training data")
    algorithm_extension = "_" + algorithms[0] if len(algorithms) == 0 else ""
    file_name = os.path.join(
        data_path, "max_performances" + algorithm_extension + ".svg"
    )
    plt.savefig(file_name)
    print("... wrote to", file_name)


def write_max_record(data_path, algorithms):
    max_performances_file = os.path.join(data_path, "max_performances.json")
    if os.path.exists(max_performances_file):
        print("Found {:40}, skipping".format(max_performances_file))
        with open(max_performances_file) as f:
            max_performances = json.load(f)

    else:
        # Get max performances
        max_performances_by_algo = dict()
        for algorithm in algorithms:
            # Read max parameters
            max_performances_per_algo_file = os.path.join(
                data_path, "max_performances_" + algorithm + ".json"
            )
            with open(max_performances_per_algo_file, "r") as f:
                max_algorithm = json.load(f)
            # Add to dictionary
            max_performances_by_algo[algorithm] = max_algorithm

        # Reduce along max
        max_performances_list = dic_of_dics_to_dic_of_lists(max_performances_by_algo)
        max_performances = dict()
        for mnk, max_list in max_performances_list.items():
            max_performances[mnk] = get_max(max_list)

        # Write to file
        with open(max_performances_file, "w") as f:
            json.dump(max_performances, f, indent="\t", sort_keys=True)
        print("\nWrote max performances to:\n", max_performances_file)

    plot_max_performances(max_performances, data_path, algorithms)


def get_derived_pars(
    data_path,
    i,
    data_chunk,
    algorithm,
    gpu_properties,
    autotuning_properties,
    max_performances,
):
    # Compute derived parameters
    data_chunk["algorithm"] = [algorithm] * len(
        data_chunk.index
    )  # add 'algorithm' column manually
    parameter_sets = PredictiveParameters(
        data_chunk, gpu_properties, autotuning_properties, max_performances
    )
    pars_to_get = derived_parameters["common"] + derived_parameters[algorithm]
    new_data = parameter_sets.get_features(pars_to_get)

    # Write to CSV
    filename = os.path.join(data_path, "training_data_{}-{}.csv".format(algorithm, i))
    new_data.to_csv(filename, index=False)

    return filename


def write_derived_data(data_path, algorithm, arch, n_jobs, chunk_size):
    """
    The predictive modelling procedure uses not only the raw parameters as features, but also some
    "derived" features computed using algorithm characteristics and hardware knowledge.
    This function reads raw parameters from `data_path`, computes derived parameters and writes them
    to the same folder.
    """
    derived_training_data_filename = os.path.join(
        data_path, "training_data_{}.csv".format(algorithm)
    )
    print(
        "\n\n------------------------------------------------------------------------"
    )
    if os.path.exists(derived_training_data_filename):
        print("Found {:40}, skipping".format(derived_training_data_filename))

    else:

        print("Didn't find {:40}, generating".format(derived_training_data_filename))

        # Read max performances, GPU properties and autotuning properties
        maxperf_file = os.path.join(data_path, "max_performances.json")
        with open(maxperf_file) as f:
            max_performances = json.load(f)
        with open("kernels/gpu_properties.json") as f:
            gpu_properties = json.load(f)["sm_" + str(arch)]
        with open("kernels/autotuning_properties.json") as f:
            autotuning_properties = json.load(f)

        # Compute derived data from raw data
        raw_training_data_filename = os.path.join(
            data_path, "raw_training_data_{}.csv".format(algorithm)
        )
        print(
            "reading raw data from {} and computing derived parameters".format(
                raw_training_data_filename
            )
        )

        derived_training_data_filenames = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(get_derived_pars, check_pickle=True)(
                data_path,
                i,
                data_chunk,
                algorithm,
                gpu_properties,
                autotuning_properties,
                max_performances,
            )
            for i, data_chunk in enumerate(
                pd.read_csv(raw_training_data_filename, chunksize=chunk_size)
            )
        )

        # Merge the CSV files (one for each iteration of the above Joblib loop) into one file
        assert len(derived_training_data_filenames) > 0, "No training data files"
        if len(derived_training_data_filenames) == 1:
            # No merging is necessary. Simply rename the file
            os.rename(
                derived_training_data_filenames[0], derived_training_data_filename
            )

        else:
            with open(derived_training_data_filename, "w") as out:
                # Write the first file, including its header
                fn_1 = derived_training_data_filenames.pop(0)
                with open(fn_1) as f:
                    out.write(f.read())
                os.remove(fn_1)
                # Write the rest of the files, skipping the header line each time
                for i, fn in enumerate(derived_training_data_filenames):
                    print(
                        "writing from {} ({}/{})".format(
                            fn, i + 1, len(derived_training_data_filenames)
                        )
                    )
                    with open(fn) as f:
                        next(f)  # skip header line
                        out.write(f.read())
                    # Delete the file we just merged
                    os.remove(fn)

        print("\tWrote", derived_training_data_filename)


# ===============================================================================
def main(data_path, algorithms_to_prep, arch, n_jobs, chunk_size, skip_derived_data):
    """
    This script is part of the workflow for predictive modelling of optimal libcusmm parameters.
    For more details, see predict.md

    """
    # ===============================================================================
    # Write baseline and maximum performance records
    for algorithm in algorithms_to_prep:
        write_baseline_and_max_records_per_algorithm(
            data_path, algorithm, arch, n_jobs, chunk_size
        )

    if set(algorithms_to_prep) == set(kernel_algorithm.keys()):
        write_baseline_record(data_path, algorithms_to_prep)
        write_max_by_algo_record(data_path, algorithms_to_prep)
        write_max_record(data_path, algorithms_to_prep)

    # ===============================================================================
    if not skip_derived_data:
        for algorithm in algorithms_to_prep:
            write_derived_data(data_path, algorithm, arch, n_jobs, chunk_size)
            write_to_parquet(data_path, algorithm)


# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Prepare the data collected with autotuning for training,
        After downloading raw data from the dedicated repository, use this script to
        - Record maximum and baseline performances of (m,n,k)-triplets in JSON files
        - Compute derived training data and write it to a CSV file
        - Compress training data csv files to parquet file format
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--folder",
        metavar="FOLDER",
        type=str,
        default=".",
        help="Path to the data to be converted to parquet.",
    )
    parser.add_argument(
        "-l",
        "--algorithm",
        metavar="ALGORITHM",
        default="",
        help="Algorithms to prepare",
    )
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCHITECTURE_NUMBER",
        type=int,
        default="60",
        help="CUDA architecture number. Options: sm_35, sm_37, sm_60, sm_70, gfx906",
    )
    parser.add_argument(
        "-j",
        "--njobs",
        default=-1,
        metavar="NUMBER",
        type=int,
        help="Number of parallel jobs that Joblib will launch. If you run into out-of-memory errors, reduce this.",
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        default=20000,
        help="Chunk size for dispatching joblib jobs. If memory errors are experienced, reduce this number",
    )
    parser.add_argument(
        "-s",
        "--skip_derived_data",
        type=bool,
        default=False,
        help="Skip the computation of derived data. Set to true if computing baseline & max records for each algoseparately",
    )

    args = parser.parse_args()
    algorithms_to_prep = (
        kernel_algorithm.keys() if args.algorithm == "" else [args.algorithm]
    )
    main(
        args.folder,
        algorithms_to_prep,
        args.arch,
        args.njobs,
        args.chunk_size,
        args.skip_derived_data,
    )
