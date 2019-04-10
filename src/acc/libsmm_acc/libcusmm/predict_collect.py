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
import re
import json
import argparse
import pandas as pd
from kernels.cusmm_predict import (
    get_max_performances_per_mnk,
    get_baseline_performances_per_mnk,
    to_string,
    PredictiveParameters,
    derived_parameters,
    kernel_algorithm,
)


# ===============================================================================
def main(tunedir, arch):
    """
    This script is part of the workflow for predictive modelling of optimal libcusmm parameters.
    For more details, see predict.md

    Once autotuning of new kernels has been run,
    - collect the parameter information and performance from log files,
    - dump them to CSV files for data analysis and training of a predictive model
    - write the max_performances and baseline_performances to JSON files
    """
    # ===============================================================================
    # Read GPU properties and autotuning properties
    with open("kernels/gpu_properties.json") as f:
        gpu_properties = json.load(f)["sm_" + str(arch)]
    with open("kernels/autotuning_properties.json") as f:
        autotuning_properties = json.load(f)

    # ===============================================================================
    # Find all the 'tune_MxNxK' folders
    kernel_folder_pattern = re.compile(r"tune_(\d+)x(\d+)x(\d+)$")
    kernel_folders = [
        os.path.join(tunedir, ak)
        for ak in os.listdir(tunedir)
        if kernel_folder_pattern.match(ak) is not None
    ]
    n_kernels = len(kernel_folders)
    print("Found {:,} kernel folders".format(n_kernels))
    max_performances_per_mnk = dict()
    baseline_performances_per_algo_per_mnk = {
        "tiny": dict(),
        "small": dict(),
        "medium": dict(),
        "largeDB1": dict(),
        "largeDB2": dict(),
    }

    # ===============================================================================
    # Collect information and write to csv
    collect_training_data(
        kernel_folders,
        kernel_folder_pattern,
        gpu_properties,
        autotuning_properties,
        max_performances_per_mnk,
        baseline_performances_per_algo_per_mnk,
    )

    # ===============================================================================
    # Print max performance dictionaries
    max_performances_per_mnk_file = os.path.join(tunedir, "max_performances.json")
    if os.path.exists(max_performances_per_mnk_file):
        # If this file already exists, read its contents and merge them with the content to write
        print(
            "Found {}, reading and merging into new data...".format(
                max_performances_per_mnk_file
            )
        )
        with open(max_performances_per_mnk_file, "r") as f:
            max_performances_per_mnk_to_merge = json.load(f)
            max_performances_per_mnk.update(max_performances_per_mnk_to_merge)
    with open(max_performances_per_mnk_file, "w") as f:
        json.dump(max_performances_per_mnk, f)
    print("\nWrote maximum performances to:\n", max_performances_per_mnk_file)

    # Print baseline performance dictionaries
    baseline_performances_per_algo_per_mnk_file = os.path.join(
        tunedir, "baseline_performances_by_algo.json"
    )
    if os.path.exists(baseline_performances_per_algo_per_mnk_file):
        # If this file already exists, read its contents and merge them with the content to write
        print(
            "Found {}, reading and merging into new data...".format(
                baseline_performances_per_algo_per_mnk_file
            )
        )
        with open(baseline_performances_per_algo_per_mnk_file, "r") as f:
            baseline_performances_per_algo_per_mnk_to_merge = json.load(f)
            baseline_performances_per_algo_per_mnk.update(
                baseline_performances_per_algo_per_mnk_to_merge
            )
    with open(baseline_performances_per_algo_per_mnk_file, "w") as f:
        json.dump(baseline_performances_per_algo_per_mnk, f)
    print(
        "\nWrote baseline performances to:\n",
        baseline_performances_per_algo_per_mnk_file,
    )

    # ===============================================================================
    # Print commands to merge CSVs into one big CSV for training data
    print(
        "Merge all individual CSV files into one by running the following commands:\n"
    )
    print_merging_commands(kernel_folders, kernel_folder_pattern, tunedir)


# ===============================================================================
# Helper variables and functions (formatting & writing)
autotuning_line = re.compile(
    r"OK Kernel_dnt_(\w+) m (\d+)\s+n (\d+)\s+k (\d+)\s+"
    + r"(?:tile_m (\d+)\s+tile_n (\d+)\s+(?:w (\d+)\s+v (\d+)\s+)?)?"
    + r"threads (\d+)\s+grouping (\d+)\s+minblocks (\d+)\s+GFlop/s (\d+(?:\.\d+)?)"
)


def read_log_file(log_folder, m, n, k):
    """
    Given a folder of kernel autotuning, read and parse the autotuning information in the log file
    and return it in the form of a pandas Dataframe.
    :param log_folder: folder of kernel autotuning
    :return: pandas Dataframe containing autotuning information
    """
    # Find log files in the log folder
    log_files = [f for f in os.listdir(log_folder) if f[-4:] == ".log"]
    assert len(log_files) > 0
    log_files = sorted(log_files)

    # Parse the log files and collect data
    data = list()
    for log_file in log_files:

        print("Processing log file", log_file)
        with open(os.path.join(log_folder, log_file), "r") as f:
            log_file_content = f.read().splitlines()

        for l in log_file_content:

            if "OK" in l:  # this line contains autotuning data

                # Parse the line
                match = autotuning_line.match(l)
                assert match is not None, "Found null match: " + l

                # Get algorithm, parameters, and performance
                data.append(
                    {
                        "m": m,
                        "n": n,
                        "k": k,
                        "algorithm": match.group(1),
                        "threads": int(match.group(9)),
                        "grouping": int(match.group(10)),
                        "minblocks": int(match.group(11)),
                        "tile_m": int(match.group(5))
                        if match.group(5) is not None
                        else None,
                        "tile_n": int(match.group(6))
                        if match.group(6) is not None
                        else None,
                        "w": int(match.group(7))
                        if match.group(7) is not None
                        else None,
                        "v": int(match.group(8))
                        if match.group(8) is not None
                        else None,
                        "perf (Gflop/s)": float(match.group(12)),
                    }
                )

    print("Autotuning lines found: ", len(data))

    # Merge dictionaries into a pandas dataframe
    dataframe = pd.DataFrame(data)

    return dataframe


def collect_training_data(
    kernel_folders,
    kernel_folder_pattern,
    gpu_properties,
    autotuning_properties,
    max_performances_per_mnk,
    baseline_performances_per_algo_per_mnk,
):
    """
    Collect training data from log files resulting of autotuning
    """

    n_kernels = len(kernel_folders)

    # For each folder:
    for i, kernel_folder in enumerate(kernel_folders):

        print("\nProcess folder {} ({}/{:,})".format(kernel_folder, i + 1, n_kernels))

        # Find (m, n, k)
        match = kernel_folder_pattern.search(kernel_folder).groups()
        m = int(match[0])
        n = int(match[1])
        k = int(match[2])

        # ===============================================================================
        # Collect info from log files
        data = read_log_file(kernel_folder, m, n, k)

        # Collect max performances per (m, n, k)
        max_performances = get_max_performances_per_mnk(data)
        max_performances_per_mnk.update(
            dict(zip(to_string(*max_performances.keys()), max_performances.values()))
        )

        # ===============================================================================
        # Write parameters to CSV
        for name_algo, kernel_algo in kernel_algorithm.items():

            # if applicable to this mnk
            if name_algo in data["algorithm"].values:

                # Get the data corresponding to this algorithm
                data_algo = data[data["algorithm"] == name_algo]

                # Collect baseline performances per algo, per (m, n, k)
                baseline_performances_algo = get_baseline_performances_per_mnk(
                    data_algo, name_algo, gpu_properties, autotuning_properties
                )
                baseline_performances_per_algo_per_mnk[name_algo].update(
                    dict(
                        zip(
                            to_string(*baseline_performances_algo.keys()),
                            baseline_performances_algo.values(),
                        )
                    )
                )

                # Does collected csv file exist already?
                raw_parameters_file_name = os.path.join(
                    kernel_folder,
                    "raw_training_data_"
                    + to_string(m, n, k)
                    + "_"
                    + name_algo
                    + ".csv",
                )
                derived_parameters_file_name = os.path.join(
                    kernel_folder,
                    "training_data_" + to_string(m, n, k) + "_" + name_algo + ".csv",
                )

                if os.path.exists(raw_parameters_file_name):
                    print(
                        "\tFound csv file:", raw_parameters_file_name, ", skipping ..."
                    )

                else:

                    # Write raw parameters
                    pars_to_get = kernel_algo.launch_parameters + ["perf (Gflop/s)"]
                    data_algo[pars_to_get].to_csv(raw_parameters_file_name, index=False)
                    print("\tWrote", raw_parameters_file_name)

                if os.path.exists(derived_parameters_file_name):
                    print(
                        "\tFound csv file:",
                        derived_parameters_file_name,
                        ", skipping ...",
                    )

                else:
                    # Compute derived parameters
                    parameter_sets = PredictiveParameters(
                        data_algo,
                        gpu_properties,
                        autotuning_properties,
                        max_performances,
                    )
                    pars_to_get = (
                        derived_parameters["common"] + derived_parameters[name_algo]
                    )
                    new_df = parameter_sets.get_features(pars_to_get)
                    data_algo.merge(new_df)

                    # Write derived parameters
                    data_algo[pars_to_get].to_csv(
                        derived_parameters_file_name, index=False
                    )
                    print("\tWrote", derived_parameters_file_name)


# ===============================================================================
def print_merging_commands(kernel_folders, kernel_folder_pattern, tunedir):
    """
    Print commands to execute in order to merge CSV files
    """
    for algorithm in kernel_algorithm.keys():
        for data_type in ("raw_", ""):

            data_type_name = (
                "raw" if data_type == "raw_" else "for predictive modelling"
            )
            print(
                "\n$ # Merge instructions for algorithm",
                algorithm,
                "(",
                data_type_name,
                ")",
            )
            training_data_file = "{data_type}training_data_{algorithm}.csv".format(
                data_type=data_type, algorithm=algorithm
            )

            if os.path.exists(training_data_file):
                print(
                    "$ # Found {}, append new training data to this file:".format(
                        training_data_file
                    )
                )

            else:

                # Find an (m, n, k) for this algorithm to get its header line
                for i, kernel_folder in enumerate(kernel_folders):

                    # Find (m, n, k)
                    match = kernel_folder_pattern.search(kernel_folder).groups()
                    m = int(match[0])
                    n = int(match[1])
                    k = int(match[2])

                    file_name = os.path.join(
                        kernel_folder,
                        "{data_type}training_data_{mnk}_{algorithm}.csv".format(
                            data_type=data_type,
                            mnk=to_string(m, n, k),
                            algorithm=algorithm,
                        ),
                    )
                    if os.path.exists(file_name):
                        print(
                            "$ head -1 {base_file} > {training_data_file}".format(
                                base_file=file_name,
                                training_data_file=training_data_file,
                            )
                        )
                        break
                else:
                    print(
                        "None: did not find any existing files for algorithm",
                        algorithm,
                        "and data",
                        data_type_name,
                    )
                    continue

            print(
                "$ tail -n +2 -q {tunedir}tune_*/{data_type}training_data_*_{algorithm}.csv >> {training_data_file}".format(
                    tunedir=tunedir,
                    data_type=data_type,
                    algorithm=algorithm,
                    training_data_file=training_data_file,
                )
            )


# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Collect matrix-matrix multiplication parameters and performance from the log files resulting from autotuning.

        This script is part of the workflow for predictive modelling of optimal libcusmm parameters.
        For more details, see predict.md.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--folder",
        metavar="FOLDER",
        type=str,
        default=".",
        help="Folder in which the folders tune_*x*x*x/ are to be found",
    )
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCHITECTURE_NUMBER",
        type=int,
        default=60,
        help="CUDA architecture number. Options: 35, 37, 60",
    )

    args = parser.parse_args()
    main(args.folder, args.arch)
