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
    parameter_types,
)


# ===============================================================================
def main(tunedir):
    """
    This script is part of the workflow for predictive modelling of optimal libcusmm parameters.
    For more details, see predict.md

    Once autotuning of new kernels has been run,
    - collect the parameter information and performance from log files,
    - dump them to CSV files for data analysis and training of a predictive model
    """
    # ===============================================================================
    # Find all the 'tune_MxNxK' folders
    kernel_folder_pattern = re.compile(r"tune_(\d+)x(\d+)x(\d+)$")
    kernel_folders = [
        os.path.join(tunedir, ak)
        for ak in os.listdir(tunedir)
        if kernel_folder_pattern.match(ak) is not None
    ]
    n_kernels = len(kernel_folders)
    assert n_kernels > 0, "Found no kernel folders of format" + str(kernel_folder_pattern) + " in folder " + tunedir
    print("Found {:,} kernel folders".format(n_kernels))

    # Collect information and write to csv
    collect_training_data(
        kernel_folders,
        kernel_folder_pattern,
    )

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
                        "threads": match.group(9),
                        "grouping": match.group(10),
                        "minblocks": match.group(11),
                        "tile_m": match.group(5)
                        if match.group(5) is not None
                        else None,
                        "tile_n": match.group(6)
                        if match.group(6) is not None
                        else None,
                        "w": match.group(7)
                        if match.group(7) is not None
                        else None,
                        "v": match.group(8)
                        if match.group(8) is not None
                        else None,
                        "perf (Gflop/s)": match.group(12),
                    }
                )

    print("Autotuning lines found: ", len(data))

    # Merge dictionaries into a pandas dataframe
    dataframe = pd.DataFrame(data)
    for col in dataframe.columns:
        dataframe[col] = dataframe[col].astype(parameter_types[col], errors='ignore')

    return dataframe


def collect_training_data(
    kernel_folders,
    kernel_folder_pattern,
):
    """
    Collect training data from log files resulting of autotuning
    """

    # ===============================================================================
    # For each folder:
    n_kernels = len(kernel_folders)
    for i, kernel_folder in enumerate(kernel_folders):

        print("\nProcess folder {} ({}/{:,})".format(kernel_folder, i + 1, n_kernels))

        # Find (m, n, k)
        # Each folder contains data for just one (m, n, k) but potentially mutliple algorithms
        match = kernel_folder_pattern.search(kernel_folder).groups()
        m = int(match[0])
        n = int(match[1])
        k = int(match[2])

        # ===============================================================================
        # Collect info from log files
        data = read_log_file(kernel_folder, m, n, k)

        # ===============================================================================
        # Write parameters to CSV
        for name_algo, kernel_algo in kernel_algorithm.items():

            # if applicable to this mnk
            if name_algo in data["algorithm"].values:

                # Does collected csv file exist already?
                raw_parameters_file_name = os.path.join(
                    kernel_folder,
                    "raw_training_data_"
                    + to_string(m, n, k)
                    + "_"
                    + name_algo
                    + ".csv",
                )

                if os.path.exists(raw_parameters_file_name):
                    print(
                        "\tFound csv file:", raw_parameters_file_name, ", skipping ..."
                    )

                else:

                    # Get the data corresponding to this algorithm
                    data_algo = data[data["algorithm"] == name_algo]

                    # Write raw parameters
                    pars_to_get = kernel_algo.launch_parameters + ["perf (Gflop/s)"]
                    data_algo[pars_to_get].to_csv(raw_parameters_file_name, index=False)
                    print("\tWrote", raw_parameters_file_name)


# ===============================================================================
def print_merging_commands(kernel_folders, kernel_folder_pattern, tunedir):
    """
    Print commands to execute in order to merge CSV files
    """
    for algorithm in kernel_algorithm.keys():

        print(
            "\n$ # Merge instructions for algorithm",
            algorithm,
        )
        training_data_file = "raw_training_data_{algorithm}.csv".format(
            algorithm=algorithm
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
                    "raw_training_data_{mnk}_{algorithm}.csv".format(
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
                )
                continue

        print(
            "$ tail -n +2 -q {tunedir}tune_*/training_data_*_{algorithm}.csv >> {training_data_file}".format(
                tunedir=tunedir,
                algorithm=algorithm,
                training_data_file=training_data_file,
            )
        )


# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Collect matrix-matrix multiplication parameters and performances measured during autotuning. For that,
        parse the log files created by the autotuning and record parameter sets and their performances to CSV files.

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

    args = parser.parse_args()
    main(args.folder)
