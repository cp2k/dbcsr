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

import os
import sys
import json
import pandas as pd
from optparse import OptionParser
from kernels.cusmm_predict import (
    get_max_performances_per_mnk,
    get_baseline_performances_per_mnk,
    to_string,
    PredictiveParameters,
    derived_parameters,
    kernel_algorithm,
)


# ===============================================================================
def main():
    """
    This script is part of the workflow for predictive modelling of optimal libcusmm parameters.
    For more details, see predict.md.

    After downloading raw data from the dedicated repository, use this script to
    - Compute derived training data and write it to a CSV file
    - Record maximum and baseline performances of (m,n,k)-triplets in JSON files
    """

    parser = OptionParser()
    parser.add_option(
        "-f",
        "--folder",
        metavar="FOLDER",
        default="../../../../../dbcsr-data/P100/",
        help="Folder in which the raw downloaded data are to be found. Default: %default",
    )
    parser.add_option(
        "-a",
        "--arch",
        metavar="FOLDER",
        default="60",
        help="CUDA architecture number. Options: 35, 37, 60. Default: %default",
    )

    options, args = parser.parse_args(sys.argv)

    # ===============================================================================
    # Read GPU properties and autotuning properties
    arch = options.arch
    with open("kernels/gpu_properties.json") as f:
        gpu_properties = json.load(f)["sm_" + str(arch)]
    with open("kernels/autotuning_properties.json") as f:
        autotuning_properties = json.load(f)

    # ===============================================================================
    # Loop over algorithms
    max_performances_per_mnk = dict()
    baseline_performances_per_algo_per_mnk = {
        "tiny": dict(),
        "small": dict(),
        "medium": dict(),
        "largeDB1": dict(),
        "largeDB2": dict(),
    }
    for name_algo, kernel_algo in kernel_algorithm.items():

        raw_training_data_filename = os.path.join(options.folder, "raw_training_data_{}.csv".format(name_algo))
        print("\nReading from {}".format(raw_training_data_filename))

        # Read CSV and loop over chunks
        chunk_size = 10000  # Number of rows of CSV file to process at a time
        chunk_count = 0

        for data_chunk in pd.read_csv(raw_training_data_filename, chunksize=chunk_size):

            # Print progress
            chunk_count += 1
            print("Read chunk {:5>}".format(chunk_count))

            # Get max_performance_per_mnk
            max_performances = get_max_performances_per_mnk(data_chunk)
            max_performances_per_mnk.update(dict(zip(to_string(*max_performances.keys()), max_performances.values())))

            # Get baseline_per_mnk
            baseline_performances_algo = get_baseline_performances_per_mnk(data_chunk, name_algo, gpu_properties,
                                                                           autotuning_properties)
            baseline_performances_per_algo_per_mnk[name_algo].update(
                dict(zip(to_string(*baseline_performances_algo.keys()), baseline_performances_algo.values())))

            # Compute derived parameters
            data_chunk["algorithm"] = [name_algo] * len(data_chunk.index)  # add 'algorithm' column manually
            parameter_sets = PredictiveParameters(data_chunk, gpu_properties, autotuning_properties, max_performances)
            pars_to_get = derived_parameters["common"] + derived_parameters[name_algo]
            new_data = parameter_sets.get_features(pars_to_get)

            # Write derived parameters
            derived_training_data_filename = os.path.join(options.folder, "training_data_{}_{}.csv".format(
                name_algo, chunk_count - 1))
            new_data[pars_to_get].to_csv(derived_training_data_filename, index=False)
            print("\tWrote", derived_training_data_filename)

    # ===============================================================================
    print("\nRead all raw and computed all derived data")

    # Print header lines & merge instructions
    print("\n$ # Merge instructions:")
    print("$ cd {}".format(options.folder))
    for name_algo, kernel_algo in kernel_algorithm.items():

        # Print header line
        derived_training_data_filename_base = "training_data_{}_{}.csv"
        derived_training_data_filename_chunk = derived_training_data_filename_base.format(name_algo, 0)
        with open(derived_training_data_filename_chunk, "r") as f:
            header_line = f.readline()
        derived_training_data_filename = "training_data_{}.csv".format(name_algo)
        with open(derived_training_data_filename, "w") as f:
            f.write(header_line)
        print("$ # Wrote header line to {}".format(derived_training_data_filename))

        # Print merge instructions
        print("$ # Wrote header line to {}".format(derived_training_data_filename))
        print("$ # Append training data chunks to {} by running:".format(derived_training_data_filename))
        derived_training_data_filename_wildcard = derived_training_data_filename_base.format(name_algo, "*")
        print("$ tail -n +2 -q {to_merge} >> {training_data_file}".format(
            to_merge=derived_training_data_filename_wildcard, training_data_file=derived_training_data_filename))

    # Print max performances
    max_performances_per_mnk_file = os.path.join(options.folder, "max_performances.json")
    with open(max_performances_per_mnk_file, "w") as f:
        json.dump(max_performances_per_mnk, f)
    print("\nWrote maximum performances to:\n", max_performances_per_mnk_file)

    # Print baseline
    baseline_performances_per_algo_per_mnk_file = os.path.join(options.folder, "baseline_performances_by_algo.json")
    with open(baseline_performances_per_algo_per_mnk_file, "w") as f:
        json.dump(baseline_performances_per_algo_per_mnk, f)
    print("\nWrote baseline performances to:\n", baseline_performances_per_algo_per_mnk_file)


# ===============================================================================
main()
