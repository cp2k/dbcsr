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


# ===============================================================================
# I/O
# kernel_folder_pattern = re.compile('tune_(\d+)x(\d+)x(\d+)$')
kernel_folder_pattern = re.compile(r"tune_(\d+x\d+x\d+)$")
page_width = 5  # columns per output line


def check_autotuning_data_path(autotuning_data_path):
    # sanity checks
    assert os.path.exists(autotuning_data_path), (
        "This path does not exist: " + autotuning_data_path
    )
    assert len(os.listdir(autotuning_data_path)) > 0, (
        "No folders found in path: " + autotuning_data_path
    )
    # print infos
    print(
        "Number of tuning data folders found: {}".format(
            len(os.listdir(autotuning_data_path))
        )
    )


def get_folders_to_read(to_read, autotuning_data_path):
    if to_read == "all":
        folders_to_read = [
            os.path.join(autotuning_data_path, f)
            for f in os.listdir(autotuning_data_path)
            if kernel_folder_pattern.match(f) is not None
        ]
    elif isinstance(to_read, int):
        folders_to_read = [
            os.path.join(autotuning_data_path, f)
            for f in os.listdir(autotuning_data_path)
            if kernel_folder_pattern.match(f) is not None
        ]
        folders_to_read = folders_to_read[:to_read]
    elif isinstance(to_read, str):
        to_read = re.compile(to_read)
        folders_to_read = [
            os.path.join(autotuning_data_path, f)
            for f in os.listdir(autotuning_data_path)
            if to_read.match(f) is not None
        ]
    else:
        assert False, "Cannot recognize option: " + to_read

    num_folders_to_read = len(folders_to_read)
    assert num_folders_to_read > 0
    print("Data folders to be read from (total: {:,})\n".format(num_folders_to_read))
    for f in folders_to_read:
        print(f)

    return folders_to_read


def get_algorithm_to_explore(algo):
    algo_to_read = (
        [algo] if algo != "all" else ["tiny", "small", "medium", "largeDB1", "largeDB2"]
    )
    print("Algorithm(s) to explore:")
    for a in algo_to_read:
        print(a)

    return algo_to_read


def get_files_to_read(folders_to_read, algo_to_read):
    files_to_read = list()
    for i, kernel_folder in enumerate(folders_to_read):
        print(
            "\nfrom {}, read                                  ({}/{:,})".format(
                kernel_folder, i + 1, len(folders_to_read)
            )
        )

        for name_algo in algo_to_read:

            mnk_string = kernel_folder_pattern.search(kernel_folder).groups()[0]
            raw_file_base = "raw_training_data_" + mnk_string + "_" + name_algo + ".csv"
            raw_file = os.path.join(kernel_folder, raw_file_base)
            derived_file_base = "training_data_" + mnk_string + "_" + name_algo + ".csv"
            derived_file = os.path.join(kernel_folder, derived_file_base)

            if os.path.exists(raw_file) and os.path.exists(derived_file):

                # Read raw parameters file
                files_to_read.append(raw_file)

                # Read derived parameters file
                files_to_read.append(derived_file)

            else:

                if not os.path.exists(raw_file):
                    print("\t...{:50} no file".format(raw_file_base))
                if not os.path.exists(derived_file):
                    print("\t...{:50} no file".format(derived_file_base))

    return files_to_read
