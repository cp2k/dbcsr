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
import re
import sys
import json
import pandas as pd
from optparse import OptionParser
from kernels.cusmm_dnt_helper import *


# ===============================================================================
def main():
    """
    Once autotuning of new kernels has been run,
    - collect the parameter information, compilation information and performance from log files,
    - dump them to CSV files for data analysis and training of a predictive model
    """

    parser = OptionParser()
    parser.add_option("-f", "--folder", metavar="FOLDER", default=".",
                      help="Folder in which the folders tune_*x*x*x/ are to be found. Default: %default")
    parser.add_option("-a", "--arch", metavar="FOLDER", default="60",
                      help="CUDA architecture number. Options: 35, 37, 60. Default: %default")

    options, args = parser.parse_args(sys.argv)

    # ===============================================================================
    # Read GPU properties and autotuning properties
    arch = options.arch
    with open('kernels/gpu_properties.json') as f:
        gpu_properties = json.load(f)["sm_" + str(arch)]
    with open('kernels/autotuning_properties.json') as f:
        autotuning_properties = json.load(f)

    # ===============================================================================
    # Find all the tune_MxNxK folders
    kernel_folder_pattern = re.compile('tune_(\d+)x(\d+)x(\d+)$')
    mnk_string = '{}x{}x{}'
    kernel_folders = [os.path.join(options.folder, ak) for ak in os.listdir(options.folder)
                      if kernel_folder_pattern.match(ak) is not None]
    n_kernels = len(kernel_folders)
    print('Found {:,} kernel folders'.format(n_kernels))
    max_performances_per_mnk = dict()
    max_performances_per_algo_per_mnk = {'tiny': dict(), 'small': dict(), 'medium': dict(),
                                         'largeDB1': dict(), 'largeDB2': dict()}
    baseline_performances_per_algo_per_mnk = {'tiny': dict(), 'small': dict(), 'medium': dict(),
                                              'largeDB1': dict(), 'largeDB2': dict()}

    # ===============================================================================
    # Collect information and write to csv
    # For each folder:
    for i, kernel_folder in enumerate(kernel_folders):

        print('\nProcess folder {} ({}/{:,})'.format(kernel_folder, i + 1, n_kernels))

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
        max_performances_per_mnk.update(dict(zip(to_string(*max_performances.keys()), max_performances.values())))

        # ===============================================================================
        # Write parameters to CSV
        for name_algo, kernel_algo in kernel_algorithm.items():

            # if relevant for this mnk
            if name_algo in data['algorithm'].values:

                data_algo = data[data['algorithm'] == name_algo]

                # Collect max performances per algo, per (m, n, k)
                max_performances_algo = get_max_performances_per_mnk(data_algo)
                max_performances_per_algo_per_mnk[name_algo].update(
                    dict(zip(to_string(*max_performances_algo.keys()), max_performances_algo.values())))

                # Collect baseline performances per algo, per (m, n, k)
                baseline_performances_algo = get_baseline_performances_per_mnk(data_algo, name_algo)
                baseline_performances_per_algo_per_mnk[name_algo].update(
                    dict(zip(to_string(*baseline_performances_algo.keys()), baseline_performances_algo.values())))

                # Does collected csv file exist already?
                raw_parameters_file_name = os.path.join(
                    kernel_folder, 'raw_training_data_' + mnk_string.format(m, n, k) + '_' + name_algo + '.csv')

                derived_parameters_file_name = os.path.join(
                    kernel_folder, 'training_data_' + mnk_string.format(m, n, k) + '_' + name_algo + '.csv')

                if os.path.exists(raw_parameters_file_name) and os.path.exists(derived_parameters_file_name):
                    print('Found csv files:', raw_parameters_file_name, ', skipping ...')

                else:

                    # Write raw parameters
                    pars_to_get = raw_parameters
                    data_algo[pars_to_get].to_csv(raw_parameters_file_name)
                    print('Wrote', raw_parameters_file_name)

                    # Compute derived parameters
                    parameter_sets = PredictiveParameters(data_algo,
                                                          gpu_properties, autotuning_properties, max_performances)
                    pars_to_get = derived_parameters['common'] + derived_parameters[name_algo]
                    new_df = parameter_sets.get_features(pars_to_get)
                    data_algo.merge(new_df)

                    # Write derived parameters
                    data_algo[pars_to_get].to_csv(derived_parameters_file_name)
                    print('Wrote', derived_parameters_file_name)

    # ===============================================================================
    # Print max performance dictionaries
    max_performances_per_mnk_file = os.path.join(options.folder, 'max_performances.json')
    with open(max_performances_per_mnk_file, 'w') as f:
        json.dump(max_performances_per_mnk, f)
    max_performances_per_algo_per_mnk_file = os.path.join(options.folder, 'max_performances_by_algo.json')
    with open(max_performances_per_algo_per_mnk_file, 'w') as f:
        json.dump(max_performances_per_algo_per_mnk, f)
    baseline_performances_per_algo_per_mnk_file = os.path.join(options.folder, 'baseline_performances_by_algo.json')
    with open(baseline_performances_per_algo_per_mnk_file, 'w') as f:
        json.dump(baseline_performances_per_algo_per_mnk, f)
    print('Wrote max. and baseline performances to:\n',
          max_performances_per_mnk_file, ',\n',
          max_performances_per_algo_per_mnk_file, ' and\n',
          baseline_performances_per_algo_per_mnk_file)

    # ===============================================================================
    # TODO
    # Search for already existing training data
    # if none:
        # if a lot are there: give command for concatenating all of these CSVs
        # if not a lot, just do it in Python
    # if existing:
        # Read it (in chunks?)
            # New data: just add it in
            # Overlapping data:
                # Give three options: 1) all new, 2) all old, 3) pick line by line


# ===============================================================================
# Helper variables and functions (formatting & writing)
autotuning_line = re.compile(
    'OK Kernel_dnt_(\w+) m (\d+)\s+n (\d+)\s+k (\d+)\s+' +
    '(?:tile_m (\d+)\s+tile_n (\d+)\s+(?:w (\d+)\s+v (\d+)\s+)?)?' +
    'threads (\d+)\s+grouping (\d+)\s+minblocks (\d+)\s+GFlop/s (\d+(?:\.\d+)?)'
)


def read_log_file(log_folder, m, n, k):
    """
    Given a folder of kernel autotuning, read in autotuning information
    :param log_folder: folder of kernel autotuning
    :return: dictionary containing above information
    """
    # Autotuning information (performance)
    log_files = [f for f in os.listdir(log_folder) if f[-4:] == '.log']
    assert len(log_files) > 0
    log_files = sorted(log_files)
    print('Found log files:', log_files)

    # Collect data
    data = dict()
    for log_file in log_files:

        print('Processing log file', log_file)
        with open(os.path.join(log_folder, log_file), 'r') as f:
            log_file_content = f.read().splitlines()

        for l in log_file_content:

            if 'OK' in l:

                # Get algo, parameters, and performance
                match = autotuning_line.match(l)
                assert match is not None, "Found null match: " + l
                algo = match.group(1)
                tile_m = int(match.group(5)) if match.group(5) is not None else None
                tile_n = int(match.group(6)) if match.group(6) is not None else None
                w = int(match.group(7)) if match.group(7) is not None else None
                v = int(match.group(8)) if match.group(8) is not None else None
                threads = int(match.group(9))
                grouping = int(match.group(10))
                minblocks = int(match.group(11))
                perf = float(match.group(12))
                assert perf > 0, "Found a kernel with 0 performance:\n" + l

                # Add to dictionary
                key = (algo, m, n, k, tile_m, tile_n, w, v, threads, grouping, minblocks)
                if key in data:
                    data[key]['perf (Gflop/s)'] = perf
                else:
                    data[key] = {'perf (Gflop/s)': perf,
                                 'nregs': None,
                                 'nbytes_smem': None,
                                 'nbytes_cmem': None}

    # Assemble and return pandas dataframe
    print('Autotuning lines found: ', len(data))

    # Merge dictionaries into a pandas dataframe
    data_ = list()
    for kernel_pars in data.keys():
        pars_dict = {
            'm': kernel_pars[1], 'n': kernel_pars[2], 'k': kernel_pars[3],
            'algorithm': kernel_pars[0],
            'threads_per_blk': kernel_pars[8], 'grouping': kernel_pars[9], 'minblocks': kernel_pars[10],
            'tile_m': kernel_pars[4], 'tile_n': kernel_pars[5],
            'w': kernel_pars[6], 'v': kernel_pars[7],
            'perf (Gflop/s)': data[kernel_pars]['perf (Gflop/s)'],
        }
        data_.append(pars_dict)

    data = pd.DataFrame(data_)

    return data


# ===============================================================================
main()
