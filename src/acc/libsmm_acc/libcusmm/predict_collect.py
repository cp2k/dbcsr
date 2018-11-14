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


import sys
import os
import os
import re
import pickle
import sys
import math
import json
import numpy as np
import pandas as pd
from optparse import OptionParser
from kernels.cusmm_dnt_helper import *


# ===============================================================================
def main():
    """
    Once autotuning of new kernels has been run, collect the parameter information, compilation information and
    performance from slurm.out and log files, and dump them to CSV for model training
    """

    parser = OptionParser()
    parser.add_option("-f", "--folder", metavar="FOLDER", default=".",
                      help="Folder in which the folders tune_*x*x*x/ are to be found. Default: %default")
    parser.add_option("-a", "--arch", metavar="FOLDER", default="60",
                      help="CUDA architecture number. Options: 35, 37, 60. Default: %default")

    options, args = parser.parse_args(sys.argv)

    # ===============================================================================
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
    print('Found', n_kernels, 'kernel folders')

    # ===============================================================================
    # For each folder:
    for i, kernel_folder in enumerate(kernel_folders):

        print('\nProcess folder', kernel_folder, '(', i + 1, '/', n_kernels, ')')

        # Find (m, n, k)
        match = kernel_folder_pattern.search(kernel_folder).groups()
        m = int(match[0])
        n = int(match[1])
        k = int(match[2])

        # Does collected csv file exist already?
        data_csv = os.path.join(kernel_folder, 'training_data_' + mnk_string.format(m, n, k) + '.csv')
        if os.path.exists(data_csv):
            print('Found csv file:', data_csv, ', skipping ...')

        else:

            # Collect info from slurm output and log files
            data_slurm = read_slurm_file(kernel_folder)
            data_log = read_log_file(kernel_folder, m, n, k)

            # Merge dictionaries carefully into a pandas dataframe
            assert len(data_slurm) == len(data_log)
            data = list()
            for kernel_pars, compile_info in data_slurm.items():
                data.append({
                    'm': kernel_pars[1], 'n': kernel_pars[2], 'k': kernel_pars[3],
                    'algorithm': kernel_pars[0],
                    'threads_per_blk': kernel_pars[8], 'grouping': kernel_pars[9], 'minblocks': kernel_pars[10],
                    'tile_m': kernel_pars[4], 'tile_n': kernel_pars[5],
                    'w': kernel_pars[6], 'v': kernel_pars[7],
                    'perf (Gflop/s)': data_log[kernel_pars]['perf (Gflop/s)'],
                    'regs_per_thread': compile_info['nregs'],
                    'nbytes_smem': compile_info['nbytes_smem'],
                    'nbytes_cmem': compile_info['nbytes_cmem']
                })
            data = pd.DataFrame(data)

            # Collect max performances per (m, n, k)
            max_performances = get_max_performances_per_mnk(data)

            # Write parameters to CSV
            for name_algo, kernel_algo in kernel_algorithm.items():

                # if relevant for this mnk
                if name_algo in data['algorithm'].values:

                    data_algo = data[data['algorithm'] == name_algo]

                    for use_compile_info in [True, False]:

                        # Write raw parameters
                        raw_parameters_file_name = os.path.join(
                            kernel_folder, 'raw_training_data_' + mnk_string.format(m, n, k) + '_' + name_algo + '.csv')
                        if use_compile_info:
                            raw_parameters_file_name = raw_parameters_file_name.replace('.csv', '_withcompileinfo.csv')
                            data_algo[raw_parameters_withcompileinfo].to_csv(raw_parameters_file_name)
                        else:
                            data_algo[raw_parameters].to_csv(raw_parameters_file_name)
                        print('Wrote', raw_parameters_file_name)

                        # Compute derived parameters
                        parameter_sets = PredictiveParameters(data_algo,
                                                              gpu_properties, autotuning_properties, max_performances)
                        pars_to_get = derived_parameters['common'] + derived_parameters[name_algo]
                        if use_compile_info:
                            pars_to_get = derived_parameters_withcompileinfo['common'] + \
                                          derived_parameters_withcompileinfo[name_algo]
                        new_df = parameter_sets.get_features(pars_to_get)
                        data_algo.merge(new_df)

                        # Write derived parameters
                        derived_parameters_file_name = os.path.join(
                            kernel_folder, 'training_data_' + mnk_string.format(m, n, k) + '_' + name_algo + '.csv')
                        if use_compile_info:
                            derived_parameters_file_name = derived_parameters_file_name.replace('.csv', '_withcompileinfo.csv')
                        data_algo[pars_to_get].to_csv(derived_parameters_file_name)
                        print('Wrote', derived_parameters_file_name)

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
def print_dic(dic):
    for k, v in dic.items():
        if isinstance(v, str):
            print('{:<40}: {:>8}'.format(k, v))
        else:
            print('{:<40}: {:>8,}'.format(k, v))


def format_pars(df):
    df.fillna(value=0, inplace=True)
    df = df.rename(columns={'threads': 'threads_per_blk', 'nregs': 'regs_per_thread'})
    return df


# ===============================================================================
# Read data
tiny_mangle = re.compile(
    '_Z\d+cusmm_dnt_tiny' +
    '[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]' +
    '+PKdS3_Pd'
)
smallmed_mangle = re.compile(
    '_Z\d+cusmm_dnt_(small|medium)' +
    '[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]' +
    '+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]' +
    '+PKdS3_Pd'
)
largeDB_mangle = re.compile(
    '_Z\d+cusmm_dnt_largeDB(1|2)' +
    '[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]' +
    '+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]+(\d+)[a-zA-Z]' +
    '+PKdS3_Pd'
)


def quick_demangle(mangled_name):
    """
    Basic function name demangling, examples:
        _Z14cusmm_dnt_tinyILi4ELi4ELi8ELi32ELi15ELi30EEvPKiiPKdS3_Pd
            -> void cusmm_dnt_tiny<4, 4, 8, 32, 15, 30>(int const*, int, double const*, double const*, double*)
        _Z15cusmm_dnt_smallILi16ELi16ELi16ELi2ELi2ELi64ELi29ELi8EEvPKiiPKdS3_Pd
            -> void cusmm_dnt_small<16, 16, 16, 2, 2, 64, 29, 8>(int const*, int, double const*, double const*, double*)
        _Z16cusmm_dnt_mediumILi25ELi32ELi5ELi5ELi1ELi192ELi16ELi7EEvPKiiPKdS3_Pd
            -> void cusmm_dnt_medium<25, 32, 5, 5, 1, 192, 16, 7>(int const*, int, double const*, double const*, double*)
    :return: dictionary of algorithm and parameters
    """
    if 'tiny' in mangled_name:
        match = tiny_mangle.match(mangled_name)
        assert match is not None, "Cannot match name:\n" + mangled_name
        pars = {'algorithm': 'tiny',
                'm': int(match.group(1)), 'n': int(match.group(2)), 'k': int(match.group(3)),
                'tile_m': None, 'tile_n': None, 'w': None, 'v': None,
                'threads': int(match.group(4)), 'grouping': int(match.group(5)), 'minblocks': int(match.group(6))}
    elif 'small' in mangled_name or 'medium' in mangled_name:
        match = smallmed_mangle.match(mangled_name)
        assert match is not None, "Cannot match name:\n" + mangled_name
        pars = {'algorithm': match.group(1),
                'm': int(match.group(2)), 'n': int(match.group(3)), 'k': int(match.group(4)),
                'tile_m': int(match.group(5)), 'tile_n': int(match.group(6)),
                'w': None, 'v': None,
                'threads': int(match.group(7)), 'grouping': int(match.group(8)), 'minblocks': int(match.group(9))}
    elif 'largeDB' in mangled_name:
        match = largeDB_mangle.match(mangled_name)
        assert match is not None, "Cannot match name:\n" + mangled_name
        pars = {'algorithm': 'largeDB' + match.group(1),
                'm': int(match.group(2)), 'n': int(match.group(3)), 'k': int(match.group(4)),
                'tile_m': int(match.group(5)), 'tile_n': int(match.group(6)),
                'w': int(match.group(7)), 'v': int(match.group(8)),
                'threads': int(match.group(9)), 'grouping': int(match.group(10)), 'minblocks': int(match.group(11))}
    else:
        assert False, "Cannot find base name in:\n" + mangled_name

    return pars


ptxas_intro = re.compile('ptxas info\s+: Function properties for (_Z\d+.*)$')
ptxas_values = re.compile('ptxas info\s+: Used (\d+) registers, (\d+) bytes smem, (\d+) bytes cmem\[0\]')


def read_slurm_file(log_folder):
    """
    Given a folder of kernel autotuning, read in compilation information
    :param log_folder: folder of kernel autotuning
    :return: dictionary containing above information
    """
    # Find slurm file
    slurm_files = [f for f in os.listdir(log_folder) if f[:5] == 'slurm']
    assert len(slurm_files) == 1
    slurm_file = slurm_files[0]
    print('Process slurm file:', slurm_file)

    # Collect data
    with open(os.path.join(log_folder, slurm_file), 'r') as f:
        slurm_file_content = f.read().splitlines()

    # Sometimes, the output of the compilation of different functions gets garbled. E.g.:
    #     ptxas info    : Compiling entry function '_Z15cusmm_dnt_smallILi4ELi4ELi4ELi4ELi3ELi32ELi21ELi4EEvPKiiPKdS3_Pd' for 'sm_60'
    #     ptxas info    : Function properties for _Z15cusmm_dnt_smallILi4ELi4ELi4ELi4ELi3ELi32ELi21ELi4EEvPKiiPKdS3_Pd
    #         0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    #     ptxas info    : Function properties for _Z15cusmm_dnt_smallILi4ELi4ELi4ELi2ELi2ELi32ELi3ELi18EEvPKiiPKdS3_Pd
    #         0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    #     ptxas info    : Used 80 registers, 576 bytes smem, 360 bytes cmem[0]
    #     ptxas info    : Used 51 registers, 296 bytes smem, 360 bytes cmem[0]
    # We assume that the output related to the compilation of a function can get garbled with the ones from other
    # functions, but that order in the output of a given function is preserved.
    kernels = list()
    ptxas_info = list()
    for i, l in enumerate(slurm_file_content):

        if 'ptxas info    : Function properties' in l:
            if ptxas_intro.match(l):
                kernel_dic = quick_demangle(ptxas_intro.match(l).group(1))
                kernels.append(tuple(kernel_dic.values()))
            else:
                assert False, "No match at line " + str(i) + ":\n" + l
        if 'Used' in l:
            if ptxas_values.search(l):
                m = ptxas_values.search(l)
                ptxas_info.append({'perf (Gflop/s)': 0,
                                   'nregs': int(m.group(1)),
                                   'nbytes_smem': int(m.group(2)),
                                   'nbytes_cmem': int(m.group(3))})
            else:
                assert False, "No match at line " + str(i) + ":\n" + l

    assert len(kernels) == len(ptxas_info)
    print("Found", len(kernels), "data items in slurm file")
    data = dict(zip(kernels, ptxas_info))
    return data


log_file_ = re.compile('tune_(\d+)x(\d+)x(\d+)_exe')
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
    return data


# ===============================================================================
main()
