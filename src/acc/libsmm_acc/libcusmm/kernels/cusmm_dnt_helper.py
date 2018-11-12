####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

#===============================================================================
# Correspondance between CUDA compute versions and parameter_file
arch_number = {
    "parameters_K20X.json": 35,
    "parameters_K40.json": 35,
    "parameters_K80.json": 37,
    "parameters_P100.json": 60
}


#===============================================================================
from kernels.cusmm_dnt_largeDB1 import Kernel_dnt_largeDB1
from kernels.cusmm_dnt_largeDB2 import Kernel_dnt_largeDB2
from kernels.cusmm_dnt_medium import Kernel_dnt_medium
from kernels.cusmm_dnt_small import Kernel_dnt_small
from kernels.cusmm_dnt_tiny import Kernel_dnt_tiny

kernel_algorithm = {
    'tiny': Kernel_dnt_tiny,
    'small': Kernel_dnt_small,
    'medium': Kernel_dnt_medium,
    'largeDB1': Kernel_dnt_largeDB1,
    'largeDB2': Kernel_dnt_largeDB2
}


def params_dict_to_kernel(**params):

    algo = params.pop('algorithm')
    kernel_init_params = ['m', 'n', 'k', 'threads', 'grouping', 'minblocks', 'perf', 'source']
    if algo in ['small', 'medium', 'largeDB1', 'largeDB2']:
        kernel_init_params.append("tile_m")
        kernel_init_params.append("tile_n")
        if algo in ['largeDB1', 'largeDB2']:
            kernel_init_params.append("v")
            kernel_init_params.append("w")

    kernel_init_params_dict = dict()
    if 'threads_per_blk' in params.keys():
        kernel_init_params_dict['threads'] = params['threads_per_blk']
        kernel_init_params.remove('threads')

    for k in kernel_init_params:
        kernel_init_params_dict[k] = params[k]

    return kernel_algorithm[algo](**kernel_init_params_dict)


def descr_to_kernel(kernel_descr, source='autotuned'):
    import re
    from ast import literal_eval
    re_kernel_descr = re.compile(r"Kernel_dnt_(\w+)(\(.*\)) , # (\d+\.\d+) GFlop/s")
    match = re_kernel_descr.search(kernel_descr).groups()
    algo = match[0]
    m = match[1].replace('=', '\':')
    m = m.replace(', ', ', \'')
    m = m.replace('(', '{\'')
    m = m.replace(')', '}')
    params = dict(literal_eval(m))
    params['perf'] = float(match[2])
    params['source'] = source
    return kernel_algorithm[algo](**params)


def compatible_mnk(algo, m, n, k):
    max_sizes = max(m * k, n * k, m * n)
    compatible = True
    if algo == 'tiny':
        if max_sizes > 64:
            compatible = False
    elif algo == 'small':
        if max_sizes > 128:
            compatible = False
    elif algo in ['largeDB1', 'largeDB2']:
        if max_sizes < 250:
            compatible = False

    return compatible
