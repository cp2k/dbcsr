####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

import re

# ===============================================================================
# Dictionary of available kernel algorithms
# keys: kernel name
# values: kernel implementation class
from kernels.smm_acc_dnt_largeDB1 import Kernel_dnt_largeDB1
from kernels.smm_acc_dnt_largeDB2 import Kernel_dnt_largeDB2
from kernels.smm_acc_dnt_medium import Kernel_dnt_medium
from kernels.smm_acc_dnt_small import Kernel_dnt_small
from kernels.smm_acc_dnt_tiny import Kernel_dnt_tiny

kernel_algorithm = {
    "tiny": Kernel_dnt_tiny,
    "small": Kernel_dnt_small,
    "medium": Kernel_dnt_medium,
    "largeDB1": Kernel_dnt_largeDB1,
    "largeDB2": Kernel_dnt_largeDB2,
}


# ===============================================================================
# Dictionary of parameter types
# keys: parameter name
# values: type of the parameter
parameter_types = {
    "m": int,
    "n": int,
    "k": int,
    "algorithm": str,
    "threads": int,
    "grouping": int,
    "minblocks": int,
    "tile_m": int,
    "tile_n": int,
    "w": int,
    "v": int,
    "perf (Gflop/s)": float,
}


# ===============================================================================
# Dictionary of available GPU architectures.
# keys: parameter_file
# values: CUDA compute versions / AMD processor versions
gpu_architectures = {
    "parameters_K20X.json": "sm_35",
    "parameters_K40.json": "sm_35",
    "parameters_K80.json": "sm_37",
    "parameters_P100.json": "sm_60",
    "parameters_V100.json": "sm_70",
    "parameters_Vega10.json": "gfx900",
    "parameters_Mi50.json": "gfx906",
}


# ===============================================================================
def compatible_mnk(algo, m, n, k):
    """Determine whether a given algorithm is compatible with given m, n, k values"""

    max_sizes = max(m * k, n * k, m * n)
    compatible = True
    if algo == "tiny":
        if max_sizes > 64:
            compatible = False
    elif algo == "small":
        if max_sizes > 128:
            compatible = False
    elif algo in ["largeDB1", "largeDB2"]:
        if max_sizes < 250:
            compatible = False
    else:
        if algo != "medium":
            assert False, "Cannot identify algorithm:" + str(algo)

    return compatible


# ===============================================================================
def params_dict_to_kernel(**params):
    """Given a dictionary of parameters, return the corresponding Kernel class instance"""

    # Get the 'algorithm' field
    algo = params.pop("algorithm")

    # Get the list of fields needed to initialize a Kernel instance of this given algorithm
    kernel_init_params = kernel_algorithm[algo].launch_parameters + ["perf", "source"]

    # Fill in dictionary fields
    kernel_init_params_dict = dict()
    for k in kernel_init_params:
        if (
            k == "perf"
            and "perf" not in params.keys()
            and params["source"] == "predicted"
        ):
            # the performance of predicted parameter sets is not given
            kernel_init_params_dict["perf"] = None
        else:
            kernel_init_params_dict[k] = params[k]

    return kernel_algorithm[algo](**kernel_init_params_dict)


def descr_to_kernel(kernel_descr, source="autotuned"):
    """Given a kernel description from the autotuning output, return the corresponding Kernel class instance"""

    from ast import literal_eval

    re_kernel_descr = re.compile(
        r"Kernel_dnt_(\w+)(\(.*\)) , # (\d+(?:\.\d+)?) GFlop/s"
    )
    kernel_descr_matched = re_kernel_descr.search(kernel_descr)
    assert kernel_descr_matched is not None, (
        'Could not match kernel description in "' + kernel_descr + '"'
    )
    match = kernel_descr_matched.groups()
    algo = match[0]
    m = match[1].replace("=", "':")
    m = m.replace(", ", ", '")
    m = m.replace("(", "{'")
    m = m.replace(")", "}")
    params = dict(literal_eval(m))
    params["perf"] = float(match[2])
    params["source"] = source
    return kernel_algorithm[algo](**params)


def to_string(*iterable):
    """
    Given a (list of) m,n,k-triplet(s), return the corresponding (list of) string(s) "mxnxk"
    """
    mnk_string = "{}x{}x{}"
    if len(iterable) == 3 and isinstance(iterable[0], int):
        m, n, k = iterable
        iterable_to_string = mnk_string.format(m, n, k)
    else:
        iterable_to_string = [mnk_string.format(m, n, k) for m, n, k in iterable]
    if len(iterable_to_string) == 1:
        iterable_to_string = iterable_to_string[0]
    return iterable_to_string


mnk_pattern = re.compile(r"(\d+)x(\d+)x(\d+)")


def to_tuple(*iterable):
    """
    Given a (list of) string(s) "mxnxk", return the corresponding (list of) m,n,k-triplet(s)
    """
    tuple_mnks = list()
    for mnk in iterable:
        m, n, k = mnk_pattern.match(mnk).groups()
        tuple_mnks.append((int(m), int(n), int(k)))
    if len(tuple_mnks) == 1:
        tuple_mnks = tuple_mnks[0]
    return tuple_mnks
