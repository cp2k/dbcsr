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

import json
import argparse
from kernels.smm_acc_predict import params_dict_to_kernel


def main(param_fn):

    # Read new kernel parameters
    param_new = "parameters.json"
    with open(param_new) as f:
        new_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]

    # Read old kernel parameters
    with open(param_fn) as f:
        old_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]

    # Merge two parameter lists
    print("Merging", param_new, "with", param_fn)
    kernels_dict = dict(zip([(k.m, k.n, k.k) for k in old_kernels], old_kernels))
    new_kernels_dict = dict(zip([(k.m, k.n, k.k) for k in new_kernels], new_kernels))
    kernels_dict.update(new_kernels_dict)

    # Write kernel parameters to new file
    new_file = "parameters.new.json"
    with open(new_file, "w") as f:
        s = json.dumps(
            [
                kernels_dict[kernel].as_dict_for_parameters_json
                for kernel in sorted(kernels_dict.keys())
            ]
        )
        s = s.replace("}, ", "},\n")
        s = s.replace("[", "[\n")
        s = s.replace("]", "\n]")
        f.write(s)

    print("Wrote", new_file)


# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Write a new kernel parameter file (parameters.new.json) as a unique merge of an
        already-existing parameter file (specified by `-p parameters_GPU.json`) and a new
        one (parameters.json) created by tune_collect.py. If a kernel (m, n, k) is listed
        in both the original parameter file and the new parameter file, retain its parameters
        as defined in the new parameter file.

        This script is part of the workflow for autotuning optimal libsmm_acc parameters.
        For more details, see README.md#autotuning-procedure.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--params",
        metavar="parameters_GPU.json",
        type=str,
        default="../parameters/parameters_P100.json",
        help="parameter file in which to merge the newly obtained autotuned parameters",
    )

    args = parser.parse_args()
    main(args.params)
