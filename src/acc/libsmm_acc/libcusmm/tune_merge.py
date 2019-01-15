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


import json
from optparse import OptionParser
from kernels.cusmm_predict import params_dict_to_kernel


def main():
    usage = (
        "Write a new kernel parameter file as an unique merge of an old parameter file and a new one called "
        + "parameters.json as created by collect.py. If a kernel (m, n, k) is listed in both the old parameter"
        + "file and the new parameter file, retain its parameters as defined in the new parameter file."
    )
    parser = OptionParser(usage)
    parser.add_option(
        "-p", "--params", metavar="filename.json", default="parameters_P100.json", help="Default: %default"
    )

    (options, args) = parser.parse_args()
    assert len(args) == 0
    param_fn = options.params

    # Read new kernel parameters
    with open("parameters.json") as f:
        new_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]

    # Read old kernel parameters
    with open(param_fn) as f:
        old_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]

    # Merge two parameter lists
    kernels_dict = dict(zip([(k.m, k.n, k.k) for k in old_kernels], old_kernels))
    new_kernels_dict = dict(zip([(k.m, k.n, k.k) for k in new_kernels], new_kernels))
    kernels_dict.update(new_kernels_dict)

    # Write kernel parameters to new file
    new_file = "parameters.new.json"
    with open(new_file, "w") as f:
        s = json.dumps([kernels_dict[kernel].as_dict for kernel in sorted(kernels_dict.keys())])
        s = s.replace("}, ", "},\n")
        s = s.replace("[", "[\n")
        s = s.replace("]", "\n]")
        f.write(s)

    print("Wrote", new_file)


# ===============================================================================
main()
