#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from itertools import product
from optparse import OptionParser
from kernels.cusmm_dnt_helper import arch_number, kernel_algorithm, params_dict_to_kernel
from parameter_space_utils import PredictiveParameters
from sklearn.tree import DecisionTreeRegressor


########################################################################################################################
# Helpers
########################################################################################################################
def combinations(sizes):
    return list(product(sizes, sizes, sizes))


def safe_pickle_load(file_path):
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f.read(max_bytes)
    return pickle.loads(bytes_in)


########################################################################################################################
# Main
########################################################################################################################
def main(argv):
    """
    Update parameter file with new optimal parameter predictions given newly trained decision trees
    """
    del argv  # unused

    parser = OptionParser()
    parser.add_option("-p", "--params", metavar="filename.json",
                      default="parameters_P100.json", help="Default: %default")
    parser.add_option("-t", "--trees", metavar="folder/",
                      default="predictive_model/", help="Default: %default")
    options, args = parser.parse_args(sys.argv)

    ####################################################################################################################
    # Load GPU and autotuning properties as well as pre-autotuned parameters
    assert options.params in arch_number.keys(), "Cannot find compute version for file " + str(options.params)
    arch = arch_number[options.params]
    with open('kernels/gpu_properties.json') as f:
        gpu_properties = json.load(f)["sm_" + str(arch)]
    with open('kernels/autotuning_properties.json') as f:
        autotuning_properties = json.load(f)
    with open(options.params) as f:
        all_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]
    print("Libcusmm: Found %d existing parameter sets." % len(all_kernels))
    autotuned_mnks = [(k.m, k.n, k.k) for k in all_kernels if k.autotuned]
    autotuned_kernels_ = [k for k in all_kernels if k.autotuned]
    autotuned_kernels = dict(zip(autotuned_mnks, autotuned_kernels_))

    ####################################################################################################################
    # Load Predictive trees and feature list
    tree = dict()
    for algo, v in kernel_algorithm.items():
        tree[algo] = dict()
        tree[algo]['file'] = os.path.join(str(options.trees), algo + '_')
        tree[algo]['tree'] = safe_pickle_load(tree[algo]['file'] + 'predictive_tree.p')
        features = safe_pickle_load(tree[algo]['file'] + 'feature_names.p')
        features.remove('mnk')
        tree[algo]['features'] = features

    ####################################################################################################################
    # Evaluation
    optimal_kernels = list()
    top_k = 1
    for i, mnk in enumerate(combinations(list(range(4, 46)))):

        m, n, k = mnk
        print(m, "x", n, "x", k, "(", i, "/", len(combinations(list(range(4, 46)))), ")")

        if (m, n, k) in autotuned_kernels.keys():
            optimal_kernels.append(autotuned_kernels[(m, n, k)])

        else:  # find optimal parameter set for this (m, n, k)

            candidate_kernels = list()
            for algo, kernel_algo in kernel_algorithm.items():

                # Get parameter space for this (m, n, k) and this algorithm
                parameter_space = kernel_algo.promising_parameters(m, n, k, gpu_properties, autotuning_properties)
                parameter_space = pd.DataFrame(parameter_space)
                parameter_space['algorithm'] = [algo] * len(parameter_space.index)
                if len(parameter_space.index) == 0:
                    break  # if this algorithm is not compatible with this (m, n, k), move on

                # Predict performances
                parameter_sets = PredictiveParameters(parameter_space, gpu_properties, autotuning_properties)
                predictors = np.array(parameter_sets.get_features(tree[algo]['features']))
                performances = np.sqrt(tree[algo]['tree'].predict(predictors))
                parameter_performances = parameter_sets.params
                parameter_performances['perf'] = performances

                # Store kernels of this algorithm that are candidates for the top-k
                candidate_kernels = sorted(parameter_performances.to_dict('records'),
                                           key=lambda x: x['perf'], reverse=True)[:top_k]

            # Aggregate results from the different algorithms: sort and pick top_k
            optimal_kernel_dicts = sorted(candidate_kernels, key=lambda x: x['perf'], reverse=True)[:top_k]
            for d in optimal_kernel_dicts:
                optimal_kernels.append(params_dict_to_kernel(**d, source='predicted'))

    ####################################################################################################################
    # Write to file
    with open(options.params, 'w') as f:
        s = json.dumps([kernel.as_dict for kernel in optimal_kernels])
        s = s.replace('}, ', '},\n')
        s = s.replace('[', '[\n')
        s = s.replace(']', '\n]')
        f.write(s)
    print("Wrote new predicted parameters to file", options.params)


#===============================================================================
main(argv=sys.argv[1:])

#EOF
