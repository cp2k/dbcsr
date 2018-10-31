#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import sys
import pickle
import json
import numpy as np
import pandas as pd
from itertools import product
from optparse import OptionParser
from joblib import Parallel, delayed
from kernels.cusmm_dnt_helper import arch_number, kernel_algorithm, params_dict_to_kernel
from parameter_space_utils import PredictiveParameters
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


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


def dump_file(dump_folder, m, n, k, algo):
    return os.path.join(dump_folder, "optker_{}x{}x{}_{}".format(m, n, k, algo))


def find_optimal_kernel(mnk, algo, tree, tree_features, top_k, gpu_properties, autotuning_properties, dump_folder):
    """
    Find the optimal kernel parameters for a given (m, n, k) and a given algorithm
    :return: optimal_kernels: dictionary, keys: (m, n, k), values: Kernel object describing best parameters
             number of elements in dictionary = top_k
    """

    # Find optimal parameter set for this (m, n, k)
    m, n, k = mnk

    # Has this configuration already been computed?
    file_to_dump = dump_file(dump_folder, m, n, k, algo)
    if os.path.exists(file_to_dump):
        if os.stat(file_to_dump).st_size == 0:
            optimal_kernels = dict()
        else:
            from joblib import load
            optimal_kernels = load(file_to_dump)

    else:

        # Get parameter space for this (m, n, k) and this algorithm
        parameter_space_ = kernel_algorithm[algo].promising_parameters(m, n, k, gpu_properties, autotuning_properties)
        parameter_space = pd.DataFrame(parameter_space_)
        del parameter_space_
        parameter_space['algorithm'] = [algo] * len(parameter_space.index)  # Add "algorithm" column
        if len(parameter_space.index) == 0:
            optimal_kernels = dict()

        else:

            # Predict performances
            parameter_sets = PredictiveParameters(parameter_space, gpu_properties, autotuning_properties)
            predictors = np.array(parameter_sets.get_features(tree_features))
            performances = np.sqrt(tree.predict(predictors))
            del predictors
            parameter_performances = parameter_sets.params
            del parameter_sets
            parameter_performances['perf'] = performances
            del performances

            # Store kernels of this algorithm that are candidates for the top-k
            candidate_kernels = sorted(parameter_performances.to_dict('records'),
                                       key=lambda x: x['perf'], reverse=True)[:top_k]
            del parameter_performances

            # Aggregate results from the different algorithms: sort and pick top_k
            optimal_kernels = dict()
            optimal_kernel_dicts = sorted(candidate_kernels, key=lambda x: x['perf'], reverse=True)[:top_k]
            del candidate_kernels
            for d in optimal_kernel_dicts:
                optimal_kernels[(m, n, k)] = params_dict_to_kernel(**d, source='predicted')

    return optimal_kernels


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
    parser.add_option("-j", "--njobs", type=int,
                      default=-4, help="Number of joblib jobs. Default: %default")
    parser.add_option("-c", "--chunk_size", type=int,
                      default=20000, help="Chunk size for dispatching joblib jobs. Default: %default")
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
    mnks = combinations(list(range(4, 46)))
    top_k = 1
    optimal_kernels = dict()
    mnks_to_predict = list()
    dump_folder = 'optimal_kernels_dump'

    for m, n, k in mnks:
        if (m, n, k) in autotuned_kernels.keys():
            optimal_kernels[(m, n, k)] = autotuned_kernels[(m, n, k)]
        else:
            mnks_to_predict.append((m, n, k))

    # optimal_kernels_list is a list of dictionaries
    # - keys: (m, n, k),
    # - values: Kernel object describing best parameters
    # - number of elements in each dictionary = top_k
    # each element of the list corresponds to the search of optimal kernels for a given mnk and a given algorithm
    n_jobs = options.njobs
    if n_jobs == 1:

        # Ignore joblib and run serially:
        optimal_kernels_list = list()
        for mnk, algo in product(mnks_to_predict, kernel_algorithm.keys()):
            gc.collect()
            print("Find optimal kernels for mnk=", mnk, ", algo=", algo)
            optimal_kernels_list.append(
                find_optimal_kernel(
                    mnk, algo, tree[algo]['tree'], tree[algo]['features'],
                    top_k, gpu_properties, autotuning_properties, dump_folder
                )
            )
    else:

        # Chunk up tasks
        optimal_kernels_list = list()
        chunk_size = options.chunk_size
        mnk_by_algo = list(product(mnks_to_predict, kernel_algorithm.keys()))
        num_mnks_by_algo = len(list(mnk_by_algo))

        for i in range(0, num_mnks_by_algo+1, chunk_size):
            start_chunk = i
            end_chunk = min(start_chunk + chunk_size, num_mnks_by_algo+1)
            print("Completed", i, "tasks out of", num_mnks_by_algo)

            # Run prediction tasks in parallel with joblib
            optimal_kernels_list_ = Parallel(n_jobs=n_jobs, verbose=3)(
                delayed(
                    find_optimal_kernel, check_pickle=True)(
                        mnk, algo, tree[algo]['tree'], tree[algo]['features'],
                        top_k, gpu_properties, autotuning_properties, dump_folder
                    )
                for mnk, algo in mnk_by_algo[start_chunk:end_chunk])

            optimal_kernels_list += optimal_kernels_list_


    print("Finished gathering candidates for optimal parameter space")

    optimal_kernels_mnk_algo = dict()
    for optimal_kernel_mnk in optimal_kernels_list:
        for mnk, kernels_mnk in optimal_kernel_mnk.items():
            m, n, k = mnk
            if (m, n, k) in optimal_kernels_mnk_algo.keys():
                optimal_kernels_mnk_algo[(m, n, k)].append(kernels_mnk)
            else:
                optimal_kernels_mnk_algo[(m, n, k)] = [kernels_mnk]

    for mnk, candidate_kernels in optimal_kernels_mnk_algo.items():
        m, n, k = mnk
        optimal_kernel_mnk = sorted(candidate_kernels, key=lambda x: x.perf, reverse=True)[:top_k]
        optimal_kernels[(m, n, k)] = optimal_kernel_mnk[0]

    ####################################################################################################################
    # Write to file
    with open(options.params, 'w') as f:
        s = json.dumps([optimal_kernels[kernel].as_dict_for_parameters_json for kernel in sorted(optimal_kernels.keys())])
        s = s.replace('}, ', '},\n')
        s = s.replace('[', '[\n')
        s = s.replace(']', '\n]')
        f.write(s)
    print("Wrote new predicted parameters to file", options.params)


#===============================================================================
main(argv=sys.argv[1:])

#EOF
