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
import gc
import sys
import json
import numpy as np
import pandas as pd
from itertools import product
from optparse import OptionParser
from joblib import Parallel, delayed
from predict_helpers import *
from kernels.cusmm_dnt_helper import arch_number, kernel_algorithm, params_dict_to_kernel, PredictiveParameters
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# ===============================================================================
# Helpers
def combinations(sizes):
    return list(product(sizes, sizes, sizes))


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


# ===============================================================================
# Main
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
                      default=-1, help="Number of joblib jobs. Default: %default")
    parser.add_option("-s", "--start_slice", type=int,
                      default=0, help="Start mnk slice. Default: %default")
    parser.add_option("-a", "--algo",
                      default='small', help="Investigate only a given algorithm. Default: %default")
    parser.add_option("-o", "--overwrite_autotuned",
                      action="store_true",
                      default=False, help="Overwrite autotuned kernels with predicted optimal sets. Default: %default")
    parser.add_option("-e", "--slice_size", type=int,
                      default=None, help="Size mnk slice. Default: %default")
    parser.add_option("-c", "--chunk_size", type=int,
                      default=20000, help="Chunk size for dispatching joblib jobs. Default: %default")
    options, args = parser.parse_args(sys.argv)

    # ===============================================================================
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

    # ===============================================================================
    # Load Predictive trees and feature list
    tree = dict()
    if options.algo is not None:
        print("Investigating only", options.algo)
        kernel_to_investigate = {options.algo: kernel_algorithm[options.algo]}
    else:
        kernel_to_investigate = kernel_algorithm

    for algo, v in kernel_to_investigate.items():
        tree[algo] = dict()
        tree[algo]['file'] = os.path.join(str(options.trees), algo + '_')
        if os.path.exists(tree[algo]['file'] + 'predictive_tree.p'):
            tree[algo]['tree'] = safe_pickle_load(tree[algo]['file'] + 'predictive_tree.p')
            features = safe_pickle_load(tree[algo]['file'] + 'feature_names.p')
        elif os.path.exists(tree[algo]['file'] + 'feature_tree.p'):
            features, tree[algo]['tree'], _ = safe_pickle_load(tree[algo]['file'] + 'feature_tree.p')
            features = features.tolist()
        else:
            assert False, "Cannot find model files in folder:" + options.trees
        if 'mnk' in features:
            features.remove('mnk')
        tree[algo]['features'] = features

    # ===============================================================================
    # Evaluation
    dump_folder = 'optimal_kernels_dump'
    mnks = combinations(list(range(4, 46)))
    top_k = 1
    optimal_kernels = dict()

    if options.overwrite_autotuned:
        mnks_to_predict = autotuned_mnks
        #mnks_to_predict = mnks
    else:
        mnks_to_predict = list()
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
    optimal_kernels_list = list()
    mnk_by_algo = list(product(mnks_to_predict, kernel_to_investigate.keys()))
    if options.slice_size is not None:
        start_slice = options.start_slice
        end_slice = min(start_slice + options.slice_size, len(list(mnk_by_algo)))
        mnk_by_algo = mnk_by_algo[start_slice:end_slice]
    num_mnks_by_algo = len(mnk_by_algo)
    n_jobs = options.njobs
    if n_jobs == 1:

        # Ignore joblib and run serially:
        for mnk, algo in mnk_by_algo:
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
        chunk_size = options.chunk_size
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

    # Group optimal kernel candidates by (m,n,k) in a dictionary
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

    # ===============================================================================
    # Write to file
    with open(options.params, 'w') as f:
        s = json.dumps([optimal_kernels[kernel].as_dict_for_parameters_json for kernel in sorted(optimal_kernels.keys())])
        s = s.replace('}, ', '},\n')
        s = s.replace('[', '[\n')
        s = s.replace(']', '\n]')
        f.write(s)
    print("Wrote new predicted parameters to file", options.params)


# ===============================================================================
main(argv=sys.argv[1:])

#EOF
