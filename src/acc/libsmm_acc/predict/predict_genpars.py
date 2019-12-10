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

import gc
import os
import sys
import json
import pandas as pd
from itertools import product
import argparse
from joblib import Parallel, delayed
from predict_helpers import safe_pickle_load
from kernels.smm_acc_predict import (
    gpu_architectures,
    kernel_algorithm,
    to_string,
    to_tuple,
    params_dict_to_kernel,
    PredictiveParameters,
)


# ===============================================================================
def main(params, njobs, baseline, paths_to_models, chunk_size):
    """
    This script is part of the workflow for predictive modelling of optimal libsmm_acc parameters.
    For more details, see predict.md

    Update parameter file with new optimal parameter predictions given newly trained decision trees
    """
    # ===============================================================================
    # Load GPU and autotuning properties
    assert (
        os.path.basename(params) in gpu_architectures.keys()
    ), "Cannot find compute version for file " + str(params)
    arch_code = gpu_architectures[os.path.basename(params)]
    with open("../kernels/gpu_properties.json") as f:
        gpu_properties = json.load(f)[arch_code]
    with open("../kernels/autotuning_properties.json") as f:
        autotuning_properties = json.load(f)

    # Load autotuned kernel parameters
    with open(params) as f:
        all_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]
    print("libsmm_acc: Found %d existing parameter sets." % len(all_kernels))
    autotuned_mnks = [(k.m, k.n, k.k) for k in all_kernels if k.autotuned]
    autotuned_kernels_ = [k for k in all_kernels if k.autotuned]
    autotuned_kernels = dict(zip(autotuned_mnks, autotuned_kernels_))

    # ===============================================================================
    # Construct the list of (m,n,k)-triplets for which parameter sets should be made available to libcusmm
    mnks = combinations(list(range(4, 46)))
    mnks = set.union(set(mnks), set(autotuned_kernels.keys()))

    # ===============================================================================
    # Compute parameter sets
    mnks_to_predict = list()
    kernels_to_print = dict()
    for m, n, k in mnks:
        if (m, n, k) in autotuned_kernels.keys():
            kernels_to_print[(m, n, k)] = autotuned_kernels[(m, n, k)]
        else:
            mnks_to_predict.append((m, n, k))

    if baseline:
        kernels = get_baseline_kernels(
            mnks_to_predict, gpu_properties, autotuning_properties
        )
    else:
        kernels = get_optimal_kernels(
            mnks_to_predict,
            njobs,
            chunk_size,
            paths_to_models,
            gpu_properties,
            autotuning_properties,
            1,
        )

    kernels_to_print.update(kernels)

    # ===============================================================================
    # Write to file
    with open(params, "w") as f:
        s = json.dumps(
            [
                kernels_to_print[kernel].as_dict_for_parameters_json
                for kernel in sorted(kernels_to_print.keys())
            ]
        )
        s = s.replace("}, ", "},\n")
        s = s.replace("[", "[\n")
        s = s.replace("]", "\n]")
        f.write(s)
    print("Wrote new predicted parameters to file", params)


# ===============================================================================
# Helpers
def combinations(sizes):
    return list(product(sizes, sizes, sizes))


def remove_empty_entries(ld):
    """
    Given a list of dictionaries "ld", remove its list elements that are empty dicts
    """
    return [d for d in ld if d]  # empty dictionaries evaluate to False


def find_optimal_kernel(
    mnk, algo, tree, tree_features, gpu_properties, autotuning_properties
):
    """
    Find the optimal kernel parameter set for a given (m, n, k) and a given algorithm
    :return: optimal_kernels: dictionary, keys: (m, n, k), values: Kernel object describing best parameters
    """

    # Get parameter space for this (m, n, k) and this algorithm
    m, n, k = mnk
    parameter_space_ = kernel_algorithm[algo].promising_parameters(
        m, n, k, gpu_properties, autotuning_properties
    )
    parameter_space = pd.DataFrame(parameter_space_)
    del parameter_space_
    parameter_space["algorithm"] = [algo] * len(
        parameter_space.index
    )  # Add "algorithm" column
    if len(parameter_space.index) == 0:
        optimal_kernels = dict()

    else:

        # Get predictor features from raw parameters
        parameter_sets = PredictiveParameters(
            parameter_space, gpu_properties, autotuning_properties, None
        )
        predictors = parameter_sets.get_features(tree_features)
        if algo == "medium":
            predictors = predictors.rename(
                columns=dict(
                    zip(
                        predictors.columns,
                        [
                            "f{}".format(i)
                            for i in range(0, len(predictors.columns) + 1)
                        ],
                    )
                )
            )

        # Predict performances
        performances_scaled = tree.predict(predictors)
        del predictors
        parameter_performances = parameter_sets.params
        del parameter_sets
        parameter_performances["perf"] = performances_scaled
        del performances_scaled

        # Pick optimal kernel
        optimal_kernel = max(
            parameter_performances.to_dict("records"), key=lambda x: x["perf"]
        )
        del parameter_performances
        optimal_kernels = dict()
        optimal_kernels[(m, n, k)] = params_dict_to_kernel(
            **optimal_kernel, source="predicted"
        )

    return optimal_kernels


def get_optimal_kernels(
    mnks_to_predict,
    njobs,
    chunk_size,
    paths_to_models,
    gpu_properties,
    autotuning_properties,
    top_k,
):
    # optimal_kernels_list is a list of dictionaries
    # - keys: (m, n, k),
    # - values: Kernel object describing best parameters
    # - number of elements in each dictionary = top_k
    # each element of the list corresponds to the search of optimal kernels for a given mnk and a given algorithm

    print("Getting optimal kernels")

    # ===============================================================================
    # Load predictive trees and feature list
    tree = dict()
    kernel_to_investigate = dict()
    for algo in kernel_algorithm.keys():
        path_to_model = paths_to_models[algo]
        if path_to_model is not None:
            print(
                "Algorithm: {:<8}, loading model from: {}".format(algo, path_to_model)
            )
            tree[algo] = dict()
            tree[algo]["file"] = path_to_model
            features, tree[algo]["tree"] = safe_pickle_load(tree[algo]["file"])
            tree[algo]["features"] = features.tolist()
            kernel_to_investigate[algo] = kernel_algorithm[algo]
        else:
            print("Algorithm: {:<8}, no model found.".format(algo))

    if len(kernel_to_investigate) == 0:
        print("No model found. Specify path to predictive models using ")
        sys.exit(1)

    # ===============================================================================
    # Get mnks_by_algo to compute:
    mnks_by_algo = list(product(mnks_to_predict, kernel_to_investigate.keys()))
    num_mnks_by_algo = len(mnks_by_algo)
    optimal_kernels_list = list()
    ckpt_folder_name = "predict_genpars_ckpt"

    if not os.path.exists(ckpt_folder_name):
        os.mkdir(ckpt_folder_name)
    print("Caching intermediate results to:", ckpt_folder_name)

    for i in range(0, num_mnks_by_algo + 1, chunk_size):

        # Chunk up tasks
        start_chunk = i
        end_chunk = int(min(start_chunk + chunk_size, num_mnks_by_algo + 1))
        print("Completed {:,} tasks out of {:,}".format(i, num_mnks_by_algo))

        # Create checkpoint file or load checkpointed data from it
        checkpoint_file_name = os.path.join(
            ckpt_folder_name, "chunk_{}-{}.json".format(start_chunk, end_chunk)
        )
        if os.path.exists(checkpoint_file_name):
            with open(checkpoint_file_name, "r") as f:
                optimal_kernels_list__ = json.load(f)
                optimal_kernels_list_ = list()
                for i, optker in enumerate(optimal_kernels_list__):
                    optimal_kernels_list_.append({})
                    for k, v in optker.items():
                        algo = v.pop("algorithm")
                        optimal_kernels_list_[i][to_tuple(k)] = kernel_algorithm[algo](
                            **v
                        )
            print("Read chunk {}-{}\n".format(start_chunk, end_chunk))

        else:

            if njobs == 1:

                # Ignore joblib and run serially:
                for mnk, algo in mnks_by_algo:
                    gc.collect()
                    print("Find optimal kernels for mnk=", mnk, ", algo=", algo)
                    optimal_kernels_list_ = find_optimal_kernel(
                        mnk,
                        algo,
                        tree[algo]["tree"],
                        tree[algo]["features"],
                        gpu_properties,
                        autotuning_properties,
                    )
            else:

                # Run prediction tasks in parallel with joblib
                optimal_kernels_list_ = Parallel(n_jobs=njobs, verbose=2)(
                    delayed(find_optimal_kernel, check_pickle=True)(
                        mnk,
                        algo,
                        tree[algo]["tree"],
                        tree[algo]["features"],
                        gpu_properties,
                        autotuning_properties,
                    )
                    for mnk, algo in mnks_by_algo[start_chunk:end_chunk]
                )

            optimal_kernels_list_ = remove_empty_entries(optimal_kernels_list_)
            with open(checkpoint_file_name, "w") as f:
                optimal_kernels_list__ = list()
                for i, optker in enumerate(optimal_kernels_list_):
                    optimal_kernels_list__.append({})
                    for k, v in optker.items():
                        optimal_kernels_list__[i][to_string(k)] = v.as_dict
                json.dump(optimal_kernels_list__, f)

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

    # Find optimal kernel per mnk among the different algorithm possibilities
    optimal_kernels = dict()
    for mnk, candidate_kernels in optimal_kernels_mnk_algo.items():
        m, n, k = mnk
        optimal_kernel_mnk = sorted(
            candidate_kernels, key=lambda x: x.perf, reverse=True
        )[:top_k]
        optimal_kernels[(m, n, k)] = optimal_kernel_mnk[0]

    return optimal_kernels


def get_baseline_kernels(mnks_to_predict, gpu_propertes, autotuning_properties):

    print("Getting baseline kernels")
    baseline_algorithm = "medium"
    baseline_kernels = list()
    for m, n, k in mnks_to_predict:
        baseline_kernels[(m, n, k)] = kernel_algorithm[baseline_algorithm].baseline(
            m, n, k, gpu_propertes, autotuning_properties
        )

    return baseline_kernels


# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Update parameter file with new optimal parameter predictions given newly trained decision trees.

        This script is part of the workflow for predictive modelling of optimal libsmm_acc parameters.
        For more details, see README.md.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--params",
        metavar="parameters_GPU.json",
        default="../parameters/parameters_P100.json",
        help="Parameter file to read and update with predictions",
    )
    parser.add_argument(
        "-j", "--njobs", type=int, default=-1, help="Number of joblib jobs"
    )
    parser.add_argument(
        "--baseline",
        default=False,
        help="Generate a parameter file corresponding to the baseline of a predictive model",
    )
    parser.add_argument(
        "--tiny",
        default=None,
        help="Path to model trained for algorithm 'tiny'. If not given, ignore this algorithm.",
    )
    parser.add_argument(
        "--small",
        default=None,
        help="Path to model trained for algorithm 'small'. If not given, ignore this algorithm.",
    )
    parser.add_argument(
        "--medium",
        default=None,
        help="Path to model trained for algorithm 'medium'. If not given, ignore this algorithm.",
    )
    parser.add_argument(
        "--largeDB1",
        default=None,
        help="Path to model trained for algorithm 'largeDB1'. If not given, ignore this algorithm.",
    )
    parser.add_argument(
        "--largeDB2",
        default=None,
        help="Path to model trained for algorithm 'largeDB2'. If not given, ignore this algorithm.",
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        default=2000,
        help="Chunk size for dispatching joblib jobs. If memory errors are experienced, reduce this number",
    )

    args = parser.parse_args()
    paths_to_models = dict()
    for algo in kernel_algorithm.keys():
        paths_to_models[algo] = args.__dict__[algo]
    main(args.params, args.njobs, args.baseline, paths_to_models, args.chunk_size)
