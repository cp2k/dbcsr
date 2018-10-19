#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
import pickle
import json
from itertools import product
from optparse import OptionParser
from kernels.cusmm_dnt_helper import arch_number, kernel_algorithm
from parameter_space_utils import PredictiveParameters
from sklearn.tree import DecisionTreeRegressor


########################################################################################################################
# Formatting and printing helpers
########################################################################################################################
def get_kernel(**params):
    return kernel_algorithm[params.pop('algorithm')](**params)


def combinations(sizes):
    return list(product(sizes, sizes, sizes))


########################################################################################################################
# To allow pickling (TMP)
########################################################################################################################
def worse_case_scorer(estimator, X, y, top_k):
    """
    :param estimator: the model that should be evaluated
    :param X: validation data
    :param y: ground truth target for X
    :return: score: a floating point number that quantifies the estimator prediction quality on X, with reference to y
    """
    mnk = pd.DataFrame()
    mnk['mnk'] = X['mnk'].copy()
    y_pred = estimator.predict(X.drop(['mnk'], axis=1))
    score = worse_rel_perf_loss_of_k(y, y_pred, top_k, mnk)
    return -score  # by scikit-learn convention, higher numbers are better, so the value should be negated


def worse_case_scorer_top1(estimator, X, y):
    return worse_case_scorer(estimator, X, y, 1)


def worse_case_scorer_top3(estimator, X, y):
    return worse_case_scorer(estimator, X, y, 3)


def worse_case_scorer_top5(estimator, X, y):
    return worse_case_scorer(estimator, X, y, 5)


def mean_scorer(estimator, X, y, top_k):
    """
    :param estimator: the model that should be evaluated
    :param X: validation data
    :param y: ground truth target for X
    :return: score: a floating point number that quantifies the estimator prediction quality on X, with reference to y
    """
    mnk = pd.DataFrame()
    mnk['mnk'] = X['mnk'].copy()
    y_pred = estimator.predict(X.drop(['mnk'], axis=1))
    score = mean_rel_perf_loss_of_k(y, y_pred, top_k, mnk)
    return -score  # by scikit-learn convention, higher numbers are better, so the value should be negated


def mean_scorer_top1(estimator, X, y):
    return mean_scorer(estimator, X, y, 1)


def mean_scorer_top3(estimator, X, y):
    return mean_scorer(estimator, X, y, 3)


def mean_scorer_top5(estimator, X, y):
    return mean_scorer(estimator, X, y, 5)


########################################################################################################################
# Predict performances
########################################################################################################################
def get_parameter_space(m, n, k, gpu):
    param_space = list()
    for algo, kernel_algo in kernel_algorithm.items():
        param_space_algo = kernel_algo.promising_parameters(m, n, k, gpu, autotuning)
        param_space += [{**p, **{'algorithm': algo}} for p in param_space_algo]
    return param_space


def predict_performances(tree, parameter_space, gpu, autotuning):
    """
    Given a list of Kernel objects, predict their performance using the given decision tree
    :param tree: decision tree, predicts the performance in Gflop/s of a given set of kernel parameters
    :param parameter_space: list of kernel objects
    :return: parameter_space: list of kernel objects with predicted performance (field in dict)
    """
    for kernel_algo in kernel_algorithm.keys():
        parameter_space_algo = [p for p in parameter_space if p['algorithm'] == kernel_algo]
        if len(parameter_space_algo) > 0:
            for params in parameter_space_algo:
                p = PredictiveParameters(params, gpu, autotuning)
                features = p.get_features(tree[kernel_algo]['features'])
                params['perf'] = tree[kernel_algo]['tree'].predict(features)

    return parameter_space


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
    # Load pre-autotuned parameters
    assert options.params in arch_number.keys(), "Cannot find compute version for file " + str(options.params)
    arch = arch_number[options.params]
    with open('kernels/gpu_properties.json') as f:
        gpu_properties = json.load(f)["sm_" + str(arch)]
    with open(options.params) as f:
        all_kernels = [get_kernel(**params) for params in json.load(f)]
    print("Libcusmm: Found %d existing parameter sets." % len(all_kernels))
    autotuned_mnks = [(k.m, k.n, k.k) for k in all_kernels if k.autotuned]
    autotuned_kernels_ = [k for k in all_kernels if k.autotuned]
    autotuned_kernels = dict(zip(autotuned_mnks, autotuned_kernels_))

    ####################################################################################################################
    # Load GridSearchCV objects
    tree = dict()
    for algo, v in kernel_algorithm.items():
        if algo in ['tiny', 'largeDB2']:
            tree[algo] = dict()
            tree[algo]['file'] = os.path.join(str(options.trees), 'tree_' + algo + '.p')
            x_train, _, _, _, _, _, gs = pickle.load(open(tree[algo]['file'], 'rb'))
            tree[algo]['tree'] = gs.best_estimator_
            features = x_train.columns.values.tolist()
            features.remove('mnk')
            tree[algo]['features'] = features

    ####################################################################################################################
    # Evaluation
    optimal_kernels = list()
    top_k = 1
    for m, n, k in combinations(list(range(4, 46))):
        if (m, n, k) in autotuned_kernels.keys():
            optimal_kernels.append(autotuned_kernels[(m, n, k)])
        else:
            parameter_space = get_parameter_space(m, n, k, gpu_properties)
            parameter_space = predict_performances(tree, parameter_space, gpu_properties, autotuning_properties)
            optimal_kernel = sorted(parameter_space, key=lambda x: x['perf'], reverse=True)[:top_k]
            optimal_kernels.append(optimal_kernel)

    ####################################################################################################################
    # Write to file
    with open(options.params, 'w') as f:
        s = json.dumps([kernel.as_dict for kernel in optimal_kernels])
        s = s.replace('}, ', '},\n')
        s = s.replace('[', '[\n')
        s = s.replace(']', '\n]')
        f.write(s)


#===============================================================================
main(argv=sys.argv[1:])

#EOF
