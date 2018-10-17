########################################################################################################################
# DBCSR optimal parameters prediction
# Shoshana Jakobovits
# August-September 2018
########################################################################################################################

import numpy as np
import pandas as pd
import sys
import pickle
import json
from itertools import product
from optparse import OptionParser
from kernels.cusmm_dnt import kernel_algorithm
from parameter_space_utils import get_feature
from sklearn.tree import DecisionTreeRegressor


########################################################################################################################
# Formatting and printing helpers
########################################################################################################################
def get_kernel(**params):
    return kernel_algorithm[params.pop('algorithm')](**params)


def combinations(*sizes):
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
def get_features(features, params):
    """

    :param features: names of features to compute
    :param params: base parameters from which to compute features
    :return:
    """
    feat_values = list()
    for feat in features:
        feat_values.append(get_feature(feat, **params))
    return feat_values


def get_parameter_space(m, n, k):
    param_space = list()
    for kernel_algo in kernel_algorithm.values():
        param_space += kernel_algo.promising_parameters(m, n, k)
    return param_space


def predict_performances(tree, parameter_space):
    """

    :param tree:
    :param parameter_space: list of kernel objects
    :return: parameter_space: list of kernel objects with predicted performance
    """
    for algo in kernel_algorithm.values():
        parameter_space_ = [p for p in parameter_space if p.algorithm == algo]
        if len(parameter_space_) > 0:
            for params in parameter_space_:
                features = get_features(tree[algo]['features'], params)
                params['perf'] = tree[algo]['tree'].predict(features)

    return parameter_space


########################################################################################################################
# Main
########################################################################################################################
def main(argv):
    """
    Update parameter file with new predictions given newly trained decision trees
    """
    del argv  # unused

    parser = OptionParser()
    parser.add_option("-p", "--params", metavar="filename.txt",
                      default="parameters_P100.txt", help="Default: %default")
    (options, args) = parser.parse_args(sys.argv)

    ####################################################################################################################
    # Load pre-autotuned parameters
    all_kernels = [get_kernel(**params) for params in json.load(open(options.params))]
    print("Libcusmm: Found %d existing parameter sets."%len(all_kernels))
    autotuned_mnks = [(k.m, k.n, k.k) for k in all_kernels if k['autotuned']]
    autotuned_kernels_ = [k for k in all_kernels if k['autotuned']]
    autotuned_kernels = dict(zip(autotuned_mnks, autotuned_kernels_))

    ####################################################################################################################
    # Load GridSearchCV objects
    tree = {
        'tiny':     {'file': 'model_selection/tiny/2018-10-11--16-00_RUNS20/Decision_Tree_0/cv.p'},
        #'small':    {'file': ''},
        #'medium':   {'file': ''},
        'largeDB1': {'file': 'model_selection/largeDB1/2018-10-15--09-29/Decision_Tree_0/cv.p'},
        'largeDB2': {'file': 'model_selection/largeDB2/2018-10-15--07-43/Decision_Tree_0/cv.p'}
    }

    for algo, v in tree.items():
        x_train, _, _, _, _, _, gs = pickle.load(open(v['file'], 'rb'))
        tree[algo]['tree'] = gs.best_estimator_
        tree[algo]['features'] = x_train.columns.values

    ####################################################################################################################
    # Evaluation
    optimal_kernels = ''
    top_k = 1
    for m, n, k in combinations(range(4, 46)):
        if (m, n, k) in autotuned_kernels.keys():
            optimal_kernels += autotuned_kernels[(m, n, k)]
        else:
            parameter_space = get_parameter_space(m, n, k)
            parameter_space = predict_performances(tree, parameter_space)
            optimal_kernel = sorted(parameter_space, key=lambda x: x['perf'], reverse=True)[:top_k]
            optimal_kernels += optimal_kernel

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
