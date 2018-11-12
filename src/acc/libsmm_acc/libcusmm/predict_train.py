#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import datetime
import random
import numpy as np
import pandas as pd
from optparse import OptionParser
from predict_helpers import *
from kernels.cusmm_dnt_helper import params_dict_to_kernel



########################################################################################################################
# Selected features and optimized hyperparameters
########################################################################################################################
selected_features = {
    'tiny': [  # 2018-10-11--16-00_RUNS20/Decision_Tree_12
        'nblks',
        'ru_tinysmallmed_unroll_factor_c_total',
        'ru_tiny_smem_per_block',
        'Gflops',
        'size_a',
        'size_c',
        'threads_per_blk',
        'ru_tiny_max_parallel_work',
        'ru_tinysmallmed_unroll_factor_a_total',
        'ru_tiny_buf_size'
    ],
    'small': [  # 2018-10-16--15-26
        'grouping',
        'k',
        'm',
        'minblocks',
        'threads_per_blk',
        'nthreads',
        'Gflops',
        'ru_tinysmallmed_unroll_factor_b',
        'ru_smallmedlarge_cmax',
        'ru_smallmedlarge_T',
        'ru_smallmedlarge_min_threads',
        'ru_smallmed_buf_size',
        'Koth_small_Nmem_shared',
    ],
    'medium': [  # result of one of the RFECVs, copied to journal
        'k',
        'm',
        'n',
        'minblocks',
        'threads_per_blk',
        'tile_m',
        'tile_n',
        'size_a',
        'size_c',
        'nthreads',
        'sm_desired',
        'nblocks_per_sm_lim_blks_warps',
        'Gflops',
        'ru_tinysmallmed_unroll_factor_a',
        'ru_tinysmallmed_unroll_factor_b',
        'ru_tinysmallmed_unroll_factor_c_total',
        'ru_smallmedlarge_cmax',
        'ru_smallmedlarge_rmax',
        'ru_smallmedlarge_T',
        'ru_smallmedlarge_min_threads',
        'ru_smallmed_unroll_factor_c',
        'ru_smallmed_loop_matmul',
        'ru_smallmed_max_parallel_work',
        'ru_smallmed_regs_per_thread'
    ],
    'largeDB1': [  # 2018-10-15--02-47
        'size_b',
        'minblocks',
        'tile_n',
        'ru_large_Pc',
        'size_c',
        'size_a',
        'Koth_large_Nmem_glob',
        'nblocks_per_sm_lim_blks_warps',
        'ru_smallmedlarge_cmax',
        'tile_m',
        'm',
        'sm_desired',
        'ru_large_Pa',
        'ru_large_loop_matmul',
        'ru_smallmedlarge_rmax',
        'w',
        'ru_large_unroll_factor_b',
        'threads_per_blk',
        'ru_large_unroll_factor_a',
        'ru_large_Pb',
        'k',
        'Gflops'
    ],
    'largeDB2': [  # 2018-10-15--07-43
        'size_a',
        'size_b',
        'tile_m',
        'sm_desired',
        'ru_smallmedlarge_rmax',
        'ru_large_loop_matmul',
        'm',
        'Koth_large_Nmem_glob',
        'ru_large_unroll_factor_b',
        'tile_n',
        'w',
        'ru_large_Pc',
        'k',
        'ru_smallmedlarge_cmax',
        'ru_large_Pa',
        'ru_large_unroll_factor_a',
        'size_c',
        'threads_per_blk',
    ]
}
optimized_hyperparameters = {
    'tiny': {
        'max_depth': 39,
        'min_samples_leaf': 8,
        'min_samples_split': 11
    },
    'small': {
        'max_depth': 18,
        'min_samples_leaf': 2,
        'min_samples_split': 13
    },
    'medium': {  # common sense
        'max_depth': 18,
        'min_samples_leaf': 2,
        'min_samples_split': 13
    },
    'largeDB1': {
        'max_depth': 18,
        'min_samples_leaf': 13,
        'min_samples_split': 5
    },
    'largeDB2': {
        'max_depth': 18,
        'min_samples_leaf': 5,
        'min_samples_split': 5
    }
}


########################################################################################################################
# Printing and dumping helpers
########################################################################################################################
def get_log_folder(algo, prefitted_model_folder):
    """Create a unique log folder for this run in which logs, plots etc. will be stored """
    if len(prefitted_model_folder) == 0:
        file_signature = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")
        folder = os.path.join("model_selection", os.path.join(algo, file_signature))
        log_file = os.path.join(folder, "log.txt")
        if not os.path.exists(folder):
            os.makedirs(folder)
    else:  # Use the original folder as a log folder, but create a new log file
        folder = prefitted_model_folder
        log_file_signature = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")
        log_file = os.path.join(folder, "log_" + log_file_signature + ".txt")

    return folder, log_file


def print_and_log(msg):
    if not isinstance(msg, str):
        msg = str(msg)
    log = '\n' + msg
    print(msg)
    return log


########################################################################################################################
# Custom loss functions and scorers
########################################################################################################################
def perf_loss(y_true, y_pred, top_k, X_mnk):
    """
    Compute the relative performance losses per mnk if one were to measure the top-k best predicted sets of parameters
    and pick the best out of this top-k

    :param y_true: ground truth performances (performance scaled between 0 and 1)
    :param y_pred: estimated performances (performance scaled between 0 and 1)
    :param top_k: number of top performances to measure
    :param X_mnk: corresponding mnks
    :return: perf_losses: array of relative performance losses (in %), one array element per mnk
    """
    assert len(y_true.index) == y_pred.flatten().size
    assert len(y_true.index) == len(X_mnk.index)

    perf_losses = list()
    mnks = np.unique(X_mnk['mnk'].values)
    for mnk in mnks:

        # Get performances per mnk
        idx_mnk = np.where(X_mnk == mnk)[0].tolist()
        assert (len(idx_mnk) > 0), "idx_mnk is empty"
        y_true_mnk = y_true.iloc[idx_mnk]
        y_pred_mnk = y_pred[idx_mnk]

        # Get top-k best predicted performances
        if top_k != 1:
            top_k_idx = np.argpartition(-y_pred_mnk, top_k)[:top_k]
        else:
            top_k_idx = np.argmax(y_pred_mnk)
        y_correspmax = y_true_mnk.iloc[top_k_idx]

        # Chosen max perf. among predicted max performances
        maxperf_chosen = np.amax(y_correspmax)

        # True Max. performances
        # maxperf = float(y_true_mnk.max(axis=0))  # true max. performance (this is always = 1 since we scale)
        # assert maxperf >= 0, "Found non-positive value for maxperf: " + str(maxperf)

        # Relative performance loss incurred by using model-predicted parameters instead of autotuned ones [%]
        perf_loss = 100 * (1.0 - maxperf_chosen)
        perf_losses.append(perf_loss)

    return perf_losses


def worse_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk):
    y = np.array(perf_loss(y_true, y_pred, top_k, X_mnk))
    return float(y.max(axis=0))


def mean_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk):
    y = np.array(perf_loss(y_true, y_pred, top_k, X_mnk))
    return float(y.mean(axis=0))


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


########################################################################################################################
# Read and prepare data
########################################################################################################################
def perf_Kothapalli(N_mem, nblks, threads_per_blk, Gflops):
    c_K = nblks * threads_per_blk * N_mem  # ignore number of threads per warp
    return Gflops / c_K # ignore clock rate


def add_Kothapalli(df, gpu, Nmem_glob, Nmem_shared, Nmem, perf_K):
    df[perf_K] = np.vectorize(perf_Kothapalli)(
        gpu['Global memory access latency'] * df[Nmem_glob] + gpu['Shared memory access latency'] * df[Nmem_shared],
        df['nblks'], df['threads_per_blk'], df['Gflops'])


def scale(Y, X_mnk, scale_on_all_algos, params=None):
    """
    Scale raw performances in [Gflop/s] between 0 and 1, where
        0 = 0 Gflop/s
        1 = performance equal to autotuned maximum
    :param Y: dataframe with column 'perf'
    :param X_mnk: dataframe containing the corresponding mnk-triplets
    :return: numpy array of scaled performances
    """

    # For each mnk, get autotuned max performance
    assert 'perf' in Y.columns.values
    print("Scaling with scale_on_all_algos =", scale_on_all_algos)

    if scale_on_all_algos:
        import json

        assert params is not None
        with open(params) as f:
            all_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]
        mnk_string = "{}x{}x{}"
        autotuned_mnks = [mnk_string.format(k.m, k.n, k.k) for k in all_kernels if k.autotuned]
        autotuned_perfs = [k.perf for k in all_kernels if k.autotuned]
        autotuned_max = dict(zip(autotuned_mnks, autotuned_perfs))

    else:

        autotuned_max = dict()
        mnks = np.unique(X_mnk['mnk'].values)

        for mnk in mnks:

            # Get performances per mnk
            idx_mnk = np.where(X_mnk == mnk)[0].tolist()
            y_mnk = Y.iloc[idx_mnk]

            # Store maxperf
            maxperf = float(y_mnk.max(axis=0))  # max. performance found through autotuning
            autotuned_max[mnk] = maxperf

    # Scale performances
    def scale_perf(perf, mnk):
        """For a given mnk and a given performance on this mnk, return the scaled performance"""
        return perf / autotuned_max[mnk]

    vec_scale_perf = np.vectorize(scale_perf)
    Y_scaled = vec_scale_perf(Y.values, X_mnk.values)
    assert np.any(Y_scaled >= 0)  # sanity check
    assert np.any(Y_scaled <= 1)  # sanity check
    return Y_scaled


def read_data(algo, read_from, nrows, scale_on_all_algos, params, log):

    ####################################################################################################################
    # Read and fix 'X'
    X_file = os.path.join(read_from, 'train_all_' + algo + '_X.csv')
    log += print_and_log('Read training data X from ' + X_file)
    X = pd.read_csv(X_file, index_col=0)
    log += print_and_log('X    : {:>8} x {:>8} ({:>2} MB)'.format(X.shape[0], X.shape[1], sys.getsizeof(X)/10**6))

    # Fix Gflops and Koth_perf, if needed
    if X['Gflops'].isna().any():
        stack_size = 16005
        one_col = np.ones(len(X['n']))
        X['n_iter'] = np.maximum(3 * one_col, 12500 * (one_col // (X['m'].values * X['n'].values * X['k'].values)))
        X['Gflops'] = X['n_iter'] * stack_size * X['m'] * X['n'] * X['k'] * 2 * 10 ** (-9)
        X.drop(['n_iter'], axis=1, inplace=True)

        if 'Koth_med_perf_K' in X.columns.values:
            gpu = {'Shared memory access latency': 4, 'Global memory access latency': 500}
            add_Kothapalli(X, gpu, 'Koth_med_Nmem_glob', 'Koth_med_Nmem_shared', 'Koth_med_Nmem', 'Koth_med_perf_K')

    ####################################################################################################################
    # Read and fix 'X_mnk'
    X_mnk_file = os.path.join(read_from, 'train_all_' + algo + '_X_mnk.csv')
    log += print_and_log('Read training data X_mnk from ' + X_mnk_file)
    X_mnk = pd.read_csv(X_mnk_file, index_col=0, nrows=nrows)
    log += print_and_log('X_mnk: {:>8} x {:>8} ({:>2} MB)'.format(X_mnk.shape[0], X_mnk.shape[1],
                                                                  sys.getsizeof(X_mnk)/10**6))

    ####################################################################################################################
    # Read and fix 'Y'
    log += print_and_log('Read training data Y ...')
    if scale_on_all_algos:
        Y_scaled_file = os.path.join(read_from, 'train_all_' + algo + '_Y_scaled_all_algos.csv')
    else:
        Y_scaled_file = os.path.join(read_from, 'train_all_' + algo + '_Y_scaled.csv')
    if os.path.exists(Y_scaled_file):

        log += print_and_log('from ' + Y_scaled_file)
        Y = pd.read_csv(Y_scaled_file, index_col=0, nrows=nrows)

    else:

        Y_raw_file = os.path.join(read_from, 'train_all_' + algo + '_Y_perf.csv')
        log += print_and_log('from ' + Y_raw_file + "(raw Gflops) and scale performances")
        Y = pd.read_csv(Y_raw_file, index_col=0, nrows=nrows)

        if 0 in Y['perf'].tolist():

            # Remove 0-performances if they exist
            print("Found 0-values in perf")
            X['perf'] = Y['perf']
            X['mnk'] = X_mnk['mnk']
            print(X.shape)
            X = X.loc[X['perf'] != 0]
            del Y
            Y = pd.DataFrame()
            Y['perf'] = X['perf']
            del X_mnk
            X_mnk = pd.DataFrame()
            X_mnk['mnk'] = X['mnk']
            X.drop(['perf', 'mnk'], axis=1, inplace=True)

            # Re-write with dropped rows
            print("Write X, X_mnk to file")
            X_mnk.to_csv(X_mnk_file)
            X.to_csv(X_file)

        # Scale performances
        print("Scale Y")
        Y['perf_scaled'] = scale(Y, X_mnk, scale_on_all_algos, params)
        Y.drop(['perf'], axis=1, inplace=True)
        print((Y["perf_scaled"] == 1.0).sum(), ", ", len(np.unique(X_mnk["mnk"].values)))

        # Re-write CSVs
        print("Write Y to file")
        Y.to_csv(Y_scaled_file)
        log += print_and_log('Write scaled performances to ' + Y_scaled_file)

    log += print_and_log('Y    : {:>8} x {:>8} ({:>2} MB)'.format(Y.shape[0], Y.shape[1], sys.getsizeof(Y)/10**6))

    ####################################################################################################################
    # Describe and log
    n_features = len(list(X.columns))
    predictor_names = X.columns.values
    log += print_and_log('\nPredictor variables: (' + str(n_features) + ')')
    for i, p in enumerate(predictor_names):
        log += print_and_log("\t{:2}) {}".format(i+1, p))

    return X, X_mnk, Y, log


########################################################################################################################
# Predictive modelling
########################################################################################################################
def get_DecisionTree_model(algo, n_features):
    from itertools import chain
    from sklearn.tree import DecisionTreeRegressor

    # Fixed parameters
    model_name = "Decision_Tree"
    splitting_criterion = "mse"
    splitter = "best"
    max_features = None
    max_leaf_nodes = None

    # Hyper-parameters to optimize
    if algo == 'medium':
        max_depth = chain(range(6, 13, 2), range(15, 19, 3))
        param_grid = {'max_depth': list(max_depth)}
    else:
        if algo == 'tiny':
            step_small = 1
            step_med = 3
            max_depth = chain(range(4, n_features, step_small), range(n_features, n_features*3, step_med))
            min_samples_split = chain(range(2, 5, step_small), range(8, n_features, step_med))
            min_samples_leaf = chain(range(1, 5, step_small), range(8, n_features, step_med))
        else:
            max_depth = chain(range(4, 13, 2), range(15, 19, 3))
            min_samples_split = [2, 5, 13, 18]
            min_samples_leaf = [2, 5, 13, 18]
        param_grid = {
            'max_depth': list(max_depth),
            'min_samples_split': list(min_samples_split),
            'min_samples_leaf': list(min_samples_leaf)
        }

    # Tree model
    model = DecisionTreeRegressor(
        criterion=splitting_criterion,
        splitter=splitter,
        min_samples_split=optimized_hyperparameters[algo]["min_samples_split"],
        min_samples_leaf=optimized_hyperparameters[algo]["min_samples_leaf"],
        max_depth=optimized_hyperparameters[algo]["max_depth"],
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes
    )

    return model, model_name, param_grid


def get_RandomForest_model(algo, njobs, ntrees):
    from itertools import chain
    from sklearn.ensemble import RandomForestRegressor

    # Fixed parameters
    model_name = "Random Forest"
    bootstrap = True
    splitting_criterion = "mse"
    max_features = 'sqrt'

    # Parameters to optimize
    step_big = 50
    step_small = 5
    n_estimators = chain(range(1, 10, step_small), range(50, 200, step_big))
    param_grid = {**optimized_hyperparameters[algo], 'n_estimators': list(n_estimators)}

    # Random Forest model
    model = RandomForestRegressor(
        criterion=splitting_criterion,
        n_estimators=ntrees,
        min_samples_split=optimized_hyperparameters[algo]["min_samples_split"],
        min_samples_leaf=optimized_hyperparameters[algo]["min_samples_leaf"],
        max_depth=optimized_hyperparameters[algo]["max_depth"],
        bootstrap=bootstrap,
        max_features=max_features,
        n_jobs=njobs
    )

    return model, model_name, param_grid


def get_train_test_partition(X, Y, X_mnk, test, train=None):
    """
    Perform train/test partition
    :param X, Y, X_mnk: to partition
    :param test: ndarray, test-indices
    :param train (optional): ndarray
    :return:
    """
    if train is None:  # Retrieve training indices
        all_indices = set(range(len(Y.index)))
        train = list(all_indices - set(test))

    X_train = X.iloc[train, :]          # train: use for hyper-parameter optimization (via CV) and training
    X_test = X.iloc[test, :]            # test : use for evaluation of 'selected/final' model
    X_mnk_train = X_mnk.iloc[train, :]
    X_mnk_test = X_mnk.iloc[test, :]
    Y_train = Y.iloc[train, :]
    Y_test = Y.iloc[test, :]

    return X_train, Y_train, X_mnk_train, X_test, Y_test, X_mnk_test


def tune_and_train(X, X_mnk, Y, options, folder, log):


    ####################################################################################################################
    # Get options
    algo = options.algo
    model_to_train = options.model
    splits = options.splits
    njobs = options.njobs
    tune = options.tune
    plot_all = options.plot_all
    ntrees = options.ntrees
    results_file = os.path.join(folder, "feature_tree.p")

    ####################################################################################################################
    # Predictive model
    if model_to_train == "DT":
        model, model_name, param_grid = get_DecisionTree_model(algo, len(X.columns.values))
    elif model_to_train == "RF":
        model, model_name, param_grid = get_RandomForest_model(algo, njobs, ntrees)
    else:
        assert False, "Cannot recognize model: " + model_to_train + ". Options: DT, RF"
    log += print_and_log("\nStart tune/train for model " + model_name + " with parameters:")
    log += print_and_log(model)

    ####################################################################################################################
    # Testing splitter (train/test-split)
    from sklearn.model_selection import GroupShuffleSplit
    cv = GroupShuffleSplit(n_splits=2, test_size=0.2)
    train, test = cv.split(X, Y, groups=X_mnk['mnk'])
    train = train[0]
    test = test[0]
    X_train, Y_train, X_mnk_train, X_test, Y_test, X_mnk_test = get_train_test_partition(X, Y, X_mnk, train, test)
    del X, X_mnk, Y  # free memory
    log += print_and_log("\nComplete train/test split")

    ####################################################################################################################
    # Cross-validation splitter (train/validation-split)
    n_splits = splits
    test_size = 0.3
    cv = GroupShuffleSplit(n_splits=n_splits, test_size=test_size)
    predictor_names = X_train.columns.values

    if tune:  # Perform feature selection and hyperparameter optimization

        log += print_and_log("\nStart feature selection")

        ################################################################################################################
        # Feature selection
        from sklearn.feature_selection import RFECV
        log += print_and_log('----------------------------------------------------------------------------')
        log += print_and_log("Selecting optimal features among:\n" + str(predictor_names) + '\n')
        if algo in ['small', 'medium']:
            rfecv = RFECV(estimator=model, step=3, n_jobs=njobs, cv=cv, verbose=2, min_features_to_select=14)
        else:
            rfecv = RFECV(estimator=model, step=1, n_jobs=njobs, cv=cv, verbose=1)
        fit = rfecv.fit(X_train, Y_train, X_mnk_train['mnk'])
        log += print_and_log("Optimal number of features : %d" % rfecv.n_features_)

        selected_features_ = list()
        for i, f in enumerate(predictor_names):
            if fit.support_[i]:
                selected_features_.append(f)
        log += print_and_log("\nSelected features:")
        for feature in selected_features_:
            log += print_and_log("\t{}".format(feature))
        log += print_and_log("\n")

        features_to_drop = [f for f in predictor_names if f not in selected_features_]
        X_train = X_train.drop(features_to_drop, axis=1)
        X_test = X_test.drop(features_to_drop, axis=1)

        ################################################################################################################
        # Hyperparameter optimization
        log += print_and_log("\nCompleted feature selection, start hyperparameter optimization")

        # Grid search
        from sklearn.model_selection import GridSearchCV
        log += print_and_log('----------------------------------------------------------------------------')
        log += print_and_log('Parameter grid:\n' + str(param_grid))
        X_train["mnk"] = X_mnk_train['mnk']  # add to X-DataFrame (needed for scoring function)
        scoring = {'worse_top-1': worse_case_scorer_top1, 'mean_top-1': mean_scorer_top1}
        decisive_score = 'mean_top-1'
        if algo in ['tiny', 'largeDB1', 'largeDB2']:
            verbosity_level = 1
        else:
            verbosity_level = 2
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            pre_dispatch=8,
            n_jobs=njobs,
            verbose=verbosity_level,
            refit=decisive_score,
            return_train_score=False  # incompatible with ignore_in_fit
        )
        gs.fit(X_train, Y_train, X_mnk_train['mnk'], ignore_in_fit=["mnk"])

        describe_hpo(gs, X_test, Y_test, '', plot_all)

        safe_pickle([gs.param_grid, gs.cv_results_, gs.best_params_], os.path.join(folder, "cv_results.p"))
        safe_pickle([X_train.columns.values, gs.best_estimator_, test], results_file)

        plot_cv_scores(gs.param_grid, gs.cv_results_, gs.best_params_, folder, algo, splits)

        log += print_and_log("\nCompleted hyperparameter optimization, wrote results to " + results_file)
        log += print_and_log('----------------------------------------------------------------------------')
        return_model = gs

    else:

        ################################################################################################################
        # Load selected features and hyperparameters
        features_to_drop = [f for f in predictor_names if f not in selected_features[algo]]
        X_train = X_train.drop(features_to_drop, axis=1)
        X_test = X_test.drop(features_to_drop, axis=1)

        ################################################################################################################
        # Fit
        log += print_and_log('----------------------------------------------------------------------------')
        log += print_and_log("\nStart fitting model with predictors:\n")
        for i, p in enumerate(X_train.columns.values):
            log += print_and_log("\t{:>2}) {}".format(i+1, p))

        model.fit(X_train, Y_train)
        safe_pickle([X_train.columns.values, model, test], results_file)
        log += print_and_log("\nCompleted fit, wrote results to " + results_file)
        log += print_and_log('----------------------------------------------------------------------------')
        return_model = model

    # Return
    return X_train, Y_train, X_mnk_train, X_test, Y_test, X_mnk_test, return_model, log


########################################################################################################################
# Describe and evaluate model
########################################################################################################################
def describe_hpo(gs, X, Y, log, plot_all):
    predictor_names = X.columns.values.tolist()
    log += print_and_log('Predictor variables:')
    for p in predictor_names:
        log += print_and_log("\t{}".format(p))

    log += print_and_log("\nBest parameters set found on development set:")
    log += print_and_log(gs.best_params_)

    log += print_and_log("\nBest estimator:")
    best_estimator = gs.best_estimator_
    log += print_and_log(best_estimator)
    log += print_and_log('----------------------------------------------------------------------------')

    # Export tree SVG
    if plot_all:
        from dtreeviz.trees import dtreeviz
        log += print_and_log('\nExport tree to SVG:')
        viz = dtreeviz(best_estimator, X.values, Y.values.ravel(),
                       target_name='perf',
                       feature_names=predictor_names)
        viz.save("trytree.svg")
        viz.view()

    return log


def describe_model(model, X, Y, log, plot_all):
    predictor_names = X.columns.values.tolist()
    log += print_and_log('Model:')
    log += print_and_log(model)

    log += print_and_log('Predictor variables:')
    for p in predictor_names:
        log += print_and_log("\t{}".format(p))

    # Export tree SVG
    if plot_all:
        from dtreeviz.trees import dtreeviz
        log += print_and_log('\nExport tree to SVG:')
        viz = dtreeviz(model, X.values, Y.values.ravel(),
                       target_name='perf',
                       feature_names=predictor_names)
        viz.save("trytree.svg")
        viz.view()

    return log


def print_error(y_true, y_pred, X_mnk, log):
    result_line = "Relative performance loss compared to autotuned max:\n" + \
                  "top-{}: worse: {:>6.3f} [%], mean: {:>6.3f} [%]"
    for top_k in [1]:
        log += print_and_log(result_line.format(top_k,
                                               worse_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk),
                                               mean_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk)))
    return log


def plot_loss_histogram(y_true, y_pred, X_mnk, folder):
    import matplotlib.pyplot as plt

    # Get losses
    top_k = 3
    y = np.array(perf_loss(y_true, y_pred, top_k, X_mnk))

    # Losses-histogram
    num_bins = 50
    plt.hist(y, num_bins, facecolor='green', alpha=0.75)
    plt.xlabel("relative performance loss [%]")
    plt.ylabel("# occurrences")
    plt.title("Performance losses for top-k=" + str(top_k))
    plt.grid(True)
    plt.savefig(os.path.join(folder, "result_losses.svg"))


def plot_cv_scores(param_grid, results, best_pars, folder, algo, splits):
    # Inspired by http://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
    import matplotlib.pyplot as plt
    for p in param_grid.keys():

        plt.figure()
        plt.title("CV scores (" + algo + ")")
        plt.xlabel("parameter: " + p + "(best:" + str(best_pars) + ")")
        plt.ylabel("cv-score: relative perf loss [%] (mean over " + str(splits) + "folds)")
        ax = plt.gca()

        # Get the regular numpy array from the dataframe
        results_ = results.copy()
        groups_to_fix = list(best_pars.keys())
        groups_to_fix.remove(p)
        for g in groups_to_fix:
            results_ = results_.groupby('param_' + g).get_group(best_pars[g])
        X_axis = np.array(results_['param_' + p].values, dtype=float)
        X_axis_p = results_['param_' + p]

        for scorer, color in zip({'worse_top-1': worse_case_scorer_top1, 'mean_top-1': mean_scorer_top1}, ['b', 'g']):
            sample = 'test'
            style = '-'
            sample_score_mean = results_['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results_['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.05 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

            best_index = np.argmin(results_['rank_test_%s' % scorer])
            best_score = results_['mean_test_%s' % scorer][best_index]

            # Plot a dotted vertical line at the best score for that scorer marked by x
            ax.plot([X_axis_p[best_index], ] * 2, [0, best_score],
                    linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

            # Annotate the best score for that scorer
            ax.annotate("%0.2f" % best_score,
                        (X_axis_p[best_index], best_score + 0.005))

        plt.legend(loc="best")
        plt.grid(False)

        plt.savefig(os.path.join(folder, "cv_results_" + algo + "_" + p + ".svg"))


########################################################################################################################
# Main
########################################################################################################################
def main():

    parser = OptionParser()
    parser.add_option('-f', '--in_folder', metavar="foldername/",
                      default='tune_big/',
                      help='Folder from which to read data')
    parser.add_option('-a', '--algo', metavar="algoname",
                      default='tiny',
                      help='Algorithm to train on')
    parser.add_option('-b', '--scale_on_all_algos',
                      default=False,
                      help='Algorithm to train on')
    parser.add_option("-p", "--params", metavar="filename.json",
                      default="parameters_P100.json", help="Default: %default")
    parser.add_option('-c', '--plot_all',
                      default=False,
                      help='Plot more stuff' +
                           '(Warning: can be very slow for large trees and create very large files)')
    parser.add_option('-t', '--tune',
                      default=False, action='store_true',
                      help='Rune recursive feature selection and grid search on hyperparameters')
    parser.add_option('-m', '--model',
                      default='DT',
                      help='Model to train. Options: DT (Decision Trees), RF (Random Forests)')
    parser.add_option('-s', '--splits',
                      default='5', metavar="NUMBER", type="int",
                      help='Number of cross-validation splits used in RFECV and GridSearchCV')
    parser.add_option('-e', '--ntrees',
                      default=3, metavar="NUMBER", type="int",
                      help='Number of estimators in RF')
    parser.add_option('-j', '--njobs',
                      default='-1', metavar="NUMBER", type="int",
                      help='Number of cross-validation splits used in RFECV and GridSearchCV')
    parser.add_option('-r', '--nrows',
                      default=None, metavar="NUMBER", type="int",
                      help='Number of rows of data to load. Default: None (load all)')
    parser.add_option('-g', '--prefitted_model',
                      metavar="filename",
                      default='',
                      #default='model_selection/tiny/2018-11-07--12-01',
                      #default='model_selection/tiny/2018-11-07--16-10',
                      help='Path to pickled GridSearchCV object to load instead of recomputing')
    parser.add_option("-n", "--file_naive", metavar="filename.out",
                      default="libcusmm_timer_autotuned_multiply.10314368.o",
                      help="Naive result file to evaluate. Default: %default")
    options, args = parser.parse_args(sys.argv)

    ####################################################################################################################
    # Create folder to store results of this training and start a log
    folder, log_file = get_log_folder(options.algo, options.prefitted_model)
    log = ''

    ####################################################################################################################
    # Read data
    log += print_and_log('----------------------------------------------------------------------------')
    X, X_mnk, Y, log = read_data(options.algo, options.in_folder, options.nrows,
                                 options.scale_on_all_algos, options.params, log)
    if options.plot_all:
        plot_training_data(Y, X_mnk, folder, options.algo, os.path.join(folder, "y_scaled.svg"))

    ####################################################################################################################
    # Get or train model
    log += print_and_log('----------------------------------------------------------------------------')
    if len(options.prefitted_model) == 0:  # train a model

        log += print_and_log("\nPreparing to fit model...")
        X_train, Y_train, X_mnk_train, X_test, Y_test, X_mnk_test, model, log = \
            tune_and_train(X, X_mnk, Y, options, folder, log)

    else:  # fetch pre-trained model

        model_path = options.prefitted_model
        log += print_and_log("\nReading pre-fitted model from " + model_path)

        features, model, test_indices = safe_pickle_load(os.path.join(model_path, 'feature_tree.p'))
        features = features.tolist()
        if 'mnk' in features:
            features.remove('mnk')
        log += print_and_log("\nPerform train/test split")
        X_train, Y_train, X_mnk_train, X_test, Y_test, X_mnk_test = get_train_test_partition(X, Y, X_mnk, test_indices)
        predictor_names = X_train.columns.values.tolist()
        features_to_drop = [f for f in predictor_names if f not in features]
        X_train = X_train.drop(features_to_drop, axis=1)
        X_test = X_test.drop(features_to_drop, axis=1)

    ####################################################################################################################
    # Evaluate model
    log += print_and_log('----------------------------------------------------------------------------')
    log += print_and_log('Start model evaluation')
    log += describe_model(model, X_test, Y_test, log, options.plot_all)

    # Training error
    y_train_pred = model.predict(X_train)
    log += print_and_log('\nTraining error: (train&val)')
    log = print_error(Y_train, y_train_pred, X_mnk_train, log)

    # Test error
    y_test_pred = model.predict(X_test)
    log += print_and_log('\nTesting error:')
    log = print_error(Y_test, y_test_pred, X_mnk_test, log)

    # Training error (unscaled!)
    # ...

    # Test error (unscaled!)
    # ...

    # ...

    # Print histogram for "best" estimator
    log += print_and_log('\nPlot result histogram:')
    plot_loss_histogram(Y_test, y_test_pred, X_mnk_test, folder)

    # Plot CV results by evaluation metric
    if options.tune:
        cv_results_file = os.path.join(options.prefitted_model, "cv_results.p")
        if os.path.exists(cv_results_file):
            param_grid, cv_results, best_params = pickle.load(open(cv_results_file, 'rb'))
            log += print_and_log('\nPlot CV scores:')
            plot_cv_scores(param_grid, cv_results, best_params, folder, options.algo, options.splits)
        else:
            print("File", cv_results_file, "does not exist")

    # Plot choice goodness for a few mnks (validation-set)
    mnks_to_plot = random.sample(X_mnk_test.values, 1)
    for mnk in mnks_to_plot:

        # Get performances per mnk
        idx_mnk = np.where(X_mnk_test == mnk)[0].tolist()
        assert (len(idx_mnk) > 0), "idx_mnk is empty"
        y_true_mnk = Y_test.iloc[idx_mnk]
        y_pred_mnk = y_test_pred[idx_mnk]
        plot_choice_goodness(mnk, y_true_mnk, y_pred_mnk)

    # Read autotuned performances
    import json
    with open(options.params) as f:
        all_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]
    autotuned_mnks = [(k.m, k.n, k.k) for k in all_kernels if k.autotuned]
    autotuned_perfs_ = [k.perf for k in all_kernels if k.autotuned]
    autotuned_perfs = dict(zip(autotuned_mnks, autotuned_perfs_))

    # Read naïve autotuned result file
    file_perf_naive = "libcusmm_timer_autotuned_multiply.10314368.o"
    with open(file_perf_naive) as f:
        result_file = f.read().splitlines()
    results_naive = read_result_file(result_file)

    # Get results from this predictive model
    results = 1

    # compute ...
    perf_gain_res_over_naive = performance_gain(results_naive, results)
    rel_perf_gain_res_over_naive = relative_performance_gain(results_naive, results)
    perf_gain_autotuning_over_naive = performance_gain(results_naive_autotuned, autotuned_perfs)
    rel_gain_autotuning_over_naive = relative_performance_gain(results_naive_autotuned, autotuned_perfs)

    # Plot results (training set: predictive modelling VS naïve)
    plot_absolute_performance_gain(perf_gain_res_over_naive, 'Tested', 'naive', 'predictive model')
    plot_relative_performance_gain(rel_perf_gain_res_over_naive, 'Tested', 'naive', 'predictive model')
    plot_performance_gains(results_naive, results, 'Tested', 'naive', 'predictive model')

    # Plot results (training set: predictive modelling VS autotuned)
    plot_absolute_performance_gain(perf_gain_res_over_naive, 'Tested', 'naive', 'predictive model')
    plot_relative_performance_gain(rel_perf_gain_res_over_naive, 'Tested', 'naive', 'predictive model')
    plot_performance_gains(results_naive, results, 'Tested', 'naive', 'predictive model')

    # Plot results (testing set: predictive modelling VS naïve)
    plot_absolute_performance_gain(perf_gain_res_over_naive, 'Tested', 'naive', 'predictive model')
    plot_relative_performance_gain(rel_perf_gain_res_over_naive, 'Tested', 'naive', 'predictive model')
    plot_performance_gains(results_naive, results, 'Tested', 'naive', 'predictive model')

    # Plot results (testing set: predictive modelling VS autotuned)
    plot_absolute_performance_gain(perf_gain_res_over_naive, 'Tested', 'naive', 'predictive model')
    plot_relative_performance_gain(rel_perf_gain_res_over_naive, 'Tested', 'naive', 'predictive model')
    plot_performance_gains(results_naive, results, 'Tested', 'naive', 'predictive model')

    # Plot goodness of choice
    # mnks_to_plot = [(4, 4, 4), (13, 13, 13), (5, 5, 5)]
    # for m, n, k in mnks_to_plot:
    #     plot_choice_goodness()

    ####################################################################################################################
    # Print log
    log += print_and_log('----------------------------------------------------------------------------')
    with open(log_file, 'w') as f:
        f.write(log)


# ===============================================================================
main()

#EOF
