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

import os
import sys
import datetime
import json
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import dask.dataframe as dd
import matplotlib.pyplot as plt
import argparse
from predict_helpers import (
    safe_pickle,
    safe_pickle_load,
    plot_choice_goodness,
    plot_performance_gains,
    plot_scaled_performance_gains,
    plot_absolute_performance_gain,
    plot_relative_performance_gain,
    performance_gain,
)
from kernels.smm_predict import to_tuple, to_string

visual_separator = (
    "\n----------------------------------------------------------------------------"
)


# ===============================================================================
def main(
    datadir,
    destdir,
    algo,
    model_args,
    nrows,
    prefitted_model_folder,
    run_intermediate_evaluation,
):
    """
    This script is part of the workflow for predictive modelling of optimal libsmm_acc parameters.
    For more details, see predict.md

    """
    # ===============================================================================
    # Create folder to store results of this training and start a log
    folder, log_file, log = get_log_folder(prefitted_model_folder, destdir, algo)

    # ===============================================================================
    # Override algorithm option if working on a pre-fitted model, and log program options
    log += print_and_log(visual_separator)
    algo, model_args, nrows, log = dump_or_load_options(
        algo, model_args, prefitted_model_folder, nrows, folder, log
    )

    # ===============================================================================
    # Get maximum and baseline performances
    (
        max_performances,
        max_performances_algo,
        max_performances_ref,
        baseline_performances_algo,
    ) = get_reference_performances(datadir, algo)

    # ===============================================================================
    # Read data
    log += print_and_log(visual_separator)
    X, X_mnk, Y, log, data_nrows = read_data(algo, datadir, nrows, folder, log)

    # ===============================================================================
    # AT THIS POINT, WE MOVE FROM DASK (out-of-memory dataframes) TO PANDAS
    # ===============================================================================
    log += print_and_log("[moving to pandas] Compute X ...")
    X = X.compute()
    log += print_and_log("[moving to pandas] Compute Y ...")
    Y = Y.compute()
    log += print_and_log("[moving to pandas] Compute X_mnk ...")
    X_mnk = X_mnk.compute()
    log += print_and_log("[moving to pandas] Done")

    # ===============================================================================
    # Get or train partial model (i.e. trained on the "training" part of the data, not the entire dataset)
    log += print_and_log(visual_separator)
    if len(prefitted_model_folder) == 0:  # train a model

        log += print_and_log("\nPreparing to fit model...")
        (
            X_train,
            Y_train,
            X_mnk_train,
            X_test,
            Y_test,
            X_mnk_test,
            model_partial,
            log,
        ) = train_model(X, X_mnk, Y, algo, model_args, folder, log)

    else:  # load pre-trained model

        log += print_and_log(
            "\nReading partial pre-fitted partial model from " + prefitted_model_folder
        )
        (
            X_train,
            Y_train,
            X_mnk_train,
            X_test,
            Y_test,
            X_mnk_test,
            model_partial,
            log,
        ) = fetch_pre_trained_model_partial(
            X, X_mnk, Y, model_args, prefitted_model_folder, log
        )

    # ===============================================================================
    # Evaluate partial model
    if model_partial is not None:
        log = evaluate_model(
            model_partial,
            X_train,
            X_mnk_train,
            Y_train,
            X_test,
            X_mnk_test,
            Y_test,
            max_performances_ref,
            max_performances_algo,
            baseline_performances_algo,
            data_nrows,
            log,
            folder,
        )

    # ===============================================================================
    # Refit to the entire dataset
    # Get or train model fit on the entire dataset (i.e. not just on the "training" part of the data)
    model_file = os.path.join(prefitted_model_folder, "feature_tree_refit.p")
    if (
        run_intermediate_evaluation
        or len(prefitted_model_folder) == 0
        or not os.path.exists(model_file)
    ):
        log += print_and_log(visual_separator)
        log += print_and_log("\nRefit to the entire dataset:")
        X = X_train.append(X_test, ignore_index=True)
        X_mnk = X_mnk_train.append(X_mnk_test, ignore_index=True)
        Y = Y_train.append(Y_test, ignore_index=True)
        model_partial.fit(X, Y)
        model = (
            model_partial  # This model is fit on the entire dataset, it is not partial
        )
        results_file = os.path.join(folder, "feature_tree_refit.p")
        safe_pickle([X.columns.values, model], results_file)
    else:
        log += print_and_log(
            "\nReading pre-fitted model from " + prefitted_model_folder
        )
        X, model, log = fetch_pre_trained_model(prefitted_model_folder, X, log)

    # ===============================================================================
    # Evaluate refit-model
    log = evaluate_model(
        model,
        X,
        X_mnk,
        Y,
        None,
        None,
        None,
        max_performances_ref,
        max_performances_algo,
        baseline_performances_algo,
        data_nrows,
        log,
        folder,
    )

    # ===============================================================================
    # Print log
    log += print_and_log(visual_separator)
    with open(log_file, "w") as f:
        f.write(log)


# ===============================================================================
# Model hyperparameters
optimized_hyperparameters = {
    # chosen by hyperparameter optimization. The optimal parameter depends on the GPU, the data ...
    # the values below are the average of the optimal value for the P100 and the V100
    "tiny": {
        "scikit_max_depth": 16,
        "scikit_min_samples_leaf": 2,
        "scikit_min_samples_split": 15,
        "xgboost_max_depth": 12,
        "xgboost_learning_rate": 0.1,
        "xgboost_n_estimators": 100,
    },
    "small": {
        "scikit_max_depth": 16,
        "scikit_min_samples_leaf": 2,
        "scikit_min_samples_split": 15,
        "xgboost_max_depth": 14,
        "xgboost_learning_rate": 0.1,
        "xgboost_n_estimators": 170,
    },
    "medium": {
        "scikit_max_depth": 18,
        "scikit_min_samples_leaf": 2,
        "scikit_min_samples_split": 13,
        "xgboost_max_depth": 14,
        "xgboost_learning_rate": 0.1,
        "xgboost_n_estimators": 140,
    },
    "largeDB1": {
        "scikit_max_depth": 18,
        "scikit_min_samples_leaf": 2,
        "scikit_min_samples_split": 15,
        "xgboost_max_depth": 14,
        "xgboost_learning_rate": 0.1,
        "xgboost_n_estimators": 170,
    },
    "largeDB2": {
        "scikit_max_depth": 18,
        "scikit_min_samples_leaf": 2,
        "scikit_min_samples_split": 15,
        "xgboost_max_depth": 14,
        "xgboost_learning_rate": 0.1,
        "xgboost_n_estimators": 170,
    },
}


# ===============================================================================
# Printing and dumping helpers
def get_log_folder(prefitted_model_folder, destination_folder, algo):
    """Create a unique log folder for this run in which logs, plots etc. will be stored """
    if len(prefitted_model_folder) == 0:

        # Create a new folder for this model
        file_signature = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")
        folder_name = os.path.join(
            "model_selection", os.path.join(algo, file_signature)
        )
        if destination_folder != ".":
            folder = os.path.join(destination_folder, folder_name)
        else:
            folder = folder_name
        log_file = os.path.join(folder, "log.txt")
        if not os.path.exists(folder):
            while True:  # loop until we've created a folder
                try:
                    os.makedirs(folder)
                    break
                except FileExistsError:
                    time_stamp_seconds = datetime.datetime.now().strftime("-%S")
                    new_folder = folder + time_stamp_seconds
                    print(
                        "Folder {} exists already. Trying to create folder {}.".format(
                            folder, new_folder
                        )
                    )
                    folder = new_folder

    else:

        # If loading a pre-fitted model, use this pre-fitted model's folder as a log folder, but create a new log file
        folder = prefitted_model_folder
        log_file_signature = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")
        log_file = os.path.join(folder, "log_" + log_file_signature + ".txt")

    # Log folder and file
    log = ""
    log += print_and_log("\nLogging to:")
    log += print_and_log("\t" + folder)
    log += print_and_log("\t" + log_file)

    return folder, log_file, log


def dump_or_load_options(algo, model_args, prefitted_model, nrows, folder, log):

    options_file_name = os.path.join(folder, "options.json")
    pgm_options = {"folder": folder, "algo": algo, "nrows": nrows}
    pgm_options.update(model_args)

    if len(prefitted_model) == 0:
        # if we're training a model, dump options to folder so they can be reloaded in another run
        print("Dump options to", options_file_name)
        with open(options_file_name, "w") as f:
            json.dump(pgm_options, f)

    else:
        # if we're using a pre-fitted model, load options from that model
        print("Read options from", options_file_name)
        with open(options_file_name, "r") as f:
            pgm_options = json.load(f)

        algo = pgm_options["algo"]
        model_args_list = ["model", "splits", "ntrees", "njobs"]
        model_args = dict()
        for m in model_args_list:
            model_args[m] = pgm_options[m]
        nrows = pgm_options["nrows"]

    # Log options
    log += print_and_log("Predict-train running with options:")
    for opt, opt_val in pgm_options.items():
        log += print_and_log("{:<15}: {}".format(opt, opt_val))

    return algo, model_args, nrows, log


def print_and_log(msg):
    if not isinstance(msg, str):
        msg = str(msg)
    log = "\n" + msg
    print(msg)
    return log


def dask_to_pandas(*dfs):
    """Convert training data dask -> pandas"""
    pd_dfs = [df.compute() for df in dfs]
    return pd_dfs[0] if len(pd_dfs) == 1 else pd_dfs


def pandas_to_dask(*dfs):
    """Convert training data pandas -> dask"""
    dd_dfs = [dd.from_pandas(df, npartitions=3) for df in dfs]
    return dd_dfs[0] if len(dd_dfs) == 1 else dd_dfs


# ===============================================================================
# Custom loss functions and scorers
def perf_loss(y_true, y_pred, top_k, X_mnk, scaled=True):
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
    mnks = np.unique(X_mnk["mnk"].values)
    for mnk in mnks:

        # Get performances per mnk
        idx_mnk = np.where(X_mnk == mnk)[0].tolist()
        assert len(idx_mnk) > 0, "idx_mnk is empty"
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
        if not scaled:
            maxperf = float(y_true_mnk.max(axis=0))
            assert maxperf >= 0, "Found non-positive value for maxperf: " + str(maxperf)
            perf_loss = (maxperf - maxperf_chosen) / maxperf
        else:
            perf_loss = 1.0 - maxperf_chosen

        # Relative performance loss incurred by using model-predicted parameters instead of autotuned ones [%]
        perf_losses.append(100 * perf_loss)

    return perf_losses


def worse_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk, scaled=True):
    y = np.array(perf_loss(y_true, y_pred, top_k, X_mnk, scaled))
    return float(y.max(axis=0))


def mean_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk, scaled=True):
    y = np.array(perf_loss(y_true, y_pred, top_k, X_mnk, scaled))
    return float(y.mean(axis=0))


def worse_case_scorer(estimator, X, y, top_k):
    """
    :param estimator: the model that should be evaluated
    :param X: validation data
    :param y: ground truth target for X
    :return: score: a floating point number that quantifies the estimator prediction quality on X, with reference to y
    """
    mnk = dd.DataFrame()
    mnk["mnk"] = X["mnk"].copy()
    y_pred = estimator.predict(X.drop(["mnk"].values, axis=1))
    score = worse_rel_perf_loss_of_k(y, y_pred, top_k, mnk)
    return (
        -score
    )  # by scikit-learn convention, higher numbers are better, so the value should be negated


def worse_case_scorer_top1(estimator, X, y):
    return worse_case_scorer(estimator, X, y, 1)


def mean_scorer(estimator, X, y, top_k):
    """
    :param estimator: the model that should be evaluated
    :param X: validation data
    :param y: ground truth target for X
    :return: score: a floating point number that quantifies the estimator prediction quality on X, with reference to y
    """
    mnk = dd.DataFrame()
    mnk["mnk"] = X["mnk"].copy()
    y_pred = estimator.predict(X.drop(["mnk"].values, axis=1))
    score = mean_rel_perf_loss_of_k(y, y_pred, top_k, mnk)
    return (
        -score
    )  # by scikit-learn convention, higher numbers are better, so the value should be negated


def mean_scorer_top1(estimator, X, y):
    return mean_scorer(estimator, X, y, 1)


# ===============================================================================
# Read and prepare data
def get_reference_performances(folder, algo):
    import json

    maxperf_file = os.path.join(folder, "max_performances.json")
    with open(maxperf_file) as f:
        max_performances = json.load(f)

    maxperf_file = os.path.join(folder, "max_performances_by_algo.json")
    with open(maxperf_file) as f:
        max_performances_algo = json.load(f)[algo]

    max_performances_ref = max_performances

    baseline_file = os.path.join(folder, "baseline_performances_by_algo.json")
    with open(baseline_file) as f:
        baseline_performances_algo = json.load(f)[algo]

    return (
        max_performances,
        max_performances_algo,
        max_performances_ref,
        baseline_performances_algo,
    )


def read_data(algo, read_from, nrows, folder, log):

    parquet_data_file = os.path.join(read_from, "training_data_" + algo + ".parquet")
    log += print_and_log("\nRead data from " + parquet_data_file)

    # ===============================================================================
    # Get 'X'
    cols_to_ignore = [
        "perf_scaled",
        "mnk",
        "perf (Gflop/s)",
        "perf_scaled_by_algo",
        "perf_squared",
    ]
    X = dd.read_parquet(parquet_data_file)
    cols_to_drop = set(cols_to_ignore).intersection(set(X.columns.values))
    log += print_and_log("\nDropping following columns from X:\n" + str(cols_to_drop))
    X = X.drop(cols_to_drop, axis=1)
    log += print_and_log(
        "X    : {:>8,} x {:>8,} ({:>2.2} MB)".format(
            len(X), len(X.columns), sys.getsizeof(X) / 10 ** 6
        )
    )
    log += print_and_log("Head:")
    log += print_and_log(X.head())
    n_features = len(list(X.columns))
    predictor_names = X.columns.values
    log += print_and_log("\nPredictor variables: (" + str(n_features) + ")")
    for i, p in enumerate(predictor_names):
        log += print_and_log("\t{:2}) {}".format(i + 1, p))

    # ===============================================================================
    # Get 'Y'
    log += print_and_log("\nRead Y")
    Y = dd.read_parquet(parquet_data_file, columns=["perf_scaled"])
    log += print_and_log(
        "Y    : {:>8,} ({:>2.2} MB)".format(len(Y), sys.getsizeof(Y) / 10 ** 6)
    )
    log += print_and_log("Head:")
    log += print_and_log(Y.head())

    # ===============================================================================
    # Get 'X_mnk'
    log += print_and_log("\nRead X_mnk")
    X_mnk = dd.read_parquet(parquet_data_file, columns=["mnk"])
    nrows_data = len(X_mnk.index)
    log += print_and_log(
        "X_mnk : {:>8,} ({:>2.2} MB)".format(nrows_data, sys.getsizeof(X_mnk) / 10 ** 6)
    )
    log += print_and_log("Head:")
    log += print_and_log(X_mnk.head())
    log += print_and_log("# unique mnks:")
    log += print_and_log(str(X_mnk["mnk"].nunique().compute()) + "\n")

    return X, X_mnk, Y, log, nrows_data


# ===============================================================================
# Predictive modelling
def get_hyperparameter_grid(algo, model_name, n_features):
    # Hyper-parameters to optimize
    param_grid = dict()
    if "scikit" in model_name:  # it is a scikit-learn model
        if algo == "medium":
            max_depth = [10, 13, 16, 18, 21, 24]
            min_samples_split = [2, 8, 12, 18]
            min_samples_leaf = [2, 8, 12, 18]
        elif algo == "tiny":
            step = 1
            max_depth = range(4, int(2 * n_features) + 1, step)
            min_samples_split = range(1, 26, step)
            min_samples_leaf = range(1, 26, step)
        elif algo == "small":
            step = 3
            max_depth = range(4, int(2 * n_features) + 1, step)
            min_samples_split = [2, 5, 8, 13, 18]
            min_samples_leaf = [2, 5, 8, 13, 18]
        else:  # largeDB1,2
            step = 3
            max_depth = range(4, int(2 * n_features) + 1, step)
            min_samples_split = range(2, 21, step)
            min_samples_leaf = range(2, 21, step)
        param_grid = {
            model_name + "__estimator__" + "max_depth": list(max_depth),
            model_name + "__estimator__" + "min_samples_split": list(min_samples_split),
            model_name + "__estimator__" + "min_samples_leaf": list(min_samples_leaf),
        }
    elif "xgb" in model_name:  # it is an XGBOOST model
        if algo == "medium":
            max_depth = [16, 13]
            n_estimators = [100, 140]
            learning_rate = [0.1]
        elif algo == "tiny":
            max_depth = range(10, n_features + 2, 1)
            n_estimators = range(30, 160, 20)
            learning_rate = range(1, 5)
            learning_rate = [i / 10 for i in learning_rate]
        elif algo == "small":
            max_max_depth = 20
            max_depth = range(10, min(max_max_depth, n_features + 2), 4)
            n_estimators = range(50, 200, 30)
            learning_rate = [0.1, 0.3]
        else:  # largeDB1,2
            max_max_depth = 20
            max_depth = range(10, min(max_max_depth, n_features + 2), 4)
            n_estimators = range(50, 200, 30)
            learning_rate = [0.1, 0.3]
        param_grid = {
            "max_depth": list(max_depth),
            "learning_rate": list(learning_rate),
            "n_estimators": list(n_estimators),
        }
    else:
        assert False, "Cannot recognize model: " + model_name

    return param_grid


def get_scikit_DecisionTree_model(algo):
    from sklearn.tree import DecisionTreeRegressor

    model = DecisionTreeRegressor(
        criterion="mse",
        splitter="best",
        min_samples_split=optimized_hyperparameters[algo]["scikit_min_samples_split"],
        min_samples_leaf=optimized_hyperparameters[algo]["scikit_min_samples_leaf"],
        max_depth=optimized_hyperparameters[algo]["scikit_max_depth"],
        max_features=None,
        max_leaf_nodes=None,
    )
    # Feature selection through permutation importance
    from eli5.sklearn import PermutationImportance

    model_perm = PermutationImportance(model, cv=None)
    return model_perm, "scikit-Decision_Tree"


def get_scikit_RandomForest_model(algo, njobs, ntrees):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        criterion="mse",
        n_estimators=ntrees,
        min_samples_split=optimized_hyperparameters[algo]["scikit_min_samples_split"],
        min_samples_leaf=optimized_hyperparameters[algo]["scikit_min_samples_leaf"],
        max_depth=optimized_hyperparameters[algo]["scikit_max_depth"],
        bootstrap=True,
        max_features="sqrt",
        n_jobs=njobs,
    )
    return model, "scikit-Random_Forest"


def get_xgb_DecisionTree_model(algo, njobs, ntrees):
    params = {
        "max_depth": optimized_hyperparameters[algo]["xgboost_max_depth"],
        "learning_rate": optimized_hyperparameters[algo]["xgboost_learning_rate"],
        "n_estimators": optimized_hyperparameters[algo]["xgboost_n_estimators"],
        "tree_method": "exact",
        "verbosity": 2,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "n_jobs": njobs,
    }
    model = xgb.XGBRegressor(**params)
    return model, "xgb-Decision_Tree"


def get_xgb_DecisionTree_dask_model(algo, njobs, ntrees):
    params = {
        "max_depth": optimized_hyperparameters[algo]["xgboost_max_depth"],
        "learning_rate": optimized_hyperparameters[algo]["xgboost_learning_rate"],
        "n_estimators": optimized_hyperparameters[algo]["xgboost_n_estimators"],
        "tree_method": "exact",
        "verbosity": 2,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "n_jobs": njobs,
    }
    from dask_ml.xgboost import XGBRegressor_dask

    model = XGBRegressor_dask(**params)
    return model, "xgb-Decision_Tree_dask"


def get_xgb_DecisionTree_GPU_model(algo, njobs, ntrees):
    params = {
        "max_depth": optimized_hyperparameters[algo]["xgboost_max_depth"],
        "learning_rate": optimized_hyperparameters[algo]["xgboost_learning_rate"],
        "n_estimators": optimized_hyperparameters[algo]["xgboost_n_estimators"],
        "tree_method": "gpu_hist",
        "verbosity": 2,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "n_jobs": njobs,
    }
    model = xgb.XGBRegressor(**params)
    return model, "xgb-Decision_Tree_GPU"


def get_xgb_RandomForest_model(algo, njobs, ntrees):
    params = {
        "max_depth": optimized_hyperparameters[algo]["xgboost_max_depth"],
        "learning_rate": optimized_hyperparameters[algo]["xgboost_learning_rate"],
        "n_estimators": optimized_hyperparameters[algo]["xgboost_n_estimators"],
        "tree_method": "exact",
        "nthread": njobs,
        "subsample": 0.5,
        "colsample_bynode": 0.8,
        "num_parallel_tree": ntrees,
        "verbosity": 2,
        "objective": "reg:squarederror",
    }
    model = xgb.XGBRFRegressor(**params)
    return model, "xgb-Random_Forest"


def get_model(model_to_train, algo, njobs, ntrees):
    if model_to_train == "DT":
        model, model_name = get_scikit_DecisionTree_model(algo)
    elif model_to_train == "RF":
        model, model_name = get_scikit_RandomForest_model(algo, njobs, ntrees)
    elif model_to_train == "xgb-DT":
        model, model_name = get_xgb_DecisionTree_model(algo, njobs, ntrees)
    elif model_to_train == "xgb-DT-dask":
        model, model_name = get_xgb_DecisionTree_dask_model(algo, njobs, ntrees)
    elif model_to_train == "xgb-DT-GPU":
        model, model_name = get_xgb_DecisionTree_GPU_model(algo, njobs, ntrees)
    elif model_to_train == "xgb-RF":
        model, model_name = get_xgb_RandomForest_model(algo, njobs, ntrees)
    else:
        assert False, "Cannot recognize model: " + model_to_train + ". Options: DT, RF"
    return model, model_name


def get_train_test_partition(to_partition, test, train=None):
    """
    Perform train/test partition
    :param to_partition: sequence of objects to partition
    :param test: ndarray, test-indices
    :param train (optional): ndarray
    :return:
    """
    if train is None:  # Retrieve training indices
        all_indices = set(range(len(to_partition[0].index)))
        train = list(all_indices - set(test))

    print(
        "About to partition into train (len: {:,}) / test (len: {:,})".format(
            len(train), len(test)
        )
    )
    partitioned = list()
    for df in to_partition:
        df_train = df.iloc[
            train, :
        ]  # train: use for hyper-parameter optimization (via CV) and training
        partitioned.append(df_train)
        df_test = df.iloc[
            test, :
        ]  # test : use for evaluation of 'selected/final' model
        partitioned.append(df_test)

    print("Returning object of length: {}".format(len(partitioned)))
    return partitioned


def train_model(X, X_mnk, Y, algo, model_options, folder, log):

    # ===============================================================================
    # Get options
    results_file = os.path.join(folder, "feature_tree.p")

    # ===============================================================================
    # Testing splitter (train/test-split)
    from sklearn.model_selection import GroupShuffleSplit

    cv = GroupShuffleSplit(n_splits=2, test_size=0.2)
    train_test_splits = cv.split(X, Y, groups=X_mnk["mnk"])
    train, test = next(train_test_splits)
    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        X_mnk_train,
        X_mnk_test,
    ) = get_train_test_partition([X, Y, X_mnk], test, train)
    plot_train_test_partition(test, train, X_mnk, folder)
    log += print_and_log(
        "\nComplete train/test split, total size="
        + str(X.shape)
        + ", test size="
        + str(X_test.shape)
        + ", train_size="
        + str(X_train.shape)
    )
    del X, X_mnk, Y  # free memory
    predictor_names = X_train.columns.values

    # ===============================================================================
    # Predictive model
    model_to_train = model_options["model"]
    model, model_name = get_model(
        model_to_train, algo, model_options["njobs"], model_options["ntrees"]
    )
    log += print_and_log(
        "\nStart tune/train for model " + model_name + " with parameters:"
    )
    log += print_and_log(model)

    # ===============================================================================
    # Cross-validation splitter (train/validation-split)
    test_size = 0.3
    cv = GroupShuffleSplit(n_splits=model_options["splits"], test_size=test_size)

    # ===============================================================================
    # Feature selection: SelectFromModel
    from sklearn.feature_selection import SelectFromModel

    feature_importance_threshold = (
        0.0005  # only remove the features with VERY little importance
    )
    model.cv = cv.split(X_train.values, Y_train.values, groups=X_mnk_train.values)
    model.fit(X_train.values, Y_train.values)
    model_fs = SelectFromModel(
        model, threshold=feature_importance_threshold, max_features=None, prefit=True
    )
    print(model_fs)
    model.cv = None

    # ===============================================================================
    # Info on feature selection
    all_feature_names = X_train.columns.values.tolist()
    feature_support = model_fs.get_support()
    features_importances = model.feature_importances_
    feature_name_importance = zip(
        all_feature_names, features_importances, feature_support
    )
    feature_name_importance = sorted(
        feature_name_importance, key=lambda x: x[1], reverse=True
    )

    log += print_and_log(visual_separator)
    n_selected_features = np.sum(feature_support)
    log += print_and_log("Optimal number of features : {}".format(n_selected_features))

    # Selected features
    log += print_and_log("\nFeatures:")
    selected_features = list()
    selected_feature_importances = list()
    for i, (feat_name, feat_imp, feat_in) in enumerate(feature_name_importance):
        in_or_out = "accepted" if feat_in else " x rejected"
        log += print_and_log(
            "{:>2}) {:<40}, imp: {:>1.3f} {}".format(
                i + 1, feat_name, feat_imp, in_or_out
            )
        )
        if feat_in:
            selected_features.append(feat_name)
            selected_feature_importances.append(feat_imp)
    plot_feature_importance(features_importances, all_feature_names, folder)

    # Drop non-selected features
    features_to_drop = [f for f in predictor_names if f not in selected_features]
    X_train = X_train.drop(features_to_drop, axis=1)
    X_test = X_test.drop(features_to_drop, axis=1)
    n_features = len(X_train.columns)

    # ===============================================================================
    # Fit
    out_of_memory_computation = "dask" in model_options["model"]
    if out_of_memory_computation:
        X_train, Y_train = pandas_to_dask(X_train, Y_train)

    if model_options["hyperparameter_optimization"]:

        # Hyperparameter Optimization
        param_grid = get_hyperparameter_grid(algo, model_name, n_features)
        if param_grid is None:
            assert False, "param_grid object is None. Please implement!"

        # At this point, we "cheat"/"take a shortcut" in 2 ways:
        # - we split into train/test partitions using the simple default splitter, not one that is aware of mnk-groups
        # - we use an overall MSE scorer, not one that looks at the performance loss of predicted mnks wrt. autotuned
        if out_of_memory_computation:
            from dask_ml.model_selection import GridSearchCV

            gds_pars = {
                "estimator": model,
                "param_grid": param_grid,
                "cv": model_options["splits"],
                "refit": True,
                "n_jobs": 1,
            }
        else:
            from sklearn.model_selection import GridSearchCV

            gds_pars = {
                "estimator": model,
                "param_grid": param_grid,
                "cv": model_options["splits"],
                "refit": True,
                "n_jobs": 1,
                "verbose": 2,
            }
        gds = GridSearchCV(**gds_pars)
        log += print_and_log(visual_separator)
        log += print_and_log("\nStart hyperparameter optimization & training ... :\n")
        log += print_and_log("Hyper-parameter grid:")
        for par, values in param_grid.items():
            log += print_and_log("\t" + par + ": " + str(values))
        log += print_and_log("\n")
        gds.fit(X_train.values, Y_train.values)
        log += print_and_log("... done")
        describe_hpo(gds, log, folder)
        model = gds.best_estimator_

    else:

        # Fit
        log += print_and_log(visual_separator)
        log += print_and_log("\nStart fitting model with predictors:\n")
        for i, p in enumerate(X_train.columns.values):
            log += print_and_log("\t{:>2}) {}".format(i + 1, p))

        model.fit(X_train, Y_train)

    safe_pickle([X_train.columns.values, model, test], results_file)
    log += print_and_log("\nCompleted fit, wrote results to " + results_file)
    log += print_and_log(visual_separator)
    return_model = model

    # Return
    if "mnk" in X_train.columns.values:
        X_train.drop("mnk", axis=1, inplace=True)
    if "mnk" in X_test.columns.values:
        X_train.drop("mnk", axis=1, inplace=True)

    if out_of_memory_computation:
        X_train, Y_train = dask_to_pandas(X_train, Y_train)

    return X_train, Y_train, X_mnk_train, X_test, Y_test, X_mnk_test, return_model, log


def fetch_pre_trained_model(model_path_folder, X, log):
    model_path = os.path.join(model_path_folder, "feature_tree_refit.p")
    print("fetched pre-trained model from: {}".format(model_path))
    features, model = safe_pickle_load(model_path)
    print("Pickled variables:\nfeatures:{}\nmodel:{}".format(features, model))

    log += print_and_log("\nDrop non-selected features")
    predictor_names = X.columns.values.tolist()
    features_to_drop = [f for f in predictor_names if f not in features]
    X.drop(features_to_drop, axis=1, inplace=True)
    return X, model, log


def fetch_pre_trained_model_partial(X, X_mnk, Y, model_options, model_path_folder, log):

    # Load pre-trained model, selected features and indices of test-set
    model_path = os.path.join(model_path_folder, "feature_tree.p")
    print("fetched partial pre-trained model from: {}".format(model_path))
    features, model, test_indices = safe_pickle_load(model_path)
    print(
        "Pickled stuff:\nfeatures:{}\nmodel:{}\ntest_indices:{}".format(
            features, model, test_indices
        )
    )
    if "mnk" in features:
        features.remove("mnk")

    log += print_and_log("\nPerform train/test split")
    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        X_mnk_train,
        X_mnk_test,
    ) = get_train_test_partition([X, Y, X_mnk], test_indices)
    log += print_and_log(
        "\nComplete train/test split, total size="
        + str(X.shape)
        + ", test size="
        + str(X_test.shape)
        + ", train_size="
        + str(X_train.shape)
    )

    log += print_and_log("\nDrop non-selected features")
    predictor_names = X_train.columns.values.tolist()
    features_to_drop = [f for f in predictor_names if f not in features]
    X_train.drop(features_to_drop, axis=1, inplace=True)
    X_test.drop(features_to_drop, axis=1, inplace=True)

    out_of_memory_computation = "dask" in model_options["model"]
    if out_of_memory_computation:
        X_train, Y_train = pandas_to_dask(X_train, Y_train)

    return X_train, Y_train, X_mnk_train, X_test, Y_test, X_mnk_test, model, log


# ===============================================================================
# Describe and evaluate model
def describe_hpo(gs, log, folder):

    # Scores obtained during hyperparameter optimization
    columns_to_print = list()
    for par in gs.param_grid.keys():
        columns_to_print.append("param_" + par)
    columns_to_print += [
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
    ]
    log += print_and_log("\nHyperparameter search results (head):")
    cv_results = pd.DataFrame(gs.cv_results_)[columns_to_print]
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        log += print_and_log(cv_results.head())
    cv_results_path = os.path.join(folder, "hyperparameter_optimization_results.csv")
    with open(cv_results_path, "w") as f:
        cv_results.to_csv(f, index=False)
    log += print_and_log("Wrote hyperparameter results to " + cv_results_path)

    # Best parameter set
    log += print_and_log("\nBest parameters set found on development set:")
    for bestpar_name, bestpar_value in gs.best_params_.items():
        log += print_and_log("\t{}: {}".format(bestpar_name, bestpar_value))

    # Best estimator
    log += print_and_log("\nBest estimator:")
    best_estimator = gs.best_estimator_
    log += print_and_log(best_estimator)
    log += print_and_log(visual_separator)

    return log


def describe_model(model, X, Y, log):
    predictor_names = X.columns.values.tolist()
    log += print_and_log("Model:")
    log += print_and_log(model)

    log += print_and_log("Predictor variables:")
    for p in predictor_names:
        log += print_and_log("\t{}".format(p))

    return log


def print_custom_error(y_true, y_pred, X_mnk, log, scaled=True):
    result_line = (
        "\tRelative performance loss compared to autotuned max:\n"
        + "top-{}: worse: {:>6.3f} [%], mean: {:>6.3f} [%]"
    )
    for top_k in [1]:
        log += print_and_log(
            result_line.format(
                top_k,
                worse_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk, scaled),
                mean_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk, scaled),
            )
        )
    return log


def print_error(y_true, y_pred, log):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    result_line = "\tOverall error:\n" + "absolute: {:>6.3f}, mean squared {:>6.3f}"
    log += print_and_log(
        result_line.format(
            mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred)
        )
    )
    return log


def scale_back(y_scaled, x_mnk, max_performances, mnk=None):
    if mnk is None:
        corresponding_maxperf = np.array(
            [max_performances[mnk] for mnk in x_mnk["mnk"].values.tolist()]
        )
    else:
        corresponding_maxperf = max_performances[mnk]
    return y_scaled * corresponding_maxperf


def plot_train_test_partition(test_idx, train_idx, X_mnk, folder):

    import matplotlib.pyplot as plt

    mnks_string_train = X_mnk["mnk"].iloc[train_idx].unique()
    mnks_train = to_tuple(*mnks_string_train)
    mnks_string_test = X_mnk["mnk"].iloc[test_idx].unique()
    mnks_test = to_tuple(*mnks_string_test)

    y_train_product = (
        dict()
    )  # keys: m*n*k, values: how many times this mnk-product appears in training-mnks
    for m, n, k in mnks_train:
        mxnxk = m * n * k
        if mxnxk in y_train_product.keys():
            y_train_product[mxnxk] += 1
        else:
            y_train_product[mxnxk] = 1

    train_mnks = list()
    train_counts = list()
    for mnk, count in y_train_product.items():
        for c in range(count):
            train_mnks.append(mnk)
            train_counts.append(c + 1)

    y_test_product = dict()
    for m, n, k in mnks_test:
        mxnxk = m * n * k
        if mxnxk in y_test_product.keys():
            y_test_product[mxnxk] += 1
        else:
            y_test_product[mxnxk] = 1

    test_mnks = list()
    test_counts = list()
    for mnk, count in y_test_product.items():
        for c in range(count):
            test_mnks.append(mnk)
            if mnk in y_train_product.keys():
                test_counts.append(y_train_product[mnk] + c + 1)
            else:
                test_counts.append(c + 1)

    plt.figure(figsize=(30, 5))
    markersize = 12
    plt.plot(
        train_mnks,
        train_counts,
        "o",
        markersize=markersize,
        color="blue",
        label="training mnks (" + str(len(train_mnks)) + ")",
    )
    plt.plot(
        test_mnks,
        test_counts,
        "o",
        markersize=markersize,
        color="red",
        label="testing mnks (" + str(len(test_mnks)) + ")",
    )
    plot_file_path = os.path.join(folder, "train-test_split.svg")
    plt.xlabel("m * n * k triplets")
    plt.ylabel("number of occurences in data set")
    plt.title("Train/test split")
    maxcount = max(max(test_counts), max(train_counts)) + 1
    plt.ylim([0, maxcount])
    plt.legend()
    plt.savefig(plot_file_path)


def plot_feature_importance(importances, names, folder):

    plt.rcdefaults()
    fig, ax = plt.subplots()

    ax.set_title("Feature importances")
    ax.barh(range(len(names)), importances, color="g", align="center")
    ax.set_yticks(np.arange(len(importances)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    plot_file_path = os.path.join(folder, "feature_importance.svg")
    plt.savefig(plot_file_path)
    print(plot_file_path)


def plot_loss_histogram(y_true, y_pred, X_mnk, folder):
    import matplotlib.pyplot as plt

    # Get losses
    top_k = 1
    y = np.array(perf_loss(y_true, y_pred, top_k, X_mnk, False))

    # Losses-histogram
    num_bins = 100
    plt.figure()
    plt.hist(y, num_bins, facecolor="green", alpha=0.75)
    plt.xlabel("relative performance loss [%]")
    plt.ylabel("# occurrences")
    plt.title(
        "Performance losses for top-k="
        + str(top_k)
        + " ("
        + str(len(y))
        + " test mnks)"
    )
    plot_file_path = os.path.join(folder, "result_losses.svg")
    plt.savefig(plot_file_path)
    print(plot_file_path)


def plot_prediction_accuracy(m, n, k, y_true, y_pred, train, pp):

    plt.figure()
    if train:
        plt.plot(100 * y_true, 100 * y_pred, "b.", label="truth")
    else:
        plt.plot(100 * y_true, 100 * y_pred, "r.", label="truth")
    plt.xlabel("true scaled performance [%]")
    plt.ylabel("predicted scaled performance [%]")
    type = "train" if train else "test"
    plt.title("Prediction accuracy for kernel " + str((m, n, k)) + " (" + type + ")")
    pp.savefig()


def get_predive_model_performances(
    y_true, y_pred, x_mnk, max_performances_ref, max_performances_algo
):

    predictive_model_perf_scaled = dict()

    for mnk_string in x_mnk["mnk"].unique():

        idx_mnk = np.where(x_mnk == mnk_string)[0].tolist()
        assert len(idx_mnk) > 0, "idx_mnk is empty"
        m, n, k = to_tuple(mnk_string)

        perf_chosen_idx = [np.argmax(y_pred[idx_mnk])]
        perf_effective = y_true.iloc[idx_mnk].iloc[perf_chosen_idx].values.item()
        predictive_model_perf_scaled[
            (m, n, k)
        ] = perf_effective  # 'scaled' between 0 and 1

    predictive_model_perf = dict(
        zip(
            predictive_model_perf_scaled.keys(),
            [
                perf_scaled * max_performances_ref[to_string(mnk)]
                for mnk, perf_scaled in predictive_model_perf_scaled.items()
            ],
        )
    )

    # Re-scale performances by algorithm for a fair comparison
    predictive_model_perf_scaled = dict(
        zip(
            predictive_model_perf.keys(),
            [
                perf / max_performances_algo[mnk]
                for mnk, perf in predictive_model_perf.items()
            ],
        )
    )

    return predictive_model_perf, predictive_model_perf_scaled


# ===============================================================================
def evaluate_model(
    model,
    X_train,
    X_mnk_train,
    Y_train,
    X_test,
    X_mnk_test,
    Y_test,
    max_performances_ref,
    max_performances_algo,
    baseline_performances_algo,
    data_nrows,
    log,
    folder,
):
    """Main evaluation function"""
    if model is None:
        return log

    # Start evaluation
    log += print_and_log(visual_separator)
    log += print_and_log("Start model evaluation")
    if all([x is not None for x in [X_test, Y_test]]):
        log = describe_model(model, X_test, Y_test, log)

    # Training error
    if all([x is not None for x in [X_train, X_mnk_train, Y_train]]):
        y_train_pred = model.predict(X_train.values)
        log += print_and_log("\nTraining error: (train&val)")
        log = print_custom_error(Y_train, y_train_pred, X_mnk_train, log, True)
        log = print_error(Y_train, y_train_pred, log)

        # Test error
        if all([x is not None for x in [X_test, X_mnk_test, Y_test]]):
            y_test_pred = model.predict(X_test)
            log += print_and_log("\nTesting error:")
            log = print_custom_error(Y_test, y_test_pred, X_mnk_test, log, True)
            log = print_error(Y_test, y_test_pred, log)

    # Training error (scaled-back)
    if all([x is not None for x in [X_train, X_mnk_train, Y_train]]):
        log += print_and_log("\nTraining error (scaled back): (train&val)")
        y_train_pred_scaled_back = scale_back(
            y_train_pred, X_mnk_train, max_performances_ref
        )
        y_train_scaled_back = pd.DataFrame(
            scale_back(Y_train.values.flatten(), X_mnk_train, max_performances_ref)
        )
        log = print_custom_error(
            y_train_scaled_back, y_train_pred_scaled_back, X_mnk_train, log, False
        )
        log = print_error(y_train_scaled_back, y_train_pred_scaled_back, log)

        if all([x is not None for x in [X_test, X_mnk_test, Y_test]]):
            # Test error (scaled-back)
            log += print_and_log("\nTesting error (scaled back): (test&val)")
            y_test_pred_scaled_back = scale_back(
                y_test_pred, X_mnk_test, max_performances_ref
            )
            y_test_scaled_back = pd.DataFrame(
                scale_back(Y_test.values.flatten(), X_mnk_test, max_performances_ref)
            )
            log = print_custom_error(
                y_test_scaled_back, y_test_pred_scaled_back, X_mnk_test, log, False
            )
            log = print_error(y_test_scaled_back, y_test_pred_scaled_back, log)

    # ===============================================================================
    # Print histogram for "best" estimator
    if all([x is not None for x in [X_test, X_mnk_test, Y_test]]):
        log += print_and_log("\nPlot result histogram:")
        plot_loss_histogram(Y_test, y_test_pred, X_mnk_test, folder)

    # ===============================================================================
    # Plot prediction accuracy and goodness of choice for a few mnks (training-set)
    if all([x is not None for x in [X_train, X_mnk_train, Y_train]]):
        n_samples = 10 if data_nrows < 100000000 else 2
        mnks_to_plot = random.sample(X_mnk_train["mnk"].values.tolist(), n_samples)

        from matplotlib.backends.backend_pdf import PdfPages

        plot_file_path = os.path.join(folder, "evaluation_by_mnk_refit.pdf")
        if all([x is not None for x in [X_test, X_mnk_test, Y_test]]):
            plot_file_path = os.path.join(folder, "evaluation_by_mnk.pdf")
        pp = PdfPages(plot_file_path)

        for mnk_string in mnks_to_plot:

            # Get performances per mnk
            idx_mnk = np.where(X_mnk_train == mnk_string)[0].tolist()
            assert len(idx_mnk) > 0, "idx_mnk is empty"
            m_, n_, k_ = to_tuple(mnk_string)
            y_train_pred_mnk = y_train_pred[idx_mnk]
            Y_train_mnk = Y_train.iloc[idx_mnk]

            log += print_and_log("Prediction accuracy plot: " + str(mnk_string))

            plot_prediction_accuracy(
                m_, n_, k_, Y_train_mnk, y_train_pred_mnk, True, pp
            )

            log += print_and_log("Goodness plot: " + str(mnk_string))
            plot_choice_goodness(
                m_,
                n_,
                k_,
                baseline_performances_algo,
                max_performances_ref,
                Y_train["perf_scaled"].iloc[idx_mnk].values,
                y_train_pred_mnk,
                True,
                pp,
            )

        # ===============================================================================
        # Plot prediction accuracy for a few mnks (testing-set)
        if all([x is not None for x in [X_test, X_mnk_test, Y_test]]):
            mnks_to_plot = random.sample(X_mnk_test["mnk"].values.tolist(), n_samples)
            for mnk_string in mnks_to_plot:

                # Get performances per mnk
                idx_mnk = np.where(X_mnk_test == mnk_string)[0].tolist()
                assert len(idx_mnk) > 0, "idx_mnk is empty"
                m_, n_, k_ = to_tuple(mnk_string)

                log += print_and_log("Prediction accuracy plot: " + str(mnk_string))
                plot_prediction_accuracy(
                    m_, n_, k_, Y_test.iloc[idx_mnk], y_test_pred[idx_mnk], False, pp
                )

                log += print_and_log("Goodness plot: " + str(mnk_string))
                plot_choice_goodness(
                    m_,
                    n_,
                    k_,
                    baseline_performances_algo,
                    max_performances_ref,
                    Y_test["perf_scaled"].iloc[idx_mnk].values,
                    y_test_pred[idx_mnk],
                    False,
                    pp,
                    True,
                )

        if all([x is not None for x in [X_train, X_mnk_train, Y_train]]):
            pp.close()

    # ===============================================================================
    # Scale baseline and max performances
    max_performances_algo = dict(
        zip(
            [to_tuple(mnk_string) for mnk_string in max_performances_algo.keys()],
            max_performances_algo.values(),
        )
    )
    max_performances_algo_scaled = dict(
        zip(max_performances_algo.keys(), [1.0] * len(max_performances_algo))
    )
    baseline_performances_algo = dict(
        zip(
            [to_tuple(mnk_string) for mnk_string in baseline_performances_algo.keys()],
            baseline_performances_algo.values(),
        )
    )
    baseline_performances_algo_scaled = dict(
        zip(
            [(m, n, k) for m, n, k in baseline_performances_algo.keys()],
            [
                perf / max_performances_algo[(m, n, k)]
                for (m, n, k), perf in baseline_performances_algo.items()
            ],
        )
    )

    # ===============================================================================
    # Compare max performances and baseline
    from matplotlib.backends.backend_pdf import PdfPages

    plot_file_path = os.path.join(folder, "evaluation_by_overall_refit.pdf")
    if all([x is not None for x in [X_test, X_mnk_test, Y_test]]):
        plot_file_path = os.path.join(folder, "evaluation_overall.pdf")
    pp = PdfPages(plot_file_path)

    if all([x is not None for x in [X_test, X_mnk_test, Y_test]]):
        plot_performance_gains(
            max_performances_algo,
            baseline_performances_algo,
            "trained",
            "max. performance per algorithm",
            "baseline per algorithm",
            pp,
        )
        plot_scaled_performance_gains(
            max_performances_algo_scaled,
            baseline_performances_algo_scaled,
            "trained",
            "max. performance per algorithm",
            "baseline per algorithm",
            pp,
        )

    # ===============================================================================
    # 'Results' = y_true ( y_chosen )
    if all([x is not None for x in [X_train, X_mnk_train, Y_train]]):
        (
            predictive_model_perf_train,
            predictive_model_perf_train_scaled,
        ) = get_predive_model_performances(
            Y_train,
            y_train_pred,
            X_mnk_train,
            max_performances_ref,
            max_performances_algo,
        )

        if all([x is not None for x in [X_test, X_mnk_test, Y_test]]):
            (
                predictive_model_perf_test,
                predictive_model_perf_test_scaled,
            ) = get_predive_model_performances(
                Y_test,
                y_test_pred,
                X_mnk_test,
                max_performances_ref,
                max_performances_algo,
            )

    # ===============================================================================
    # Plot results (training set: predictive modelling VS naïve)
    log += print_and_log("\nPredictive model VS baseline: ")

    if all([x is not None for x in [X_train, X_mnk_train, Y_train]]):
        perf_gain_pred_train_over_baseline = performance_gain(
            baseline_performances_algo, predictive_model_perf_train
        )
        plot_absolute_performance_gain(
            perf_gain_pred_train_over_baseline,
            "trained",
            "baseline per algorithm",
            "predictive model",
            pp,
        )

        scaled_perf_gain_pred_train_over_baseline = performance_gain(
            baseline_performances_algo_scaled, predictive_model_perf_train_scaled
        )
        plot_relative_performance_gain(
            scaled_perf_gain_pred_train_over_baseline,
            "trained",
            "baseline per algorithm",
            "predictive model",
            pp,
        )

        if all([x is not None for x in [X_test, X_mnk_test, Y_test]]):
            perf_gain_pred_test_over_baseline = performance_gain(
                baseline_performances_algo, predictive_model_perf_test
            )
            plot_absolute_performance_gain(
                perf_gain_pred_test_over_baseline,
                "tested",
                "baseline per algorithm",
                "predictive model",
                pp,
            )

            scaled_perf_gain_pred_test_over_baseline = performance_gain(
                baseline_performances_algo_scaled, predictive_model_perf_test_scaled
            )
            plot_relative_performance_gain(
                scaled_perf_gain_pred_test_over_baseline,
                "tested",
                "baseline per algorithm",
                "predictive model",
                pp,
            )

            log += print_and_log("\nPredictive model VS autotuned: ")
            perf_gain_pred_train_over_max = performance_gain(
                max_performances_algo, predictive_model_perf_train
            )
            plot_absolute_performance_gain(
                perf_gain_pred_train_over_max,
                "trained",
                "max. performance per algorithm",
                "predictive model",
                pp,
            )
            scaled_perf_gain_pred_train_over_max = performance_gain(
                max_performances_algo_scaled, predictive_model_perf_train_scaled
            )
            plot_relative_performance_gain(
                scaled_perf_gain_pred_train_over_max,
                "trained",
                "max. performance per algorithm",
                "predictive model",
                pp,
            )

        if all([x is not None for x in [X_test, X_mnk_test, Y_test]]):
            perf_gain_pred_test_over_max = performance_gain(
                max_performances_algo, predictive_model_perf_test
            )
            plot_absolute_performance_gain(
                perf_gain_pred_test_over_max,
                "tested",
                "max. performance per algorithm",
                "predictive model",
                pp,
            )
            scaled_perf_gain_pred_test_over_max = performance_gain(
                max_performances_algo_scaled, predictive_model_perf_test_scaled
            )
            plot_relative_performance_gain(
                scaled_perf_gain_pred_test_over_max,
                "tested",
                "max. performance per algorithm",
                "predictive model",
                pp,
            )

        if all([x is not None for x in [X_test, X_mnk_test, Y_test]]):
            log += print_and_log("\nCompare performances: ")
            plot_performance_gains(
                baseline_performances_algo,
                predictive_model_perf_train,
                "trained",
                "baseline per algorithm",
                "predictive model",
                pp,
            )
            plot_performance_gains(
                max_performances_algo,
                predictive_model_perf_train,
                "trained",
                "max. performance per algorithm",
                "predictive model",
                pp,
            )

        if all([x is not None for x in [X_test, X_mnk_test, Y_test]]):
            plot_performance_gains(
                baseline_performances_algo,
                predictive_model_perf_test,
                "tested",
                "baseline per algorithm",
                "predictive model",
                pp,
            )
            plot_performance_gains(
                max_performances_algo,
                predictive_model_perf_test,
                "tested",
                "max. performance per algorithm",
                "predictive model",
                pp,
            )

        pp.close()

    return log


# ===============================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""
        Train predictive model on autotuning data

        This script is part of the workflow for predictive modelling of optimal libsmm_acc parameters.
        For more details, see README.md.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--destination_folder",
        metavar="FOLDER",
        type=str,
        default=".",
        help="Folder in which to write plots, models, etc.",
    )
    parser.add_argument(
        "-f",
        "--folder",
        metavar="FOLDER",
        type=str,
        default=".",
        help="Folder from which to read data",
    )
    parser.add_argument(
        "-a", "--algo", metavar="algoname", default="", help="Algorithm to train on"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="DT",
        help="Model to train. Options: DT (Decision Trees), RF (Random Forests), xgb-DT, xgb-DT-dask (out-of-memory"
        + "xgboost), xgb-DT-GPU (with GPU support), xgb-RF",
    )
    parser.add_argument(
        "-o",
        "--hyperparameter_optimization",
        default=False,
        help="Whether to do hyperparameter optimization. If False, the model will be trained with 'best guess' parameters",
    )
    parser.add_argument(
        "-s",
        "--splits",
        default=3,
        metavar="NUMBER",
        type=int,
        help="Number of cross-validation splits used in RFECV and GridSearchCV",
    )
    parser.add_argument(
        "-e",
        "--ntrees",
        default=3,
        metavar="NUMBER",
        type=int,
        help="Number of estimators in RF",
    )
    parser.add_argument(
        "-j",
        "--njobs",
        default=-1,
        metavar="NUMBER",
        type=int,
        help="Number of parallel jobs that Joblib will launch (used by GridSearchCV and XGBoost)",
    )
    parser.add_argument(
        "-r",
        "--nrows",
        default=None,
        metavar="NUMBER",
        type=int,
        help="Number of rows of data to load. Default: None (load all)",
    )
    parser.add_argument(
        "-g",
        "--prefitted_model",
        metavar="filename",
        default="",
        help="Path to pickled model object to load instead of re-training model",
    )
    parser.add_argument(
        "-i",
        "--intermediate_evaluation",
        default=False,
        help="Whether to perform evaluation of the model trained on part of the model",
    )
    parser.set_defaults(intermediate_evaluation=False)

    args = parser.parse_args()
    model_args = {
        "model": args.model,
        "splits": args.splits,
        "ntrees": args.ntrees,
        "njobs": args.njobs,
        "hyperparameter_optimization": args.hyperparameter_optimization,
    }
    main(
        args.folder,
        args.destination_folder,
        args.algo,
        model_args,
        args.nrows,
        args.prefitted_model,
        args.intermediate_evaluation,
    )
