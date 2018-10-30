########################################################################################################################
# DBCSR optimal parameters prediction
# Shoshana Jakobovits
# August-September 2018
########################################################################################################################

import numpy as np
import pandas as pd
from dtreeviz.trees import *
import os
import sys
import pickle
import datetime


########################################################################################################################
# Flags
########################################################################################################################
from absl import flags, app
flags.DEFINE_string('in_folder', 'tune_big/', 'Folder from which to read data')
flags.DEFINE_string('algo', 'tiny', 'Algorithm to train on')
flags.DEFINE_boolean('print_tree', False,
                     'Whether to export the best estimator tree to SVG (Warning: can be very slow for large trees)')
flags.DEFINE_boolean('tune', False, 'Rune recursive feature selection and grid search on hyperparameters')
flags.DEFINE_string('model', 'RF', 'Model to train. Options: DT (Decision Trees), RF (Random Forests)')
#flags.DEFINE_integer('nruns', '10', '#times to run train-test split, variable selection and GridSearch on model')
flags.DEFINE_integer('splits', '5', 'number of cross-validation splits used in RFECV and GridSearchCV')
flags.DEFINE_integer('ntrees', '10', 'number of estimators in RF')
flags.DEFINE_integer('njobs', '-1', 'number of cross-validation splits used in RFECV and GridSearchCV')
flags.DEFINE_integer('nrows', None, 'Number of rows of data to load. Default: None (load all)')
flags.DEFINE_string('gs', '', 'Path to pickled GridSearchCV object to load instead of recomputing')
FLAGS = flags.FLAGS


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
        'mnk'
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
        'mnk'
    ]
}
optimized_hyperparameters = {
    'tiny': {
        'max_depth': 39,
        'min_samples_leaf': 8,
        'min_samples_split': 11
    },
    'small': {
        'max_depth': 12,
        'min_samples_leaf': 2,
        'min_samples_split': 2
    },
    'medium': {  # common sense
        'max_depth': 18,
        'min_samples_leaf': 2,
        'min_samples_split': 2
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
# Formatting and printing helpers
########################################################################################################################
def safe_pickle(data, file):
    """
    Pickle big files safely by processing them in chunks
    :param data: data to be pickled
    :param file: file to pickle it into
    """
    max_bytes = 2**31 - 1  # Maximum number of bytes to pickle in one chunk
    pickle_out = pickle.dumps(data)
    n_bytes = sys.getsizeof(pickle_out)
    with open(file, 'wb') as f:
        count = 0
        for i in range(0, n_bytes, max_bytes):
            f.write(pickle_out[i:min(n_bytes, i + max_bytes)])
            count += 1


def print_and_log(msg, log):
    if not isinstance(msg, str):
        msg = str(msg)
    log += '\n' + msg
    print(msg)
    return log


########################################################################################################################
# Custom loss functions
########################################################################################################################
def _num_samples(x):
    """
    Return number of samples in array-like x.
    TAKEN VERBATIM FROM SKLEARN code !!
    """
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.

    TAKEN VERBATIM FROM SKLEARN code !!
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def perf_loss(y_true, y_pred, top_k, X_mnk):
    """
    Compute the relative performance losses per mnk if one were to
    :param y_true: ground truth
    :param y_pred: estimated performances
    :param top_k: #top performances to consider
    :param X_mnk: corresponding mnks
    :return: perf_losses: array of relative performance losses (in %), one element per mnk
    """
    check_consistent_length(y_true, y_pred, X_mnk)
    y_true = np.sqrt(y_true)
    y_pred = np.sqrt(y_pred)
    perf_losses = list()

    mnks = np.unique(X_mnk['mnk'].values)
    for mnk in mnks:

        # Get performances per mnk
        idx_mnk = np.where(X_mnk == mnk)[0].tolist()
        assert (len(idx_mnk) > 0), "idx_mnk is empty"
        y_true_mnk = y_true.iloc[idx_mnk]
        y_pred_mnk = y_pred[idx_mnk]
        top_k_idx = np.argpartition(-y_pred_mnk, top_k)[:top_k]
        y_correspmax = y_true_mnk.iloc[top_k_idx]

        # Max. performances
        maxperf = float(y_true_mnk.max(axis=0))  # true max. performance
        assert maxperf >= 0, "Found non-positive value for maxperf: " + str(maxperf)
        maxperf_chosen = np.amax(y_correspmax)  # chosen max perf. among predicted max performances

        # perf. loss incurred by using model-predicted parameters instead of autotuned ones
        perf_loss = 100 * (maxperf - maxperf_chosen) / maxperf
        perf_losses.append(perf_loss)

    return perf_losses


def worse_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk):
    y = np.array(perf_loss(y_true, y_pred, top_k, X_mnk))
    return float(y.max(axis=0))


def mean_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk):
    y = np.array(perf_loss(y_true, y_pred, top_k, X_mnk))
    return float(y.mean(axis=0))


########################################################################################################################
# Custom Scorers
# Ref: http://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object
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
# Tune and train predictive model
########################################################################################################################
def perf_Kothapalli(N_mem, nblks, threads_per_blk, Gflops):
    c_K = nblks * threads_per_blk * N_mem  # ignore number of threads per warp
    return Gflops / c_K # ignore clock rate


def add_Kothapalli(df, gpu, Nmem_glob, Nmem_shared, Nmem, perf_K):
    df[perf_K] = np.vectorize(perf_Kothapalli)(
        gpu['Global memory access latency'] * df[Nmem_glob] + gpu['Shared memory access latency'] * df[Nmem_shared],
        df['nblks'], df['threads_per_blk'], df['Gflops'])


def create_log_folder():
    file_signature = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")
    folder = os.path.join("model_selection", os.path.join(FLAGS.algo, file_signature))
    log_file = os.path.join(folder, "log.txt")
    if not os.path.exists(folder):
        os.makedirs(folder)
    return log_file, folder


def read_data():

    # Create folder to store results of this training
    log = ''
    log_file, folder_ = create_log_folder()

    read_from = FLAGS.in_folder
    log = print_and_log('Read training data X ...', log)
    X = pd.read_csv(os.path.join(read_from, 'train_all_' + FLAGS.algo + '_X.csv'), index_col=0)
    log = print_and_log('X    : {:>8} x {:>8} ({:>2} MB)'.format(X.shape[0], X.shape[1], sys.getsizeof(X)/10**6), log)

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

    log = print_and_log('Read training data Y ...', log)
    Y = pd.read_csv(os.path.join(read_from, 'train_all_' + FLAGS.algo + '_Y.csv'), index_col=0, nrows=FLAGS.nrows)
    log = print_and_log('Y    : {:>8} x {:>8} ({:>2} MB)'.format(Y.shape[0], Y.shape[1], sys.getsizeof(Y)/10**6), log)
    if 'perf_squared' not in Y.columns.values:
        assert 'perf' in Y.columns.values, "Y has column names:" + str(*Y.columns.values)
        Y['perf'] = Y['perf'] * Y['perf']
        Y.rename(columns={'perf': 'perf_squared'}, inplace=True)


    log = print_and_log('Read training data X_mnk ...', log)
    X_mnk = pd.read_csv(os.path.join(read_from, 'train_all_' + FLAGS.algo + '_X_mnk.csv'), index_col=0, nrows=FLAGS.nrows)
    log = print_and_log('X_mnk: {:>8} x {:>8} ({:>2} MB)'.format(X_mnk.shape[0], X_mnk.shape[1],
                                                                 sys.getsizeof(X_mnk)/10**6), log)

    # Remove 0-performances if they exist
    if 0 in Y['perf_squared'].tolist():
        print('X', X.shape)
        print('Y', Y.shape)
        print('X_mnk', X_mnk.shape)
        X['perf_squared'] = Y['perf_squared']
        X['mnk'] = X_mnk['mnk']
        print(X.shape)
        X = X.loc[X['perf_squared'] != 0]
        print(X.shape)
        del Y
        Y = pd.DataFrame()
        Y['perf_squared'] = X['perf_squared']
        del X_mnk
        X_mnk = pd.DataFrame()
        X_mnk['mnk'] = X['mnk']
        X.drop(['perf_squared', 'mnk'], axis=1, inplace=True)
        print('X', X.shape)
        print('Y', Y.shape)
        print('X_mnk', X_mnk.shape)

    n_features = len(list(X.columns))
    predictor_names = X.columns.values
    log = print_and_log('Predictor variables: (' + str(n_features) + ')', log)
    for i, p in enumerate(predictor_names):
        log = print_and_log("\t{:2}) {}".format(i+1, p), log)

    with open(log_file, 'w') as f:
        f.write(log)

    return X, X_mnk, Y, folder_


def get_DecisionTree_model(n_features):
    from itertools import chain
    from sklearn.tree import DecisionTreeRegressor

    # Fixed parameters
    model_name_DT = "Decision_Tree"
    splitting_criterion = "mse"
    splitter = "best"
    max_features = None
    max_leaf_nodes = None

    # Parameters to optimize
    if FLAGS.algo == 'tiny':
        step_small = 1
        step_med = 3
        max_depth = chain(range(4, n_features, step_small), range(n_features, n_features*3, step_med))
        min_samples_split = chain(range(2, 5, step_small), range(8, n_features, step_med))
        min_samples_leaf = chain(range(1, 5, step_small), range(8, n_features, step_med))
    elif FLAGS.algo == 'medium':
        max_depth = chain(range(6, 13, 2), range(15, 19, 3))
        min_samples_split = 2
        min_samples_leaf = 2
    else:
        max_depth = chain(range(4, 13, 2), range(15, 19, 3))
        min_samples_split = [2, 5, 13, 18]
        min_samples_leaf = [2, 5, 13, 18]

    param_grid_DT = {
        'max_depth': list(max_depth),
        'min_samples_split': list(min_samples_split),
        'min_samples_leaf': list(min_samples_leaf)
    }

    # Tree model
    model_DT = DecisionTreeRegressor(
        criterion=splitting_criterion,
        splitter=splitter,
        min_samples_split=optimized_hyperparameters[FLAGS.algo]["min_samples_split"],
        min_samples_leaf=optimized_hyperparameters[FLAGS.algo]["min_samples_leaf"],
        max_depth=optimized_hyperparameters[FLAGS.algo]["max_depth"],
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes
    )

    return model_DT, model_name_DT, param_grid_DT


def get_RandomForest_model():
    from itertools import chain
    from sklearn.ensemble import RandomForestRegressor

    # Fixed parameters
    model_name_RF = "Random Forest"
    bootstrap = True
    splitting_criterion = "mse"
    #max_depth = optimized_hyperparameters[FLAGS.algo]['max_depth'][len(optimized_hyperparameters[FLAGS.algo]['max_depth'])//2]
    #min_samples_split = optimized_hyperparameters[FLAGS.algo]['min_samples_split'][len(optimized_hyperparameters[FLAGS.algo]['min_samples_split'])//2]
    #min_samples_leaf = optimized_hyperparameters[FLAGS.algo]['min_samples_leaf'][len(optimized_hyperparameters[FLAGS.algo]['min_samples_leaf'])//2]
    max_features = 'sqrt'

    # Parameters to optimize
    step_big = 50
    step_small = 5
    n_estimators = chain(range(1, 10, step_small), range(50, 200, step_big))
    param_grid_RF = {**optimized_hyperparameters[FLAGS.algo], 'n_estimators': list(n_estimators)}

    # Random Forest model
    model_RF = RandomForestRegressor(
        criterion=splitting_criterion,
        n_estimators=FLAGS.ntrees,
        min_samples_split=optimized_hyperparameters[FLAGS.algo]["min_samples_split"],
        min_samples_leaf=optimized_hyperparameters[FLAGS.algo]["min_samples_leaf"],
        max_depth=optimized_hyperparameters[FLAGS.algo]["max_depth"],
        bootstrap=bootstrap,
        max_features=max_features,
        n_jobs=FLAGS.njobs
    )

    return model_RF, model_name_RF, param_grid_RF


def tune_and_train():

    ####################################################################################################################
    # Read data
    X, X_mnk, Y, folder_ = read_data()

    ####################################################################################################################
    # Predictive model
    if FLAGS.model == "DT":
        model, model_name, param_grid = get_DecisionTree_model(len(X.columns.values))
    elif FLAGS.model == "RF":
        model, model_name, param_grid = get_RandomForest_model()
    else:
        assert False, "Cannot recognize model: " + FLAGS.model + ". Options: DT, RF"

    ############################################################################################################
    # Setup folder and log
    folder = os.path.join(folder_, model_name)
    log_file = os.path.join(folder, 'log.txt')
    log = ''
    if not os.path.exists(folder):
        os.makedirs(folder)
    decisive_score = 'worse_top-3'

    print('\n')
    print('###############################################################################################')
    print("Start hyperparameter optimization for model", model_name)
    print('###############################################################################################')

    ########################################################################################################
    # Train/test split
    from sklearn.model_selection import GroupShuffleSplit
    cv = GroupShuffleSplit(n_splits=2, test_size=0.2)
    train, test = cv.split(X, Y, groups=X_mnk['mnk'])
    train = train[0]
    test = test[0]
    X_train = X.iloc[train, :]  # train: use for hyperparameter optimization (via CV) and training
    X_test  = X.iloc[test,  :]  # test : use for evaluation of 'selected/final' model
    del X
    X_mnk_train = X_mnk.iloc[train, :]
    X_mnk_test = X_mnk.iloc[test, :]
    del X_mnk
    Y_train = Y.iloc[train, :]
    Y_test  = Y.iloc[test,  :]
    del Y

    ########################################################################################################
    # Cross-validation splitter
    n_splits = FLAGS.splits
    test_size = 0.3
    cv = GroupShuffleSplit(n_splits=n_splits, test_size=test_size)
    predictor_names = X_train.columns.values

    if FLAGS.tune:

        ########################################################################################################
        # Feature selection
        from sklearn.feature_selection import RFECV
        log = print_and_log("Selecting optimal features among:\n" + str(predictor_names), log)
        if 'mnk' in X_train.columns.values:
            X_train.drop(["mnk"], axis=1, inplace=True)  # leftover from previous iteration (?)
        if FLAGS.algo in ['small', 'medium']:
            rfecv = RFECV(estimator=model, step=3, n_jobs=FLAGS.njobs, cv=cv, verbose=2)
        else:
            rfecv = RFECV(estimator=model, step=1, n_jobs=FLAGS.njobs, cv=cv, verbose=1)
        fit = rfecv.fit(X_train, Y_train, X_mnk_train['mnk'])
        log = print_and_log("Optimal number of features : %d" % rfecv.n_features_, log)
        selected_features_ = list()
        for i, f in enumerate(predictor_names):
            if fit.support_[i]:
                selected_features_.append(f)
        log = print_and_log("Selected Features:", log)
        for feature in selected_features_:
            log = print_and_log("\t{}".format(feature), log)

        features_to_drop = [f for f in predictor_names if f not in selected_features_]
        for f in features_to_drop:
            X_train.drop([f], axis=1, inplace=True)
            X_test.drop([f], axis=1, inplace=True)

        ########################################################################################################
        # Hyperparameter optimization

        # Grid search
        from sklearn.model_selection import GridSearchCV
        log = print_and_log('----------------------------------------------------------------------------', log)
        log = print_and_log('Parameter grid:\n' + str(param_grid), log)
        X_train["mnk"] = X_mnk_train['mnk']  # add to X-DataFrame (needed for scoring function)
        scoring = {
            'worse_top-1': worse_case_scorer_top1, 'mean_top-1': mean_scorer_top1,
            'worse_top-3': worse_case_scorer_top3, 'mean_top-3': mean_scorer_top3,
            'worse_top-5': worse_case_scorer_top5, 'mean_top-5': mean_scorer_top5
        }
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            pre_dispatch=8,
            n_jobs=FLAGS.njobs,
            verbose=2,
            refit=decisive_score,
            return_train_score=False  # incompatible with ignore_in_fit
        )
        log = print_and_log('----------------------------------------------------------------------------', log)
        gs.fit(X_train, Y_train, X_mnk_train['mnk'], ignore_in_fit=["mnk"])

        describe_model(gs, X_train, X_test, Y_train, Y_test, '')

        safe_pickle([gs.cv_results_], os.path.join(folder, "cv_results_.p"))
        safe_pickle([X_train.columns.values, gs.best_estimator_], os.path.join(folder, "feature_tree.p"))
        if FLAGS.algo not in ['small', 'medium']:
            safe_pickle([X_train, Y_train, X_mnk_train, X_test, Y_test, X_mnk_test, gs],
                        os.path.join(folder, "cv.p"))
        return_model = gs

    else:

        ########################################################################################################
        # Load selected features and hyperparameters
        features_to_drop = [f for f in predictor_names if f not in selected_features[FLAGS.algo]]
        for f in features_to_drop:
            X_train.drop([f], axis=1, inplace=True)
            X_test.drop([f], axis=1, inplace=True)


        ########################################################################################################
        # Fit
        model.fit(X_train, Y_train)
        safe_pickle([X_train.columns.values, model], os.path.join(folder, "model.p"))
        if FLAGS.algo not in ['small', 'medium']:
            safe_pickle([X_train, Y_train, X_mnk_train, X_test, Y_test, X_mnk_test, model],
                        os.path.join(folder, "cv.p"))
        return_model = model



    ############################################################################################################
    # Print log
    with open(log_file, 'w') as f:
        f.write(log)

    # Return
    return X_train, Y_train, X_mnk_train, X_test, Y_test, X_mnk_test, return_model


########################################################################################################################
# Describe and evaluate model
########################################################################################################################
def describe_model(gs, X_train, X_test, Y_train, Y_test, log):
    log = print_and_log('----------------------------------------------------------------------------', log)
    predictor_names = X_train.columns.values
    log = print_and_log('Predictor variables:', log)
    for p in predictor_names:
        log = print_and_log("\t{}".format(p), log)

    log = print_and_log("\nBest parameters set found on development set:", log)
    log = print_and_log(gs.best_params_, log)

    log = print_and_log("\nBest estimator:", log)
    best_estimator = gs.best_estimator_
    log = print_and_log(best_estimator, log)
    log = print_and_log('----------------------------------------------------------------------------', log)

    # Export tree SVG
    if FLAGS.print_tree:
        log = print_and_log('\nExport tree to SVG:', log)
        viz = dtreeviz(best_estimator, X_test.values, Y_test.values.ravel(),
                       target_name='perf',
                       feature_names=X_train.columns.values.tolist())
        viz.save("trytree.svg")
        viz.view()

    return log


def print_error(y_true, y_pred, X_mnk, log):
    result_line = "top-{}: worse: {:>6.3f} mean: {:>6.3f}"
    for top_k in [1, 3, 5]:
        log = print_and_log(result_line.format(top_k,
                                               worse_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk),
                                               mean_rel_perf_loss_of_k(y_true, y_pred, top_k, X_mnk)), log)
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


def plot_cv_scores(param_grid, scoring, results, best_pars, folder, algo):
    # Inspired by http://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
    import matplotlib.pyplot as plt
    for p in param_grid.keys():

        plt.figure()
        plt.title("CV scores (" + algo + ")")
        plt.xlabel("parameter: " + p + "(best:" + str(best_pars) + ")")
        plt.ylabel("cv-score: relative perf loss [%] (mean over " + str(FLAGS.splits) + "folds)")
        ax = plt.gca()

        # Get the regular numpy array from the dataframe
        results_ = results.copy()
        groups_to_fix = list(best_pars.keys())
        groups_to_fix.remove(p)
        for g in groups_to_fix:
            results_ = results_.groupby('param_' + g).get_group(best_pars[g])
        X_axis = np.array(results_['param_' + p].values, dtype=float)
        X_axis_p = results_['param_' + p]

        for scorer, color in zip(scoring, ['b']*2 + ['g']*2 + ['k']*2):
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


def evaluate_model(gs, X_train, X_test, X_mnk_train, X_mnk_test, Y_train, Y_test, log, folder):

    if FLAGS.tune:
        best_estimator = gs.best_estimator_
        scoring = gs.scorer_
        X_train.drop(['mnk'], axis=1, inplace=True)
    else:
        best_estimator = gs

    # Training error
    best_estimator.fit(X_train, Y_train)
    y_train_pred = best_estimator.predict(X_train)
    log = print_and_log('\nTraining error: (train&val)', log)
    log = print_error(Y_train, y_train_pred, X_mnk_train, log)

    # Test error
    y_test_pred = best_estimator.predict(X_test)
    log = print_and_log('\nTesting error:', log)
    log = print_error(Y_test, y_test_pred, X_mnk_test, log)

    # Print histogram for "best" estimator
    log = print_and_log('\nPlot result histogram:', log)
    plot_loss_histogram(Y_test, y_test_pred, X_mnk_test, folder)

    # Plot CV results by evaluation metric
    if FLAGS.tune:
        log = print_and_log('\nPlot CV scores:', log)
        plot_cv_scores(gs.param_grid, scoring, gs.cv_results, gs.best_params_, folder, FLAGS.algo)

    return log


########################################################################################################################
# Write tree to intermediate representation
########################################################################################################################
def write_tree(tree, features, filename):
    """
    Write tree to a convenient format which is easy to version control
    And can be used by an other python script to write a tree compiled to c++
    :param tree:
    :param features:
    :param filename:
    :return:
    """
    print('Write tree from python to', filename)


########################################################################################################################
# Write tree to intermediate representation
########################################################################################################################
def compile_tree_to_cpp(filename_txt, filename_cpp):
    """
    Write representation of a tree to compiled cpp
    :param filename_txt:
    :param filename_cpp:
    :return:
    """
    print('Write', filename_txt, 'to', filename_cpp)


########################################################################################################################
# Main
########################################################################################################################
def main(argv):
    del argv  # unused

    ####################################################################################################################
    # Create folder to store results of this training
    log = ''
    log_file, folder = create_log_folder()

    ####################################################################################################################
    # Get or train model
    if len(FLAGS.gs) == 0:

        # Train model
        eval = True
        X_train, Y_train, X_mnk_train, X_test, Y_test, X_mnk_test, gs = tune_and_train()

    else:

        # Get pre-trained model
        eval = FLAGS.eval
        log = print_and_log("Reading GridSearCV from " + FLAGS.gs, log)
        if FLAGS.algo in ['medium', 'small']:
            cv_results = pickle.load(open(os.path.join(FLAGS.gs, "cv_results_.p"), 'rb'))
            feature_names, best_estimator = pickle.load(open(os.path.join(FLAGS.gs, "feature_tree.p"), 'rb'))
        else:
            X_train, Y_train, X_mnk_train, X_test, Y_test, X_mnk_test, gs = pickle.load(open(FLAGS.gs, 'rb'))

    ####################################################################################################################
    # Model description
    if FLAGS.tune:
        log = describe_model(gs, X_train, X_test, Y_train, Y_test, log)

    ####################################################################################################################
    # Model evaluation
    if eval:
        log = evaluate_model(gs, X_train, X_test, X_mnk_train, X_mnk_test, Y_train, Y_test, log, folder)

    ####################################################################################################################
    # Write tree description to VC-able file
    # filename_txt = 'try_tree_write.txt'
    # best_estimator = gs.best_estimator_
    # predictor_names = ... ?
    # write_tree(best_estimator, predictor_names, filename_txt)

    ####################################################################################################################
    # Write tree description to VC-able file
    # filename_cpp = 'try_tree_write.cpp'
    # compile_tree_to_cpp(filename_txt, filename_cpp)


    ############################################################################################################
    # Print log
    with open(log_file, 'w') as f:
        f.write(log)


########################################################################################################################
# Run
########################################################################################################################
if __name__ == '__main__':
    app.run(main)
