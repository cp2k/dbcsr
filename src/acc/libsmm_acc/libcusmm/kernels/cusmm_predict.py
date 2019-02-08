####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

import re
import numpy as np

# ===============================================================================
# Dictionary of available kernel algorithms
# keys: kernel name
# values: kernel implementation class
from kernels.cusmm_dnt_largeDB1 import Kernel_dnt_largeDB1
from kernels.cusmm_dnt_largeDB2 import Kernel_dnt_largeDB2
from kernels.cusmm_dnt_medium import Kernel_dnt_medium
from kernels.cusmm_dnt_small import Kernel_dnt_small
from kernels.cusmm_dnt_tiny import Kernel_dnt_tiny

kernel_algorithm = {
    "tiny": Kernel_dnt_tiny,
    "small": Kernel_dnt_small,
    "medium": Kernel_dnt_medium,
    "largeDB1": Kernel_dnt_largeDB1,
    "largeDB2": Kernel_dnt_largeDB2,
}

# ===============================================================================
# Dictionary of available GPU architectures.
# keys: parameter_file
# values: CUDA compute versions
arch_number = {
    "parameters_K20X.json": 35,
    "parameters_K40.json": 35,
    "parameters_K80.json": 37,
    "parameters_P100.json": 60,
    "parameters_V100.json": 70,
}


# ===============================================================================
def compatible_mnk(algo, m, n, k):
    """Determine whether a given algorithm is compatible with given m, n, k values"""

    max_sizes = max(m * k, n * k, m * n)
    compatible = True
    if algo == "tiny":
        if max_sizes > 64:
            compatible = False
    elif algo == "small":
        if max_sizes > 128:
            compatible = False
    elif algo in ["largeDB1", "largeDB2"]:
        if max_sizes < 250:
            compatible = False
    else:
        if algo != "medium":
            assert False, "Cannot identify algorithm:" + str(algo)

    return compatible


# ===============================================================================
def params_dict_to_kernel(**params):
    """Given a dictionary of parameters, return the corresponding Kernel class instance"""

    # Get the 'algorithm' field
    algo = params.pop("algorithm")

    # Get the list of fields needed to initialize a Kernel instance of this given algorithm
    kernel_init_params = kernel_algorithm[algo].launch_parameters + ["perf", "source"]

    # Fill in dictionary fields
    kernel_init_params_dict = dict()
    for k in kernel_init_params:
        if k == "perf" and params["source"] == "predicted":
            # the performance of predicted parameter sets is not given
            kernel_init_params_dict["perf"] = None
        else:
            kernel_init_params_dict[k] = params[k]

    return kernel_algorithm[algo](**kernel_init_params_dict)


def descr_to_kernel(kernel_descr, source="autotuned"):
    """Given a kernel description from the autotuning output, return the corresponding Kernel class instance"""

    from ast import literal_eval

    re_kernel_descr = re.compile(r"Kernel_dnt_(\w+)(\(.*\)) , # (\d+\.\d+) GFlop/s")
    match = re_kernel_descr.search(kernel_descr).groups()
    algo = match[0]
    m = match[1].replace("=", "':")
    m = m.replace(", ", ", '")
    m = m.replace("(", "{'")
    m = m.replace(")", "}")
    params = dict(literal_eval(m))
    params["perf"] = float(match[2])
    params["source"] = source
    return kernel_algorithm[algo](**params)


def to_string(*iterable):
    """
    Given a (list of) m,n,k-triplet(s), return the corresponding (list of) string(s) "mxnxk"
    """
    mnk_string = "{}x{}x{}"
    if len(iterable) == 3 and isinstance(iterable[0], int):
        m, n, k = iterable
        iterable_to_string = mnk_string.format(m, n, k)
    else:
        iterable_to_string = [mnk_string.format(m, n, k) for m, n, k in iterable]
    if len(iterable_to_string) == 1:
        iterable_to_string = iterable_to_string[0]
    return iterable_to_string


def to_tuple(*iterable):
    """
    Given a (list of) string(s) "mxnxk", return the corresponding (list of) m,n,k-triplet(s)
    """
    mnk_pattern = re.compile(r"(\d+)x(\d+)x(\d+)")
    tuple_mnks = list()
    for mnk in iterable:
        m, n, k = mnk_pattern.match(mnk).groups()
        tuple_mnks.append((int(m), int(n), int(k)))
    if len(tuple_mnks) == 1:
        tuple_mnks = tuple_mnks[0]
    return tuple_mnks


# ===============================================================================
# Lists of derived parameters to use as training data for the predictive modelling
# Some of the computable features are commented out because they are constant or (almost) linearly dependent on another
# feature, and therefore they do not contribute to the decision tree model.
raw_parameters = [
    "m",
    "n",
    "k",
    "threads_per_blk",
    "grouping",
    "minblocks",
    "tile_m",
    "tile_n",
    "w",
    "v",
    "perf (Gflop/s)",
]
derived_parameters = {
    "common": [
        "perf_scaled",
        # 'Gflops', # linearly dependent on mxnxk
        "mxnxk",
        "size_a",
        "size_b",
        "size_c",
        # 'nblks', 'nthreads',  # constant value for largeDB, since the grouping is always = 16
        "sm_desired"
        # 'warps_per_blk', 'nwarps',  # linearly dependent on threads, nthreads
        # 'ru_param_stack_unroll_factor',  # constant values for each algo
    ],
    "tiny": [
        "nblks",
        "nthreads",
        # tiny, small, medium: resource occupation
        "ru_tinysmallmed_unroll_factor_a",
        "ru_tinysmallmed_unroll_factor_a_total",
        "ru_tinysmallmed_unroll_factor_b",
        "ru_tinysmallmed_unroll_factor_b_total",
        "ru_tinysmallmed_unroll_factor_c_total",
        # tiny: resource occupation
        "ru_tiny_max_parallel_work",
        "ru_tiny_smem_per_block",
        # 'ru_tiny_min_threads',  # equal to size_c
        # 'ru_tiny_nblks_per_sm',  # always = 32, so also removing: 'ru_tiny_nwarps_per_sm', 'ru_tiny_occupancy'
        "ru_tiny_nsm",  # 'ru_tiny_ngpu',  # highly correlated with ru_tiny_n_sm
    ],
    "small": [
        "nblks",
        "nthreads",
        # tiny, small, medium: resource occupation
        "ru_tinysmallmed_unroll_factor_a",
        "ru_tinysmallmed_unroll_factor_a_total",
        "ru_tinysmallmed_unroll_factor_b",
        "ru_tinysmallmed_unroll_factor_b_total",
        "ru_tinysmallmed_unroll_factor_c_total",
        # small, medium: resource occupation
        "ru_smallmed_unroll_factor_c",
        "ru_smallmed_loop_matmul",
        "ru_smallmed_max_parallel_work",
        # 'ru_smallmed_buf_size',  # highly correlated with ru_smallmed_smem_per_block
        "ru_smallmed_smem_per_block",
        "ru_smallmed_regs_per_thread",
        # small, medium, large: resource occupation
        "ru_smallmedlarge_cmax",
        "ru_smallmedlarge_rmax",
        "ru_smallmedlarge_T",
        "ru_smallmedlarge_min_threads",
    ],
    "medium": [
        "nblks",
        "nthreads",
        # tiny, small, medium: resource occupation
        "ru_tinysmallmed_unroll_factor_a",
        "ru_tinysmallmed_unroll_factor_a_total",
        "ru_tinysmallmed_unroll_factor_b",
        "ru_tinysmallmed_unroll_factor_b_total",
        "ru_tinysmallmed_unroll_factor_c_total",
        # medium: resource occupation
        # 'load_unroll_factor_1', 'load_unroll_factor_2',  # highly correlated with ru_tinysmallmed_unroll_factor_a,b
        # 'n_mkloads', 'n_knloads',  # constant value
        # small, medium: resource occupation
        "ru_smallmed_unroll_factor_c",
        "ru_smallmed_loop_matmul",
        "ru_smallmed_max_parallel_work",
        # 'ru_smallmed_buf_size',  # highly correlated with ru_smallmed_smem_per_block
        "ru_smallmed_smem_per_block",
        "ru_smallmed_regs_per_thread",
        # small, medium, large: resource occupation
        "ru_smallmedlarge_cmax",
        "ru_smallmedlarge_rmax",
        "ru_smallmedlarge_T",
        "ru_smallmedlarge_min_threads",
    ],
    "largeDB1": [
        # largeDB: resource occupation
        "ru_large_Pa",
        "ru_large_Pb",
        "ru_large_unroll_factor_a",
        "ru_large_unroll_factor_b",
        "ru_large_unroll_factor_c",
        "ru_large_loop_matmul",
        "ru_large_max_concurrent_work",
        "ru_large_n_DB_iter",
        "ru_large_regs_per_thread",
        # 'ru_large_buf_size',  # highly correlated with ru_large_smem_per_block
        "ru_large_smem_per_block",
        # small, medium, large: resource occupation
        "ru_smallmedlarge_cmax",
        "ru_smallmedlarge_rmax",
        "ru_smallmedlarge_T",
        "ru_smallmedlarge_min_threads",
    ],
    "largeDB2": [
        # largeDB: resource occupation
        "ru_large_Pa",
        "ru_large_Pb",
        "ru_large_unroll_factor_a",
        "ru_large_unroll_factor_b",
        "ru_large_unroll_factor_c",
        "ru_large_loop_matmul",
        "ru_large_max_concurrent_work",
        "ru_large_n_DB_iter",
        "ru_large_regs_per_thread",
        # 'ru_large_buf_size',  # highly correlated with ru_large_smem_per_block
        "ru_large_smem_per_block",
        # small, medium, large: resource occupation
        "ru_smallmedlarge_cmax",
        "ru_smallmedlarge_rmax",
        "ru_smallmedlarge_T",
        "ru_smallmedlarge_min_threads",
    ],
}


# ===============================================================================
def get_max_performances_per_mnk(data):
    """
    Construct dictionary:
        keys: (m, n, k)-tuple,
        values: maximum performance found over all algorithms for this given (m, n, k)
    """
    # Get list of different (m, n, k)s occurring in this instance
    data["mnk"] = list(zip(data["m"], data["n"], data["k"]))
    mnks = np.unique(data["mnk"])

    # Get max. performance per (m, n, k)
    max_perf = dict()

    for mnk in mnks:
        # Get indices corresponding to this mnk
        idx_mnk = np.where(data["mnk"] == mnk)[0].tolist()

        # Get performances per mnk
        perf_mnk_algo = data["perf (Gflop/s)"].values[idx_mnk]

        # Store maxperf
        maxperf = float(
            perf_mnk_algo.max(axis=0)
        )  # max. performance found through autotuning
        max_perf[mnk] = maxperf

    return max_perf


# ===============================================================================
def get_baseline_performances_per_mnk(data, algorithm, gpu, autotuning):
    """
    Construct dictionary:
        keys: (m, n, k)-tuple,
        values: baseline performance for this given (m, n, k) and the given algorithm
    """

    # Get list of different (m, n, k)s occurring in this instance
    data["mnk"] = list(zip(data["m"], data["n"], data["k"]))
    mnks = np.unique(data["mnk"])

    # Get baseline performance per (m, n, k)
    baseline_perf = dict()

    for mnk in mnks:
        m, n, k = mnk

        baseline_pars = kernel_algorithm[algorithm].baseline(m, n, k, gpu, autotuning)

        if np.isnan(baseline_pars["tile_m"]):
            idx_baseline = data[
                (data.m == baseline_pars["m"])
                & (data.n == baseline_pars["n"])
                & (data.k == baseline_pars["k"])
                & (data.threads == baseline_pars["threads"])
                & (data.grouping == baseline_pars["grouping"])
                & (data.minblocks == baseline_pars["minblocks"])
            ].index.tolist()
        elif np.isnan(baseline_pars["w"]):
            idx_baseline = data[
                (data.m == baseline_pars["m"])
                & (data.n == baseline_pars["n"])
                & (data.k == baseline_pars["k"])
                & (data.threads == baseline_pars["threads"])
                & (data.grouping == baseline_pars["grouping"])
                & (data.minblocks == baseline_pars["minblocks"])
                & (data.tile_m == baseline_pars["tile_m"])
                & (data.tile_n == baseline_pars["tile_n"])
            ].index.tolist()
        else:
            idx_baseline = data[
                (data.m == baseline_pars["m"])
                & (data.n == baseline_pars["n"])
                & (data.k == baseline_pars["k"])
                & (data.threads == baseline_pars["threads"])
                & (data.grouping == baseline_pars["grouping"])
                & (data.minblocks == baseline_pars["minblocks"])
                & (data.tile_m == baseline_pars["tile_m"])
                & (data.tile_n == baseline_pars["tile_n"])
                & (data.w == baseline_pars["w"])
                & (data.v == baseline_pars["v"])
            ].index.tolist()

        if len(idx_baseline) < 1:
            idx_baseline = data[
                (data.m == baseline_pars["m"])
                & (data.n == baseline_pars["n"])
                & (data.k == baseline_pars["k"])
                & (data.threads == baseline_pars["threads"])
            ].index.tolist()
            assert len(idx_baseline) > 0

        idx_baseline = idx_baseline[0]
        baseline_perf[mnk] = data["perf (Gflop/s)"][idx_baseline]

    return baseline_perf


# ===============================================================================
class PredictiveParameters:
    """
    Class handling predictive features for the predictive modelling of libcusmm's performance
    """

    def __init__(
        self, params_df, gpu, autotuning, max_performances, partial_initialization=False
    ):
        """
        params_df: pandas Dataframe where each row corresponds to a kernel parameter set
        """
        assert "m" in params_df.columns.values
        assert "n" in params_df.columns.values
        assert "k" in params_df.columns.values
        self.gpu = gpu  # GPU card properties
        self.autotuning = autotuning  # autotuning properties
        self.max_performances = max_performances  # dictionary of max. performances
        # keys: (m, n, k)-tuple, values: maximum performance
        # found over all algorithms for this given (m, n, k)

        if not partial_initialization:
            assert "threads" in params_df.columns.values, (
                "Missing column: threads. Available columns:\n"
                + str(params_df.columns.values)
            )
            assert "grouping" in params_df.columns.values, (
                "Missing column: grouping. Available columns:\n"
                + str(params_df.columns.values)
            )
            assert "minblocks" in params_df.columns.values, (
                "Missing column: minblocks. Available columns:\n"
                + str(params_df.columns.values)
            )
            algos = np.unique(params_df["algorithm"].values)
            assert len(algos) == 1
            algo = algos[0]
            if algo in ["small", "medium", "largeDB1", "largeDB2"]:
                assert "tile_m" in params_df.columns.values, (
                    "Missing column: tile_m. Available columns:\n"
                    + str(params_df.columns.values)
                )
                assert "tile_n" in params_df.columns.values, (
                    "Missing column: tile_n. Available columns:\n"
                    + str(params_df.columns.values)
                )
                if algo in ["largeDB1", "largeDB2"]:
                    assert "w" in params_df.columns.values, (
                        "Missing column: w. Available columns:\n"
                        + str(params_df.columns.values)
                    )
                    assert "v" in params_df.columns.values, (
                        "Missing column: v. Available columns:\n"
                        + str(params_df.columns.values)
                    )

        self.params = params_df

    def get(self, feature_name):
        """Generic function to compute any feature given by name"""

        if feature_name not in self.params.columns.values:
            if feature_name != "perf_scaled":  # not vectorizable
                vget = getattr(self, "get_" + feature_name)
            else:
                vget = np.vectorize(getattr(self, "get_" + feature_name))
            feature_val = vget()
        else:
            feature_val = self.params[feature_name].values
        return feature_val

    def get_features(self, feature_names):
        """
        Compute a list of features given by name and return them as a pandas Dataframe.
        :param feature_names: list of names of features to compute
        """
        for feat in feature_names:
            self.params.loc[:, feat] = self.get(feat)
        return self.params[feature_names]

    # ===============================================================================
    # Performances
    def get_perf_scaled(self):
        """
        Scale raw performances in [Gflop/s] between 0 and 1, where
            0 = 0 Gflop/s
            1 = performance equal to autotuned maximum FOR ALL ALGORITHMS
        :return: numpy array of scaled performances
        """

        def scale_perf(perf, mnk):
            """For a given mnk and a given performance on this mnk, return the scaled performance"""
            return perf / self.max_performances[mnk]

        vec_scale_perf = np.vectorize(scale_perf)
        ret = vec_scale_perf(self.get("perf (Gflop/s)"), self.get("mnk"))
        return ret

    # ===============================================================================
    # Matrix sizes
    def get_size_a(self):
        """Size of matrix A (first operand of A * B = C)"""
        return self.get("m") * self.get("k")

    def get_size_b(self):
        """Size of matrix B (second operand of A * B = C)"""
        return self.get("k") * self.get("n")

    def get_size_c(self):
        """Size of matrix B (result of of A * B = C)"""
        return self.get("m") * self.get("n")

    def get_mnk(self):
        """Return (m, n, k) as a tuple"""
        return self.get("m"), self.get("n"), self.get("k")

    def get_mxnxk(self):
        """Return the product m*n*k"""
        return self.get("m") * self.get("n") * self.get("k")

    # ===============================================================================
    # Launch parameters
    def get_need_sync(self):
        """(mn > warp_size || mk > warp_size || kn > warp_size || threads > warp_size)"""
        return (
            np.where(self.get("size_c") > self.gpu["Threads_/_Warp"], True, False)
            | np.where(self.get("size_a") > self.gpu["Threads_/_Warp"], True, False)
            | np.where(self.get("size_b") > self.gpu["Threads_/_Warp"], True, False)
            | np.where(self.get("threads") > self.gpu["Threads_/_Warp"], True, False)
        )

    def get_nblks(self):
        """Number of thread blocks needed to multiply all matrices on the stack"""
        return np.ceil(self.autotuning["stack_size"] / self.get("grouping"))

    def get_warps_per_blk(self):
        """Number of warps per block"""
        return np.ceil(self.get("threads") / self.gpu["Threads_/_Warp"])

    def get_nwarps(self):
        """Total number of warps needed to multiply all matrices on the stack"""
        return self.get("warps_per_blk") * self.get("nblks")

    def get_sm_desired(self):
        """
        Number of multiprocessors desired to multiply all matrices on the stack.
        For more details, see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds
        """
        return np.ceil(self.get("nblks") / self.get("minblocks"))

    def get_nthreads(self):
        """Total number of threads needed to multiply all matrices on the stack"""
        return self.get("threads") * self.get("nblks")

    # ===============================================================================
    # Resource usage (common)
    def get_ru_param_stack_unroll_factor(self):
        """Number of executions of the body of the loop that loads data from the parameter stack"""
        return np.ceil(self.get("grouping") / self.get("threads"))

    def get_n_iter(self):
        """Number of benchmark repetitions in autotuning procedure"""
        return np.maximum(
            3, 12500 * (1 // (self.get("m") * self.get("n") * self.get("k")))
        )

    def get_Gflops(self):
        """Number of floating point operations in [Gflops] carried out during autotuning"""
        return (
            self.get("n_iter")
            * self.autotuning["stack_size"]
            * self.get("m")
            * self.get("n")
            * self.get("k")
            * 2
            * 10 ** (-9)
        )

    # ===============================================================================
    # Resource occupancy estimations
    def get_nblocks_per_sm_lim_blks_warps(self):
        """Resource occupations in terms of warps and blocks (Follows CUDA calculator sheet)"""
        return np.minimum(
            self.gpu["Thread_Blocks_/_Multiprocessor"],
            np.floor(self.gpu["Warps_/_Multiprocessor"] / self.get("warps_per_blk")),
        )

    # ===============================================================================
    # Resource usage (tiny, small, medium)
    def get_ru_tinysmallmed_unroll_factor_a(self):
        """loop unroll factor of the loop on m*k"""
        return np.ceil(self.get("size_a") / self.get("threads"))

    def get_ru_tinysmallmed_unroll_factor_b(self):
        """loop unroll factor of the loop on k*m"""
        return np.ceil(self.get("size_b") / self.get("threads"))

    def get_ru_tinysmallmed_unroll_factor_a_total(self):
        """loop unroll factor multiplied by number of times the loop is run"""
        return self.get("ru_tinysmallmed_unroll_factor_a") * self.get("grouping")

    def get_ru_tinysmallmed_unroll_factor_b_total(self):
        """loop unroll factor multiplied by number of times the loop is run"""
        return self.get("ru_tinysmallmed_unroll_factor_b") * self.get("grouping")

    def get_ru_tinysmallmed_unroll_factor_c_total(self):
        """loop unroll factor of the loop writing back to C multiplied by number of times the loop is run"""
        return self.get("k") * self.get("grouping")

    # ===============================================================================
    # Resource usage (tiny)
    def get_ru_tiny_max_parallel_work(self):
        """Total number of iterations in each loop"""
        return np.maximum.reduce(
            [
                self.get("grouping"),
                self.get("size_a"),
                self.get("size_b"),
                self.get("size_c"),
            ]
        )

    def get_ru_tiny_min_threads(self):
        """Minimum number of threads required to run the kernel and produce correct results"""
        return self.get("size_c")

    def get_ru_tiny_buf_size(self):
        """Buffer size"""
        return self.get("k") * (self.get("m") + self.get("n"))

    def get_ru_tiny_smem_per_block(self):
        """"Shared memory usage per block (estimate)"""
        return (self.get("ru_tiny_buf_size") * self.autotuning["sizeof_double"]) + (
            self.autotuning["npars"]
            * self.get("grouping")
            * self.autotuning["sizeof_int"]
        )

    def get_ru_tiny_nblks_per_sm(self):
        """
        Occupancy estimation: assumption (verified on a sample of mnks): nblks is always limited by number of threads
        for algorithm tiny
        """
        return self.get("nblocks_per_sm_lim_blks_warps")

    def get_ru_tiny_nwarps_per_sm(self):
        """Number of wars per multiprocessor"""
        return self.get("ru_tiny_nblks_per_sm") * self.get("warps_per_blk")

    def get_ru_tiny_nsm(self):
        """Number of multiprocessors"""
        return np.ceil(self.get("nblks") / self.get("ru_tiny_nblks_per_sm"))

    def get_ru_tiny_ngpu(self):
        """Number of GPUs"""
        return np.ceil(self.get("ru_tiny_nsm") / self.gpu["Multiprocessors"])

    def get_ru_tiny_occupancy(self):
        return self.get("ru_tiny_nwarps_per_sm") / self.gpu["Warps_/_Multiprocessor"]

    # ===============================================================================
    # Resource usage (small, medium, large)
    def get_ru_smallmedlarge_cmax(self):
        return np.ceil(self.get("n") / self.get("tile_n"))

    def get_ru_smallmedlarge_rmax(self):
        return np.ceil(self.get("m") / self.get("tile_m"))

    def get_ru_smallmedlarge_T(self):
        return self.get("tile_m") * self.get("tile_n")

    def get_ru_smallmedlarge_min_threads(self):
        return self.get("ru_smallmedlarge_cmax") * self.get("ru_smallmedlarge_rmax")

    # ===============================================================================
    # Resource usage estimation and loop counts (small, medium)
    def get_ru_smallmed_tm_max(self):
        return self.get("m")

    def get_ru_smallmed_tn_max(self):
        return self.get("n")

    def get_ru_smallmed_unroll_factor_c(self):
        """loop unroll factor of the loop on m*n"""
        return np.ceil(self.get("size_c") / self.get("threads"))

    def get_ru_smallmed_loop_matmul(self):
        """Actual multiplication loop"""
        return self.get("k") * self.get("tile_m") * self.get("tile_n")

    def get_ru_smallmed_max_parallel_work(self):
        """Maximum parallel work"""
        return np.maximum.reduce(
            [
                self.get("grouping"),
                self.get("size_a"),
                self.get("size_b"),
                self.get("size_c"),
                self.get("ru_smallmedlarge_min_threads"),
            ]
        )

    def get_ru_smallmed_buf_size(self):
        """Buffer size"""
        intermediate1 = self.get("size_a") + self.get("k") * self.get(
            "tile_n"
        ) * self.get("ru_smallmedlarge_cmax")
        intermediate2 = (
            self.get("tile_m") * self.get("ru_smallmedlarge_rmax") * self.get("k") + 1
        )
        return np.maximum.reduce([self.get("size_c"), intermediate1, intermediate2])

    def get_ru_smallmed_smem_per_block(self):
        """Shared memory usage per block"""
        return (self.get("ru_smallmed_buf_size") * self.autotuning["sizeof_double"]) + (
            self.autotuning["npars"]
            * self.get("grouping")
            * self.autotuning["sizeof_int"]
        )

    def get_ru_smallmed_regs_per_thread(self):
        """Register usage per thread (estimated)"""
        return (
            self.get("tile_m") * self.get("tile_n")
            + (self.get("m") * self.get("k") + self.get("threads") - 1)
            // self.get("threads")
            + (self.get("k") * self.get("n") + self.get("threads") - 1)
            // self.get("threads")
        )

    # ===============================================================================
    # Resource usage (medium)
    # Loop bounds
    def get_load_unroll_factor_1(self):
        return self.get("size_a") // self.get("threads") + 1

    def get_load_unroll_factor_2(self):
        return self.get("size_b") // self.get("threads") + 1

    def get_n_mkloads(self):
        return self.get("size_a") // (
            self.get("load_unroll_factor_1") * self.get("threads")
        )

    def get_n_knloads(self):
        return self.get("size_b") // (
            self.get("load_unroll_factor_2") * self.get("threads")
        )

    # ===============================================================================
    # Resource usage (large)
    def get_ru_large_Pa(self):
        """Input slab size"""
        return self.get("m") * self.get("w")

    def get_ru_large_Pb(self):
        """Input slab size"""
        return self.get("w") * self.get("n")

    def get_ru_large_Pc(self):
        """Output slab size"""
        return self.get("m") * self.get("v")

    def get_ru_large_unroll_factor_a(self):
        return np.ceil(self.get("ru_large_Pa") / self.get("threads"))

    def get_ru_large_unroll_factor_b(self):
        return np.ceil(self.get("ru_large_Pb") / self.get("threads"))

    def get_ru_large_unroll_factor_c(self):
        return np.ceil(self.get("ru_large_Pc") / self.get("threads"))

    def get_ru_large_loop_matmul(self):
        return self.get("w") * self.get("tile_m") * self.get("tile_n")

    def get_ru_large_max_concurrent_work(self):
        """Maximum concurrent work"""
        return np.maximum.reduce(
            [
                self.get("grouping"),
                self.get("ru_large_Pa"),
                self.get("ru_large_Pb"),
                self.get("ru_large_Pc"),
                self.get("ru_smallmedlarge_T"),
            ]
        )

    def get_ru_large_regs_per_thread(self):
        """Register usage per thread (estimated)"""
        return (
            self.get("tile_m") * self.get("tile_n")
            + (self.get("w") * self.get("m") + self.get("threads") - 1)
            // self.get("threads")
            + (self.get("w") * self.get("n") + self.get("threads") - 1)
            // self.get("threads")
        )

    def get_ru_large_n_DB_iter(self):
        """Number of double-buffering iterations"""
        return self.get("k") // (2 * self.get("w"))

    def get_ru_large_buf_size(self):
        """Buffer size"""
        intermediate1 = (self.get("w") - 1) * self.get("m") + self.get(
            "ru_smallmedlarge_rmax"
        ) * self.get("tile_m")
        intermediate2 = (
            self.get("m") * self.get("w")
            + (self.get("w") - 1) * self.get("n")
            + self.get("ru_smallmedlarge_cmax") * self.get("tile_n")
        )
        return np.maximum.reduce(
            [self.get("ru_large_Pc"), intermediate1, intermediate2]
        )

    def get_ru_large_smem_per_block(self):
        """Shared memory usage per block"""
        return (
            self.get("ru_large_buf_size") * self.autotuning["sizeof_double"]
            + self.autotuning["npars"]
            * self.get("grouping")
            * self.autotuning["sizeof_int"]
        )
