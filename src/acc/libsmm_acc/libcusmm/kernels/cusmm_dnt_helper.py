####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

import numpy as np


# ===============================================================================
#  Computing helpers
def round_up_to_nearest_multiple(x, step):
    result = np.where(x % step == 0, x, x + step - x % step).astype(float)
    if result.size == 1:
        result = result.item()  # extract single element of numpy array
    return result


def round_down_to_nearest_multiple(x, step):
    result = np.where(x % step == 0, x, x - x % step).astype(float)
    if result.size == 1:
        result = result.item()  # extract single element of numpy array
    return result


# ===============================================================================
# Available kernel algorithms and the classes that implement them
from kernels.cusmm_dnt_largeDB1 import Kernel_dnt_largeDB1
from kernels.cusmm_dnt_largeDB2 import Kernel_dnt_largeDB2
from kernels.cusmm_dnt_medium import Kernel_dnt_medium
from kernels.cusmm_dnt_small import Kernel_dnt_small
from kernels.cusmm_dnt_tiny import Kernel_dnt_tiny

kernel_algorithm = {
    'tiny': Kernel_dnt_tiny,
    'small': Kernel_dnt_small,
    'medium': Kernel_dnt_medium,
    'largeDB1': Kernel_dnt_largeDB1,
    'largeDB2': Kernel_dnt_largeDB2
}


# ===============================================================================
# Correspondence between CUDA compute versions and parameter_file
arch_number = {
    "parameters_K20X.json": 35,
    "parameters_K40.json": 35,
    "parameters_K80.json": 37,
    "parameters_P100.json": 60
}


# ===============================================================================
def compatible_mnk(algo, m, n, k):
    """Determine whether a given algorithm is compatible with given m, n, k values, """

    max_sizes = max(m * k, n * k, m * n)
    compatible = True
    if algo == 'tiny':
        if max_sizes > 64:
            compatible = False
    elif algo == 'small':
        if max_sizes > 128:
            compatible = False
    elif algo in ['largeDB1', 'largeDB2']:
        if max_sizes < 250:
            compatible = False

    return compatible


# ===============================================================================
def params_dict_to_kernel(**params):
    """Given a dictionary of parameters, return the corresponding Kernel class instance"""

    algo = params.pop('algorithm')
    kernel_init_params = ['m', 'n', 'k', 'threads', 'grouping', 'minblocks', 'perf', 'source']
    if algo in ['small', 'medium', 'largeDB1', 'largeDB2']:
        kernel_init_params.append("tile_m")
        kernel_init_params.append("tile_n")
        if algo in ['largeDB1', 'largeDB2']:
            kernel_init_params.append("v")
            kernel_init_params.append("w")

    kernel_init_params_dict = dict()
    if 'threads_per_blk' in params.keys():
        kernel_init_params_dict['threads'] = params['threads_per_blk']
        kernel_init_params.remove('threads')

    for k in kernel_init_params:
        kernel_init_params_dict[k] = params[k]

    return kernel_algorithm[algo](**kernel_init_params_dict)


def descr_to_kernel(kernel_descr, source='autotuned'):
    """Given a kernel description as outputed by autotuning, return the corresponding Kernel class instance"""

    import re
    from ast import literal_eval

    re_kernel_descr = re.compile(r"Kernel_dnt_(\w+)(\(.*\)) , # (\d+\.\d+) GFlop/s")
    match = re_kernel_descr.search(kernel_descr).groups()
    algo = match[0]
    m = match[1].replace('=', '\':')
    m = m.replace(', ', ', \'')
    m = m.replace('(', '{\'')
    m = m.replace(')', '}')
    params = dict(literal_eval(m))
    params['perf'] = float(match[2])
    params['source'] = source
    return kernel_algorithm[algo](**params)


def to_string(*iterable):
    mnk_string = '{}x{}x{}'
    return [mnk_string.format(m, n, k) for m, n, k in iterable]


def to_tuple(*iterable):
    import re
    mnk_pattern = re.compile('(\d+)x(\d+)x(\d+)')
    tuple_mnks = list()
    for mnk in iterable:
        match_ = mnk_pattern.match(mnk).groups()
        m, n, k = match_
        tuple_mnks.append((int(m), int(n), int(k)))
    return tuple_mnks


# ===============================================================================
# Lists of raw/derived parameters to be computed y algorithm
raw_parameters = ['m', 'n', 'k',
                  'threads_per_blk', 'grouping', 'minblocks',
                  'tile_m', 'tile_n', 'w', 'v',
                  'perf (Gflop/s)']
raw_parameters_withcompileinfo = raw_parameters + ['regs_per_thread', 'nbytes_smem', 'nbytes_cmem']
derived_parameters = {
    'common': [
        'mxnxk', 'size_a', 'size_b', 'size_c',
        # 'nblks', 'nthreads',  # constant value for largeDB, since the grouping is always = 16
        'sm_desired'
        # 'warps_per_blk', 'nwarps',  # linearly dependent on threads_per_blk, nthreads
        # 'ru_param_stack_unroll_factor',  # always = 1 for tiny, added to each algo
    ],
    'tiny': [
        # tiny, small, medium: resource occupation
        'ru_tinysmallmed_unroll_factor_a', 'ru_tinysmallmed_unroll_factor_a_total',
        'ru_tinysmallmed_unroll_factor_b', 'ru_tinysmallmed_unroll_factor_b_total',
        'ru_tinysmallmed_unroll_factor_c_total',

        # tiny: resource occupation
        'ru_tiny_max_parallel_work', 'ru_tiny_min_threads', 'ru_tiny_smem_per_block',
        'ru_tiny_nblks_per_sm', 'ru_tiny_nwarps_per_sm', 'ru_tiny_nsm', 'ru_tiny_ngpu', 'ru_tiny_occupancy',

        # Kothapalli
        'Koth_tiny_Nmem', 'Koth_tiny_perf_K'

    ],
    'small': [

        'grouping', 'nblks', 'nthreads',

        # tiny, small, medium: resource occupation
        'ru_tinysmallmed_unroll_factor_a', 'ru_tinysmallmed_unroll_factor_a_total',
        'ru_tinysmallmed_unroll_factor_b', 'ru_tinysmallmed_unroll_factor_b_total',
        'ru_tinysmallmed_unroll_factor_c_total',

        # small, medium: resource occupation
        'ru_smallmed_unroll_factor_c', 'ru_smallmed_loop_matmul',
        'ru_smallmed_max_parallel_work', 'ru_smallmed_buf_size',
                                         'ru_smallmed_smem_per_block', 'ru_smallmed_regs_per_thread',

        # small, medium, large: resource occupation
        'ru_smallmedlarge_cmax', 'ru_smallmedlarge_rmax',
        'ru_smallmedlarge_T', 'ru_smallmedlarge_min_threads',

        # Kothapalli
        'Koth_small_Nmem', 'Koth_small_perf_K'

    ],
    'medium': [

        'grouping', 'nblks', 'nthreads',

        # tiny, small, medium: resource occupation
        'ru_tinysmallmed_unroll_factor_a', 'ru_tinysmallmed_unroll_factor_a_total',
        'ru_tinysmallmed_unroll_factor_b', 'ru_tinysmallmed_unroll_factor_b_total',
        'ru_tinysmallmed_unroll_factor_c_total',

        # medium: resource occupation
        # 'load_unroll_factor_1', 'load_unroll_factor_2',  # highly correlated with ru_tinysmallmed_unroll_factor_a,b
        # 'n_mkloads', 'n_knloads',  # constant value

        # small, medium: resource occupation
        'ru_smallmed_unroll_factor_c', 'ru_smallmed_loop_matmul',
        'ru_smallmed_max_parallel_work', 'ru_smallmed_buf_size',
        'ru_smallmed_smem_per_block', 'ru_smallmed_regs_per_thread',

        # small, medium, large: resource occupation
        'ru_smallmedlarge_cmax', 'ru_smallmedlarge_rmax',
        'ru_smallmedlarge_T', 'ru_smallmedlarge_min_threads',

        # Kothapalli
        'Koth_med_Nmem', 'Koth_med_perf_K'

    ],
    'largeDB1': [

        # largeDB: resource occupation
        'ru_large_Pa', 'ru_large_Pb',
        'ru_large_unroll_factor_a', 'ru_large_unroll_factor_b', 'ru_large_unroll_factor_c',
        'ru_large_loop_matmul', 'ru_large_max_concurrent_work',
        'ru_large_regs_per_thread', 'ru_large_n_DB_iter', 'ru_large_buf_size', 'ru_large_smem_per_block',

        # small, medium, large: resource occupation
        'ru_smallmedlarge_cmax', 'ru_smallmedlarge_rmax',
        'ru_smallmedlarge_T', 'ru_smallmedlarge_min_threads',

        # Kothapalli
        'Koth_large_Nmem', 'Koth_large_perf_K'

    ],
    'largeDB2': [

        # largeDB: resource occupation
        'ru_large_Pa', 'ru_large_Pb',
        'ru_large_unroll_factor_a', 'ru_large_unroll_factor_b', 'ru_large_unroll_factor_c',
        'ru_large_loop_matmul', 'ru_large_max_concurrent_work',
        'ru_large_regs_per_thread', 'ru_large_n_DB_iter', 'ru_large_buf_size', 'ru_large_smem_per_block',

        # small, medium, large: resource occupation
        'ru_smallmedlarge_cmax', 'ru_smallmedlarge_rmax',
        'ru_smallmedlarge_T', 'ru_smallmedlarge_min_threads',

        # Kothapalli
        'Koth_large_Nmem', 'Koth_large_perf_K'

    ]
}
derived_parameters_withcompileinfo = {
    'common': derived_parameters['common'] + [
        'nblocks_per_sm_lim_blks_warps',
        'nblocks_per_sm_lim_reg',
        'smem_per_block',
        'nblocks_per_sm_lim_smem',
        'nblks_per_sm',
        'nwarps_per_sm',
        'nsm', 'ngpu', 'occupancy'
    ],
    'tiny': [p for p in derived_parameters['tiny']
             if p not in ('ru_tiny_smem_per_block',  # the smem estimation is correct for algo 'tiny'
                          'ru_tiny_nblks_per_sm')],    # equal to nblocks_per_sm_lim_blks_warps
    'small': derived_parameters['small'],
    'medium': derived_parameters['medium'],
    'largeDB1': derived_parameters['largeDB1'],
    'largeDB2': derived_parameters['largeDB2']
}


# ===============================================================================
def get_max_performances_per_mnk(data):
    """
    Construct dictionary:
        keys: (m, n, k)-tuple,
        values: maximum performance found over all algorithms for this given (m, n, k)
    """
    # Get list of different (m, n, k)s occurring in this instance
    data['mnk'] = list(zip(data['m'], data['n'], data['k']))
    mnks = np.unique(data['mnk'])

    # Get max. performance per (m, n, k)
    max_perf = dict()

    for mnk in mnks:
        # Get indices corresponding to this mnk
        idx_mnk = np.where(data['mnk'] == mnk)[0].tolist()

        # Get performances per mnk
        perf_mnk_algo = data['perf (Gflop/s)'].values[idx_mnk]

        # Store maxperf
        maxperf = float(perf_mnk_algo.max(axis=0))  # max. performance found through autotuning
        max_perf[mnk] = maxperf

    return max_perf


# ===============================================================================
def get_baseline_performances_per_mnk(data, algorithm):
    """
    Construct dictionary:
        keys: (m, n, k)-tuple,
        values: baseline performance for this given (m, n, k) and the given algorithm
    """
    from predict_helpers import baseline

    # Get list of different (m, n, k)s occurring in this instance
    data['mnk'] = list(zip(data['m'], data['n'], data['k']))
    mnks = np.unique(data['mnk'])

    # Get baseline performance per (m, n, k)
    baseline_perf = dict()

    for mnk in mnks:
        m, n, k = mnk
        baseline_pars = baseline(m, n, k, algorithm)

        if np.isnan(baseline_pars['tile_m']):
            idx_baseline = data[
                (data.m == baseline_pars['m']) &
                (data.n == baseline_pars['n']) &
                (data.k == baseline_pars['k']) &
                (data.threads_per_blk == baseline_pars['threads']) &
                (data.grouping == baseline_pars['grouping']) &
                (data.minblocks == baseline_pars['minblocks'])
            ].index.tolist()
        elif np.isnan(baseline_pars['w']):
            idx_baseline = data[
                (data.m == baseline_pars['m']) &
                (data.n == baseline_pars['n']) &
                (data.k == baseline_pars['k']) &
                (data.threads_per_blk == baseline_pars['threads']) &
                (data.grouping == baseline_pars['grouping']) &
                (data.minblocks == baseline_pars['minblocks']) &
                (data.tile_m == baseline_pars['tile_m']) &
                (data.tile_n == baseline_pars['tile_n'])
            ].index.tolist()
        else:
            idx_baseline = data[
                (data.m == baseline_pars['m']) &
                (data.n == baseline_pars['n']) &
                (data.k == baseline_pars['k']) &
                (data.threads_per_blk == baseline_pars['threads']) &
                (data.grouping == baseline_pars['grouping']) &
                (data.minblocks == baseline_pars['minblocks']) &
                (data.tile_m == baseline_pars['tile_m']) &
                (data.tile_n == baseline_pars['tile_n']) &
                (data.tile_m == baseline_pars['w']) &
                (data.tile_n == baseline_pars['v'])
            ].index.tolist()

        assert len(idx_baseline) == 1
        idx_baseline = idx_baseline[0]

        baseline_perf[mnk] = data['perf (Gflop/s)'][idx_baseline]

    return baseline_perf


# ===============================================================================
class PredictiveParameters:

    def __init__(self, params_df, gpu, autotuning, max_performances, partial_initialization=False):
        """
        params_df: pandas Dataframe where each row corresponds to a kernel parameter set
        """
        assert "m" in params_df.columns.values
        assert "n" in params_df.columns.values
        assert "k" in params_df.columns.values
        self.gpu = gpu                              # GPU card properties
        self.autotuning = autotuning                # autotuning properties
        self.max_performances = max_performances    # dictionary of max. performances
                                                    # keys: (m, n, k)-tuple, values: maximum performance
                                                    # found over all algorithms for this given (m, n, k)
        self.atomicAdd_factor = 5

        if not partial_initialization:
            assert "threads_per_blk" in params_df.columns.values
            assert "grouping" in params_df.columns.values
            assert "minblocks" in params_df.columns.values
            algos = np.unique(params_df["algorithm"].values)
            assert len(algos) == 1
            algo = algos[0]
            if algo in ['small', 'medium', 'largeDB1', 'largeDB2']:
                assert "tile_m" in params_df.columns.values
                assert "tile_n" in params_df.columns.values
                if algo in ['largeDB1', 'largeDB2']:
                    assert "w" in params_df.columns.values
                    assert "v" in params_df.columns.values

            # Possible additional fields, if compilation information is available:
            # 'nbytes_smem', 'regs_per_thread', nytes_cmem

        self.params = params_df

    def get(self, feature_name):
        """Generic function to compute any feature given by name"""

        if feature_name not in self.params.columns.values:
            if feature_name not in ['perf_scaled', 'perf_scaled_by_algo']:  # not vectorizable
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
    def get_perf_squared(self):
        return self.get('perf (Gflop/s)') * self.get('perf (Gflop/s)')

    def get_perf_scaled_by_algo(self):
        """
        Scale raw performances in [Gflop/s] between 0 and 1, where
            0 = 0 Gflop/s
            1 = performance equal to autotuned maximum FOR THIS SPECIFIC ALGORITHM
        :return: numpy array of scaled performances
        """
        # This function is written with the assumption that all parameter sets in this object instance have the same
        # algorithm
        assert len(np.unique(self.get('algorithm'))) == 1, "More than one type of algorithm found"

        # Get list of different (m, n, k)s occurring in this instance
        mnks = np.unique(self.get('mnk'))

        # Get max. performance per (m, n, k), per algorithm
        autotuned_max = dict()

        for mnk in mnks:

            # Get indices corresponding to this mnk
            #idx_mnk = np.where(self.get('mnk') == mnk)[0].tolist()
            m, n, k = mnk
            blob = np.where((self.get('m') == m) & (self.get('n') == n) & (self.get('k') == k))
            idx_mnk_ = blob[0]
            idx_mnk = idx_mnk_.tolist()

            # Get performances per mnk
            perf_mnk_algo = self.get('perf (Gflop/s)')[idx_mnk]

            # Store maxperf
            maxperf = float(perf_mnk_algo.max(axis=0))  # max. performance found through autotuning
            autotuned_max[mnk] = maxperf

        # Scale performances
        def scale_perf(perf, mnk):
            """For a given mnk and a given performance on this mnk, return the scaled performance"""
            return perf / autotuned_max[mnk]

        vec_scale_perf = np.vectorize(scale_perf)
        ret = vec_scale_perf(self.get('perf (Gflop/s)'), self.get('mnk'))
        return ret

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
        ret = vec_scale_perf(self.get('perf (Gflop/s)'), self.get('mnk'))
        return ret

    # ===============================================================================
    # Matrix sizes
    def get_size_a(self):
        """Size of matrix A (first operand of A * B = C)"""
        return self.get('m') * self.get('k')

    def get_size_b(self):
        """Size of matrix B (second operand of A * B = C)"""
        return self.get('k') * self.get('n')

    def get_size_c(self):
        """Size of matrix B (result of of A * B = C)"""
        return self.get('m') * self.get('n')

    def get_mnk(self):
        """Return (m, n, k) as a tuple"""
        return self.get('m'), self.get('n'), self.get('k')

    def get_mxnxk(self):
        """Return the product m*n*k"""
        return self.get('m') * self.get('n') * self.get('k')

    # ===============================================================================
    # Launch parameters
    def get_need_sync(self):
        """(mn > warp_size || mk > warp_size || kn > warp_size || threads > warp_size)"""
        return np.where(self.get('size_c') > self.gpu['Threads_/_Warp'], True, False) \
               | np.where(self.get('size_a') > self.gpu['Threads_/_Warp'], True, False) \
               | np.where(self.get('size_b') > self.gpu['Threads_/_Warp'], True, False) \
               | np.where(self.get('threads_per_blk') > self.gpu['Threads_/_Warp'], True, False)

    def get_nblks(self):
        return np.ceil(self.autotuning['stack_size'] / self.get('grouping'))

    def get_warps_per_blk(self):
        return np.ceil(self.get('threads_per_blk') / self.gpu['Threads_/_Warp'])

    def get_nwarps(self):
        return self.get('warps_per_blk') * self.get('nblks')

    def get_sm_desired(self):
        return np.ceil(self.get('nblks') / self.get('minblocks'))

    def get_nthreads(self):
        return self.get('threads_per_blk') * self.get('nblks')

    # ===============================================================================
    # Resource occupancy estimations
    # Note: these features need compilation information: nbytes of shared memory used and number of registers used
    def get_nblocks_per_sm_lim_blks_warps(self):
        """Resource occupations in terms of warps and blocks (Follows CUDA calculator sheet)"""
        return np.minimum(self.gpu['Thread_Blocks_/_Multiprocessor'],
                          np.floor(self.gpu['Warps_/_Multiprocessor'] / self.get('warps_per_blk')))

    def get_nblocks_per_sm_lim_reg(self):
        """Resource occupations in terms of warps and blocks (Follows CUDA calculator sheet)"""
        intermediate1 = round_down_to_nearest_multiple(
            self.gpu['Max_Registers_/_Block'] / round_up_to_nearest_multiple(
                self.get('regs_per_thread') * self.gpu['Threads_/_Warp'],
                self.gpu['Register_Allocation_Unit_Size']),
            self.gpu['Warp_Allocation_Granularity'])
        intermediate2 = np.floor(intermediate1 / self.get('warps_per_blk')) * \
                        np.floor(self.gpu['Register_File_Size_/_Multiprocessor_(32-bit_registers)'] /
                                 self.gpu['Max_Registers_/_Block'])
        return np.where(intermediate2 != 0, intermediate2, 1)

    def get_smem_per_block(self):
        """Resource occupations in terms of shared memory (Follows CUDA calculator sheet)"""
        return round_up_to_nearest_multiple(self.get('nbytes_smem'), self.gpu['Shared_Memory_Allocation_Unit_Size'])

    def get_nblocks_per_sm_lim_smem(self):
        return np.floor(self.gpu['Shared_Memory_/_Multiprocessor_(bytes)'] / self.get('smem_per_block'))

    def get_nblks_per_sm(self):
        return np.minimum.reduce([self.get('nblocks_per_sm_lim_blks_warps'),
                                  self.get('nblocks_per_sm_lim_reg'),
                                  self.get('nblocks_per_sm_lim_smem')])

    def get_nwarps_per_sm(self):
        return self.get('nblks_per_sm') * self.get('warps_per_blk')

    def get_nsm(self):
        return np.ceil(self.get('nblks') / self.get('nblks_per_sm'))

    def get_ngpu(self):
        return np.ceil(self.get('nsm') / self.gpu['Multiprocessors'])

    def get_occupancy(self):
        return self.get('nwarps_per_sm') / self.gpu['Warps_/_Multiprocessor']

    # ===============================================================================
    # Resource usage (common)
    def get_ru_param_stack_unroll_factor(self):
        """Loop counts"""
        return np.ceil(self.get('grouping') / self.get('threads_per_blk'))

    def get_n_iter(self):
        return np.maximum(3, 12500 * (1 // (self.get('m') * self.get('n') * self.get('k'))))

    def get_Gflops(self):
        return self.get('n_iter') * self.autotuning['stack_size'] * self.get('m') * self.get('n') * self.get('k') * 2 * 10**(-9)

    # ===============================================================================
    # Resource usage (tiny, small, medium)
    def get_ru_tinysmallmed_unroll_factor_a(self):
        """loop unroll factor of the loop on m*n"""
        return np.ceil(self.get('size_a') / self.get('threads_per_blk'))

    def get_ru_tinysmallmed_unroll_factor_b(self):
        """loop unroll factor of the loop on k*m"""
        return np.ceil(self.get('size_b') / self.get('threads_per_blk'))

    def get_ru_tinysmallmed_unroll_factor_a_total(self):
        """loop unroll factor multplied by number of times the loop is run"""
        return self.get('ru_tinysmallmed_unroll_factor_a') * self.get('grouping')

    def get_ru_tinysmallmed_unroll_factor_b_total(self):
        """loop unroll factor multplied by number of times the loop is run"""
        return self.get('ru_tinysmallmed_unroll_factor_b') * self.get('grouping')

    def get_ru_tinysmallmed_unroll_factor_c_total(self):
        return self.get('k') * self.get('grouping')

    # ===============================================================================
    # Resource usage (tiny)
    def get_ru_tiny_max_parallel_work(self):
        """Total number of iterations in each loop"""
        return np.maximum.reduce([self.get('grouping'), self.get('size_a'), self.get('size_b'), self.get('size_c')])

    def get_ru_tiny_min_threads(self):
        """Minimum number of threads required to run the kernel and produce correct results"""
        return self.get('size_c')

    def get_ru_tiny_buf_size(self):
        return self.get('k') * (self.get('m') + self.get('n'))

    def get_ru_tiny_smem_per_block(self):
        return (self.get('ru_tiny_buf_size') * self.autotuning['sizeof_double']) + (
            self.autotuning['npars'] * self.get('grouping') * self.autotuning['sizeof_int'])

    def get_ru_tiny_nblks_per_sm(self):
        """Occupancy estimation: assumption (verified on a sample of mnks): nblks is always limited by number of threads
        for algorithm tiny"""
        return self.get('nblocks_per_sm_lim_blks_warps')

    def get_ru_tiny_nwarps_per_sm(self):
        return self.get('ru_tiny_nblks_per_sm') * self.get('warps_per_blk')

    def get_ru_tiny_nsm(self):
        return np.ceil(self.get('nblks') / self.get('ru_tiny_nblks_per_sm'))

    def get_ru_tiny_ngpu(self):
        return np.ceil(self.get('ru_tiny_nsm') / self.gpu['Multiprocessors'])

    def get_ru_tiny_occupancy(self):
        return self.get('ru_tiny_nwarps_per_sm') / self.gpu['Warps_/_Multiprocessor']

    # ===============================================================================
    # Resource usage (small, medium, large)
    def get_ru_smallmedlarge_cmax(self):
        return np.ceil(self.get('n') / self.get('tile_n'))

    def get_ru_smallmedlarge_rmax(self):
        return np.ceil(self.get('m') / self.get('tile_m'))

    def get_ru_smallmedlarge_T(self):
        return self.get('tile_m') * self.get('tile_n')

    def get_ru_smallmedlarge_min_threads(self):
        return self.get('ru_smallmedlarge_cmax') * self.get('ru_smallmedlarge_rmax')

    # ===============================================================================
    # Resource usage estimation and loop counts (small, medium)
    def get_ru_smallmed_tm_max(self):
        return self.get('m')

    def get_ru_smallmed_tn_max(self):
        return self.get('n')

    def get_ru_smallmed_unroll_factor_c(self):
        return np.ceil(self.get('size_c') / self.get('threads_per_blk'))

    def get_ru_smallmed_loop_matmul(self):
        return self.get('k') * self.get('tile_m') * self.get('tile_n')

    def get_ru_smallmed_max_parallel_work(self):
        return np.maximum.reduce([self.get('grouping'), self.get('size_a'), self.get('size_b'), self.get('size_c'),
                                  self.get('ru_smallmedlarge_min_threads')])

    def get_ru_smallmed_buf_size(self):
        intermediate1 = self.get('size_a') + self.get('k') * self.get('tile_n') * self.get('ru_smallmedlarge_cmax')
        intermediate2 = self.get('tile_m') * self.get('ru_smallmedlarge_rmax') * self.get('k') + 1
        return np.maximum.reduce([self.get('size_c'), intermediate1, intermediate2])

    def get_ru_smallmed_smem_per_block(self):
        return (self.get('ru_smallmed_buf_size') * self.autotuning['sizeof_double']) + (
                self.autotuning['npars'] * self.get('grouping') * self.autotuning['sizeof_int'])

    def get_ru_smallmed_regs_per_thread(self):
        return self.get('tile_m') * self.get('tile_n') + (self.get('m') * self.get('k') +
                                                          self.get('threads_per_blk') - 1) // \
               self.get('threads_per_blk') + (self.get('k') * self.get('n') + self.get('threads_per_blk') - 1) // \
               self.get('threads_per_blk')

    # ===============================================================================
    # Resource usage (medium)
    # Loop bounds
    def get_load_unroll_factor_1(self):
        return self.get('size_a') // self.get('threads_per_blk') + 1

    def get_load_unroll_factor_2(self):
        return self.get('size_b') // self.get('threads_per_blk') + 1

    def get_n_mkloads(self):
        return self.get('size_a') // (self.get('load_unroll_factor_1') * self.get('threads_per_blk'))

    def get_n_knloads(self):
        return self.get('size_b') // (self.get('load_unroll_factor_2') * self.get('threads_per_blk'))

    # ===============================================================================
    # Resource usage (large)
    def get_ru_large_Pa(self):
        """input slab size"""
        return self.get('m') * self.get('w')

    def get_ru_large_Pb(self):
        """input slab size"""
        return self.get('w') * self.get('n')

    def get_ru_large_Pc(self):
        """output slab size"""
        return self.get('m') * self.get('v')

    def get_ru_large_unroll_factor_a(self):
        return np.ceil(self.get('ru_large_Pa') / self.get('threads_per_blk'))

    def get_ru_large_unroll_factor_b(self):
        return np.ceil(self.get('ru_large_Pb') / self.get('threads_per_blk'))

    def get_ru_large_unroll_factor_c(self):
        return np.ceil(self.get('ru_large_Pc') / self.get('threads_per_blk'))

    def get_ru_large_loop_matmul(self):
        return self.get('w') * self.get('tile_m') * self.get('tile_n')

    def get_ru_large_max_concurrent_work(self):
        return np.maximum.reduce([self.get('grouping'), self.get('ru_large_Pa'), self.get('ru_large_Pb'),
                                  self.get('ru_large_Pc'), self.get('ru_smallmedlarge_T')])

    def get_ru_large_regs_per_thread(self):
        return self.get('tile_m') * self.get('tile_n') + \
               (self.get('w') * self.get('m') + self.get('threads_per_blk') - 1) // self.get('threads_per_blk') + \
               (self.get('w') * self.get('n') + self.get('threads_per_blk') - 1) // self.get('threads_per_blk')

    def get_ru_large_n_DB_iter(self):
        """Number of double-buffering iterations"""
        return self.get('k') // (2 * self.get('w'))

    def get_ru_large_buf_size(self):
        intermediate1 = (self.get('w') - 1) * self.get('m') + self.get('ru_smallmedlarge_rmax') * self.get('tile_m')
        intermediate2 = self.get('m') * self.get('w') + (self.get('w') - 1) * self.get('n') + \
                        self.get('ru_smallmedlarge_cmax') * self.get('tile_n')
        return np.maximum.reduce([self.get('ru_large_Pc'), intermediate1, intermediate2])

    def get_ru_large_smem_per_block(self):
        return self.get('ru_large_buf_size') * self.autotuning['sizeof_double'] + \
               self.autotuning['npars'] * self.get('grouping') * self.autotuning['sizeof_int']

    # ===============================================================================
    # Kothapalli et al. metrics
    def kothapalli_nmem(self, nmem_glob, nmem_shared):
        return self.gpu['Global_memory_access_latency'] * nmem_glob + \
               self.gpu['Shared_memory_access_latency'] * nmem_shared

    def kothapalli_perf(self, n_mem, nblks, threads_per_blk, gflops):
        c_K = nblks * threads_per_blk * n_mem  # ignore number of threads per warp
        return gflops / c_K  # ignore clock rate (constant factor)

    # ===============================================================================
    # Kothapalli et al. metrics (for 'tiny')
    def get_Koth_tiny_Nmem_shared(self):
        """Kothapalli et al. modelling, under communication-bound assumption"""
        return 3 * self.get('grouping') + self.get('grouping') * (
                3 + self.get('ru_tinysmallmed_unroll_factor_a') +
                self.get('ru_tinysmallmed_unroll_factor_b') + 2 * self.get('k'))

    def get_Koth_tiny_Nmem_glob(self):
        return 3 * self.get('grouping') + self.get('grouping') * (
                self.get('ru_tinysmallmed_unroll_factor_a') + self.get('ru_tinysmallmed_unroll_factor_b'))

    def get_Koth_tiny_Nmem(self):
        return self.kothapalli_nmem(self.get('Koth_tiny_Nmem_glob'), self.get('Koth_tiny_Nmem_shared'))

    def get_Koth_tiny_perf_K(self):
        return self.kothapalli_perf(self.get('Koth_tiny_Nmem'), self.get('nblks'), self.get('threads_per_blk'),
                                    self.get('Gflops'))

    # ===============================================================================
    # Kothapalli et al. metrics (for 'small')
    def get_Koth_small_Nmem_shared(self):
        """Kothapalli et al. modelling, under communication-bound assumption"""
        i = 3 * self.get('grouping') + self.get('grouping') * (
                3 + self.get('ru_tinysmallmed_unroll_factor_a') +
                self.get('ru_tinysmallmed_unroll_factor_b') +
                2 * self.get('k') * self.get('tile_m') * self.get('tile_n') +
                self.get('tile_m') * self.get('tile_n'))

        return np.where(np.logical_and(self.get('tile_m') > 1, self.get('tile_n') > 1),
                        i + self.atomicAdd_factor * self.get('ru_smallmed_unroll_factor_c'), i)

    def get_Koth_small_Nmem_glob(self):
        return 3 * self.get('grouping') + self.get('grouping') * (
                self.get('ru_tinysmallmed_unroll_factor_a') + self.get('ru_tinysmallmed_unroll_factor_b') +
                self.atomicAdd_factor * self.get('ru_smallmed_unroll_factor_c'))

    def get_Koth_small_Nmem(self):
        return self.kothapalli_nmem(self.get('Koth_small_Nmem_glob'), self.get('Koth_small_Nmem_shared'))

    def get_Koth_small_perf_K(self):
        return self.kothapalli_perf(self.get('Koth_small_Nmem'), self.get('nblks'), self.get('threads_per_blk'),
                                    self.get('Gflops'))

    # ===============================================================================
    # Kothapalli et al. metrics (for 'medium')
    def get_Koth_med_Nmem_shared(self):
        """Kothapalli et al. modelling, under communication-bound assumption"""
        i = 3 * self.get('grouping') + self.get('grouping') * (
                2 + self.get('n_mkloads') * self.get('load_unroll_factor_1') +
                self.get('n_knloads') * self.get('load_unroll_factor_2') +
                2 + 2 * self.get('k') * self.get('tile_m') * self.get('tile_n') + 1)

        return np.where((self.get('tile_m') > 1) & (self.get('tile_n') > 1),
                        i + self.atomicAdd_factor * self.get('ru_smallmed_unroll_factor_c'), i)

    def get_Koth_med_Nmem_glob(self):
        return 3 * self.get('grouping') + self.get('grouping') * (
                2 * self.get('n_mkloads') * self.get('load_unroll_factor_1') +
                self.get('n_knloads') * self.get('load_unroll_factor_2') +
                self.atomicAdd_factor * self.get('ru_smallmed_unroll_factor_c'))

    def get_Koth_med_Nmem(self):
        return self.kothapalli_nmem(self.get('Koth_med_Nmem_glob'), self.get('Koth_med_Nmem_shared'))

    def get_Koth_med_perf_K(self):
        return self.kothapalli_perf(self.get('Koth_med_Nmem'), self.get('nblks'), self.get('threads_per_blk'),
                                    self.get('Gflops'))

    # ===============================================================================
    # Kothapalli et al. metrics (for 'largeDB')
    def get_Koth_large_Nmem_shared(self):
        """Kothapalli et al. modelling, under communication-bound assumption"""
        return 3 * self.get('grouping') + self.get('grouping') * (
                (self.get('m') * self.get('w')) // self.get('threads_per_blk') +
                (self.get('n') * self.get('w')) // self.get('threads_per_blk') +  # load_gmem_into_smem
                (self.get('k') // self.get('w')) * (
                        (self.get('m') * self.get('w')) // self.get('threads_per_blk') +
                        (self.get('n') * self.get('w')) // self.get('threads_per_blk') +  # load_regs_into_smem
                        3 * self.get('w') * self.get('tile_m') * self.get('tile_n')  # multiply
                ) +  # double-buffering
                (self.get('n') // self.get('v')) * (
                    self.get('tile_m') * self.get('tile_n') +  # store_results_into_smem
                    self.get('ru_large_Pc') // self.get('threads_per_blk')  # result accumulation
                )  # write out
        )

    def get_Koth_large_Nmem_glob(self):
        return 3 * self.get('grouping') + self.get('grouping') * (
                2 +
                (self.get('m') * self.get('w')) // self.get('threads_per_blk') +
                (self.get('n') * self.get('w')) // self.get('threads_per_blk') +  # load_gmem_into_smem
                (self.get('k') // self.get('w')) * (
                        (self.get('m') * self.get('w')) // self.get('threads_per_blk') +
                        (self.get('n') * self.get('w')) // self.get('threads_per_blk')  # load_gmem_into_regs
                ) +  # double-buffering
                (self.get('n') // self.get('v')) * (
                    # result accumulation
                    self.atomicAdd_factor * (self.get('ru_large_Pc') // self.get('threads_per_blk'))
                )  # write out
        )

    def get_Koth_large_Nmem(self):
        return self.kothapalli_nmem(self.get('Koth_large_Nmem_glob'), self.get('Koth_large_Nmem_shared'))

    def get_Koth_large_perf_K(self):
        return self.kothapalli_perf(self.get('Koth_large_Nmem'), self.get('nblks'), self.get('threads_per_blk'),
                                    self.get('Gflops'))
