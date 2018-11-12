#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math


########################################################################################################################
# Computing helpers
########################################################################################################################
def ceiling(x, step):
    return np.where(x % step == 0, x, x + step - x % step)


def flooring(x, step):
    return np.where(x % step == 0, x, x - x % step)


def ceil_division(a, b):
    return (a + b - 1) // b


########################################################################################################################
# Class
########################################################################################################################
class PredictiveParameters:

    def __init__(self, params_df, gpu, autotuning, partial_initialization=False):
        assert "m" in params_df.columns.values
        assert "n" in params_df.columns.values
        assert "k" in params_df.columns.values
        self.gpu = gpu
        self.autotuning = autotuning
        self.atomicAdd_factor = 5

        if not partial_initialization:
            assert "threads" in params_df.columns.values
            params_df.rename(columns={'threads': 'threads_per_blk'}, inplace=True)
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
            # Possible additional fields, if compilatio information is available:
            # 'nbytes_smem', 'regs_per_thread'

        self.params = params_df

    def get(self, feature_name):
        if feature_name not in self.params.columns.values:
            vget = np.vectorize(getattr(self, "get_" + feature_name))
            feature_val = vget()
        else:
            feature_val = self.params[feature_name].values
        return feature_val

    def get_features(self, feature_names):
        """
        :param feature_names: names of features to compute
        :return: feature_values: list of feature values computed from raw parameters
        """
        for feat in feature_names:
            self.params[feat] = self.get(feat)
        return self.params[feature_names]

    ####################################################################################################################
    # Matrix sizes
    def get_size_a(self):
        return self.get('m') * self.get('k')

    def get_size_b(self):
        return self.get('k') * self.get('n')

    def get_size_c(self):
        return self.get('m') * self.get('n')

    def get_mnk(self):
        return self.get('m').astype(str) + 'x' + self.get('n').astype(str) + 'x' + self.get('k').astype(str)

    def get_mxnxk(self):
        return self.get('m') * self.get('n') * self.get('k')

    ####################################################################################################################
    # Launch parameters
    def get_need_sync(self):
        """(mn > warp_size || mk > warp_size || kn > warp_size || threads > warp_size)"""
        return np.where(self.get('size_c') > self.gpu['Threads_/_Warp'], True, False) \
               | np.where(self.get('size_a') > self.gpu['Threads_/_Warp'], True, False) \
               | np.where(self.get('size_b') > self.gpu['Threads_/_Warp'], True, False) \
               | np.where(self.get('threads_per_blk') > self.gpu['Threads_/_Warp'], True, False)

    def get_nblks(self):
        return ceil_division(self.autotuning['stack_size'], self.get('grouping'))

    def get_warps_per_blk(self):
        return ceil_division(self.get('threads_per_blk'), self.gpu['Threads_/_Warp'])

    def get_nwarps(self):
        return self.get('warps_per_blk') * self.get('nblks')

    def get_sm_desired(self):
        return ceil_division(self.get('nblks'), self.get('minblocks'))

    def get_nwarps_inv(self):
        return ceil_division(1., self.get('nwarps'))

    def get_nthreads(self):
        return self.get('threads_per_blk') * self.get('nblks')

    ####################################################################################################################
    # Resource occupancy estimations
    # Note: these features need compilation information: nbytes of shared memory used and number of registers used
    def get_nblocks_per_sm_lim_blks_warps(self):
        """Resource occupations in terms of warps and blocks (Follows CUDA calculator sheet)"""
        return np.minimum(self.gpu['Threads_/_Multiprocessor'],
                          np.floor(self.gpu['Warps_/_Multiprocessor'] / self.get('warps_per_blk')))

    def get_nblocks_per_sm_lim_reg(self):
        """Resource occupations in terms of warps and blocks (Follows CUDA calculator sheet)"""
        intermediate1 = flooring(
            self.gpu['Max_Registers_/_Block'] / ceiling(
                self.get('regs_per_thread') * self.gpu['Threads_/_Warp'],
                self.gpu['Register_Allocation_Unit_Size']),
            self.gpu['Warp_Allocation_Granularity'])
        intermediate2 = np.floor(intermediate1 / self.get('warps_per_blk')) * \
                        math.floor(self.gpu['Register_File_Size_/_Multiprocessor_(32-bit_registers)'] /
                                   self.gpu['Max_Registers_/_Block'])
        return intermediate2 if intermediate2 != 0 else 1

    def get_smem_per_block(self):
        """Resource occupations in terms of shared memory (Follows CUDA calculator sheet)"""
        return ceiling(self.get('nbytes_smem'), self.gpu['Shared_Memory_Allocation_Unit_Size'])

    def get_nblocks_per_sm_lim_smem(self):
        return np.floor(self.gpu['Shared_Memory_/_Multiprocessor_(bytes)'] / self.get('smem_per_block'), 1)

    def get_nblks_per_sm(self):
        return np.minimum(self.get('nblocks_per_sm_lim_blks_warps'),
                          self.get('nblocks_per_sm_lim_reg'),
                          self.get('nblocks_per_sm_lim_smem'))

    def get_nwarps_per_sm(self):
        return self.get('nblks_per_sm') * self.get('warps_per_blk')

    def get_nsm(self):
        return ceiling(self.get('nblks'), self.get('nblks_per_sm'))

    def get_ngpu(self):
        return ceiling(self.get('nsm'), self.gpu['Multiprocessors'])

    def get_occupancy(self):
        return self.get('nwarps_per_sm') / self.gpu['Warps_/_Multiprocessor']

    ####################################################################################################################
    # Resource usage (common)
    def get_ru_param_stack_unroll_factor(self):
        """Loop counts"""
        return ceil_division(self.get('grouping'), self.get('threads_per_blk'))

    def get_ru_param_stack_unroll_factor_inv(self):
        return ceil_division(1., self.get('ru_param_stack_unroll_factor'))

    def get_n_iter(self):
        return np.maximum(3, 12500 * (1 // (self.get('m') * self.get('n') * self.get('k'))))

    def get_Gflops(self):
        return self.get('n_iter') * self.autotuning['stack_size'] * self.get('m') * self.get('n') * self.get('k') * 2 * 10**(-9)

    ####################################################################################################################
    # Resource usage (tiny, small, medium)
    def get_ru_tinysmallmed_unroll_factor_a(self):
        return ceil_division(self.get('size_a'), self.get('threads_per_blk'))

    def get_ru_tinysmallmed_unroll_factor_a_inv(self):
        return ceil_division(1., self.get('ru_tinysmallmed_unroll_factor_a'))

    def get_ru_tinysmallmed_unroll_factor_b(self):
        return ceil_division(self.get('size_b'), self.get('threads_per_blk'))

    def get_ru_tinysmallmed_unroll_factor_b_inv(self):
        return ceil_division(1., self.get('ru_tinysmallmed_unroll_factor_b'))

    def get_ru_tinysmallmed_unroll_factor_a_total(self):
        return self.get('ru_tinysmallmed_unroll_factor_a') * self.get('grouping')

    def get_ru_tinysmallmed_unroll_factor_b_total(self):
        return self.get('ru_tinysmallmed_unroll_factor_b') * self.get('grouping')

    def get_ru_tinysmallmed_unroll_factor_c_total(self):
        return self.get('k') * self.get('grouping')

    ####################################################################################################################
    # Resource usage (tiny)
    def get_ru_tiny_max_parallel_work(self):
        return np.maximum.reduce([self.get('grouping'), self.get('size_a'), self.get('size_b'), self.get('size_c')])

    def get_ru_tiny_min_threads(self):
        return self.get('size_c')

    def get_ru_tiny_max_threads(self):
        return ceiling(self.get('ru_tiny_max_parallel_work'), self.gpu['Threads_/_Warp'])

    def get_ru_tiny_buf_size(self):
        return self.get('k') * (self.get('m') + self.get('n'))

    def get_ru_tiny_smem_per_block(self):
        return (self.get('ru_tiny_buf_size') * self.autotuning['sizeof_double']) + (
            self.autotuning['npars'] * self.get('grouping') * self.autotuning['sizeof_int'])

    def get_ru_tiny_nblks_per_sm(self):
        """Occupancy estimation: assumption (verified on a sample of mnks): nblks is always limited by number of threads"""
        return self.get('nblocks_per_sm_lim_blks_warps')

    def get_ru_tiny_nwarps_per_sm(self):
        return self.get('nblks_per_sm') * self.get('warps_per_blk')

    def get_ru_tiny_nsm(self):
        return ceiling(self.get('nblks'), self.get('nblks_per_sm'))

    def get_ru_tiny_ngpu(self):
        return ceiling(self.get('nsm'), self.gpu['Multiprocessors'])

    def get_ru_tiny_occupancy(self):
        return self.get('nwarps_per_sm') / self.gpu['Warps_/_Multiprocessor']

    ####################################################################################################################
    # Resource usage (small, medium, large)
    def get_ru_smallmedlarge_cmax(self):
        return ceil_division(self.get('n'), self.get('tile_n'))

    def get_ru_smallmedlarge_rmax(self):
        return ceil_division(self.get('m'), self.get('tile_m'))

    def get_ru_smallmedlarge_T(self):
        return self.get('tile_m') * self.get('tile_n')

    def get_ru_smallmedlarge_min_threads(self):
        return self.get('ru_smallmedlarge_cmax') * self.get('ru_smallmedlarge_rmax')

    ####################################################################################################################
    # Resource usage estimation and loop counts (small, medium)
    def get_ru_smallmed_tm_max(self):
        return self.get('m')

    def get_ru_smallmed_tn_max(self):
        return self.get('n')

    def get_ru_smallmed_unroll_factor_c(self):
        return ceil_division(self.get('size_c'), self.get('threads_per_blk'))

    def get_ru_smallmed_loop_matmul(self):
        return self.get('k') * self.get('tile_m') * self.get('tile_n')

    def get_ru_smallmed_max_parallel_work(self):
        return np.maximum.reduce([self.get('grouping'), self.get('size_a'), self.get('size_b'), self.get('size_c'),
                                  self.get('ru_smallmedlarge_min_threads')])

    def get_ru_smallmed_max_threads(self):
        return ceiling(self.get('ru_smallmed_max_parallel_work'), self.gpu['Threads_/_Warp'])

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

    ####################################################################################################################
    # Resource usage (medium)
    # Loop bounds
    def get_load_unroll_factor_1(self):
        return ceil_division(self.get('size_a'), self.get('threads_per_blk')) + 1

    def get_load_unroll_factor_2(self):
        return ceil_division(self.get('size_b'), self.get('threads_per_blk')) + 1

    def get_n_mkloads(self):
        return ceil_division(self.get('size_a'), (self.get('load_unroll_factor_1') * self.get('threads_per_blk')))

    def get_n_knloads(self):
        return ceil_division(self.get('size_b'), (self.get('load_unroll_factor_2') * self.get('threads_per_blk')))

    ####################################################################################################################
    # Resource usage (large)
    def get_ru_large_Pa(self):
        """input slab size"""
        return self.get('m') * self.get('w')

    def get_ru_large_Pb(self):
        """input slab size"""
        return self.get('w') * self.get('n')

    def get_ru_large_Pc(self):
        """input slab size"""
        return self.get('m') * self.get('v')  # Output slabs

    def get_ru_large_unroll_factor_a(self):
        return ceil_division(self.get('ru_large_Pa'), self.get('threads_per_blk'))

    def get_ru_large_unroll_factor_b(self):
        return ceil_division(self.get('ru_large_Pb'), self.get('threads_per_blk'))

    def get_ru_large_unroll_factor_c(self):
        return ceil_division(self.get('ru_large_Pc'), self.get('threads_per_blk'))

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
        return self.get('k') // (2 * self.get('w'))

    def get_ru_large_buf_size(self):
        intermediate1 = (self.get('w') - 1) * self.get('m') + self.get('ru_smallmedlarge_rmax') * self.get('tile_m')
        intermediate2 = self.get('m') * self.get('w') + (self.get('w') - 1) * self.get('n') + \
                        self.get('ru_smallmedlarge_cmax') * self.get('tile_n')
        return np.maximum.reduce([self.get('ru_large_Pc'), intermediate1, intermediate2])

    def get_ru_large_smem_per_block(self):
        return self.get('ru_large_buf_size') * self.autotuning['sizeof_double'] + \
               self.autotuning['npars'] * self.get('grouping') * self.autotuning['sizeof_int']

    ####################################################################################################################
    # Kothapalli et al. metrics
    def kothapalli_nmem(self, nmem_glob, nmem_shared):
        return self.gpu['Global_memory_access_latency'] * nmem_glob + \
               self.gpu['Shared_memory_access_latency'] * nmem_shared

    def kothapalli_perf(self, n_mem, nblks, threads_per_blk, gflops):
        c_K = nblks * threads_per_blk * n_mem  # ignore number of threads per warp
        return gflops / c_K  # ignore clock rate

    # tiny
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

    # small
    def get_Koth_small_Nmem_shared(self):
        """Kothapalli et al. modelling, under communication-bound assumption"""
        i = 3 * self.get('grouping') + self.get('grouping') * (
                3 + self.get('ru_tinysmallmed_unroll_factor_a') +
                self.get('ru_tinysmallmed_unroll_factor_b') +
                2 * self.get('k') * self.get('tile_m') * self.get('tile_n') +
                self.get('tile_m') * self.get('tile_n'))

        return np.where(np.logical_and(self.get('tile_m') > 1, self.get('tile_n') > 1),
                        i + self.atomicAdd_factor * self.get('ru_smallmed_unroll_factor_c'),
                        i)

    def get_Koth_small_Nmem_glob(self):
        return 3 * self.get('grouping') + self.get('grouping') * (
                self.get('ru_tinysmallmed_unroll_factor_a') + self.get('ru_tinysmallmed_unroll_factor_b') +
                self.atomicAdd_factor * self.get('ru_smallmed_unroll_factor_c'))

    def get_Koth_small_Nmem(self):
        return self.kothapalli_nmem(self.get('Koth_small_Nmem_glob'), self.get('Koth_small_Nmem_shared'))

    def get_Koth_small_perf_K(self):
        return self.kothapalli_perf(self.get('Koth_small_Nmem'), self.get('nblks'), self.get('threads_per_blk'),
                                    self.get('Gflops'))

    # medium
    def get_Koth_med_Nmem_shared(self):
        """Kothapalli et al. modelling, under communication-bound assumption"""
        i = 3 * self.get('grouping') + self.get('grouping') * (
                2 + self.get('n_mkloads') * self.get('load_unroll_factor_1') +
                self.get('n_knloads') * self.get('load_unroll_factor_2') +
                2 + 2 * self.get('k') * self.get('tile_m') * self.get('tile_n') + 1)
        return i + self.atomicAdd_factor * self.get('ru_smallmed_unroll_factor_c') \
            if self.get('tile_m') > 1 and self.get('tile_n') > 1 else i

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

    # largeDB
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
