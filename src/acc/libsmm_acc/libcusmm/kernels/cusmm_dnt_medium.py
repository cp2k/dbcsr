# -*- coding: utf-8 -*-
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

from kernels import cusmm_dnt as cu


class Kernel_dnt_medium(cu.Kernel):

    algorithm = 'medium'
    algorithm_num = 3
    launch_parameters = ['m', 'n', 'k', 'tile_m', 'tile_n', 'threads', 'grouping', 'minblocks']

    def __init__(self, m, n, k, threads, tile_m, tile_n, grouping, minblocks, perf, source):
        self.m = m
        self.n = n
        self.k = k
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.threads = threads
        self.grouping = grouping
        self.minblocks = minblocks
        self.perf = perf
        self.source = source
        assert self.threads * self.minblocks <= 2048
        min_threads = ((self.m + self.tile_m - 1) // self.tile_m) * ((self.n + self.tile_n - 1) // self.tile_n)
        assert min_threads <= self.threads

    @property
    def func_signature(self):
        return "cusmm_dnt_medium<%(m)d,%(n)d,%(k)d,%(tile_m)d,%(tile_n)d,%(threads)d,%(grouping)d,%(minblocks)d>;\n" \
               % self.__dict__

    @staticmethod
    def promising_parameters(m, n, k, gpu, autotuning):
        params = []
        for minblocks in range(1, 28):  # for exhaustive search: range(1, gpu["Thread_Blocks_/_Multiprocessor"] + 1):
                                        # heuristic: the optimal minblocks is never > 28
            if m >= 28:  # heuristic: investigate a smaller search space of grouping for large matrices
                grouping_range = (3, 4, 5, 24, 26, 29, 32,)
            else: 
                grouping_range = range(1, 32 + 1, 1)
            for grouping in grouping_range:
                for tm in range(1, min(12, m) + 1):  # heuristic: the optimal tile_m is never above 12
                    for tn in range(1, min(12, n) + 1):  # heuristic: the optimal tile_m is never above 12

                        if tm * tn > 16:
                            continue  # heuristic: performance decreases for very large tiles

                        # Number of tiled columns, rows
                        cmax = (n + tn - 1) // tn
                        rmax = (m + tm - 1) // tm

                        # Max work ("operations") which can be run concurrently
                        max_concurrent_work = max(grouping, m*k, k*n, m*n, cmax*rmax)

                        # Minimum number of threads required to have one thread per tile
                        # i.e., cover the result matrix
                        min_threads = cmax * rmax

                        # Shared memory buffer size
                        buf_sz = max(m*n, m*k + k*tn*cmax, tm*rmax*k + 1)
                        smem_tot = buf_sz * autotuning["sizeof_double"] + autotuning["npar"] * grouping * autotuning["sizeof_int"]
                        if smem_tot > gpu["Max_Shared_Memory_/_Block_(bytes)"]:
                            continue
                        if smem_tot * minblocks > gpu["Shared_Memory_/_Multiprocessor_(bytes)"]:
                            continue

                        # Use all concurrency available: fill warps
                        for threads in range(gpu["Threads_/_Warp"], gpu["Max_Thread_Block_Size"] + 1, gpu["Threads_/_Warp"]):

                            if threads > cu.round_up_to_multiple(max_concurrent_work, gpu["Threads_/_Warp"]):
                                continue  # soft: too much concurrency harms performance
                            if threads * minblocks > gpu["Threads_/_Multiprocessor"]:
                                continue
                            if threads < min_threads:
                                continue

                            params.append({'m': m, 'n': n, 'k': k,
                                           'tile_m': tm, 'tile_n': tn,
                                           'threads': threads,
                                           'grouping': grouping,
                                           'minblocks': minblocks,
                                           'perf': 0})
        return params
