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


class Kernel_dnt_tiny(cu.Kernel):

    algorithm = "tiny"
    algorithm_num = 5
    launch_parameters = ['m', 'n', 'k', 'threads', 'grouping', 'minblocks']

    def __init__(self, m, n, k, threads, grouping, minblocks, perf, source):
        self.m = m
        self.n = n
        self.k = k
        self.threads = threads
        self.grouping = grouping
        self.minblocks = minblocks
        self.perf = perf
        self.source = source
        assert self.m * self.n <= self.threads

    @property
    def func_signature(self):
        return "cusmm_dnt_tiny<%(m)d,%(n)d,%(k)d,%(threads)d,%(grouping)d,%(minblocks)d>;\n" % self.__dict__

    @staticmethod
    def promising_parameters(m, n, k, gpu, autotuning):

        # Shared memory buffer size
        buf_sz = k * (m + n)   # number of elements in the a_block buffer = mk, and in the b_block buffer = kn

        # Minimum number of threads required to cover the result matrix c
        min_threads = m*n 

        # Parameter space: 
        params = []
        for minblocks in range(1, gpu["Thread_Blocks_/_Multiprocessor"] + 1):
            for grouping in range(2, 32 + 1, 1):  # heuristic: never seen optimal=1 hence start from 2
            
                # Max work ("operations")  which can be run concurrently
                max_concurrent_work = max(grouping, m*k, k*n, m*n)

                # Shared memory utilisation (bytes)
                smem_tot = buf_sz * autotuning["sizeof_double"] + autotuning["npar"] * grouping * autotuning["sizeof_int"]
                if smem_tot > gpu["Max_Shared_Memory_/_Block_(bytes)"]:
                    continue
                if smem_tot * minblocks > gpu["Max_Shared_Memory_/_Block_(bytes)"]:
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
                                   'threads': threads,
                                   'grouping': grouping,
                                   'minblocks': minblocks,
                                   'perf': 0})
        return params
