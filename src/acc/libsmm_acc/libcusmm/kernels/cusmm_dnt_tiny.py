# -*- coding: utf-8 -*-
from kernels import cusmm_dnt
import cusmm_common as cu


class Kernel_dnt_tiny(cusmm_dnt.Kernel):

    algorithm = "tiny"
    launch_parameters = ['m', 'n', 'k', 'threads', 'grouping', 'minblocks']

    def __init__(self, *, m, n, k, threads, grouping, minblocks, perf):
        self.m = m
        self.n = n
        self.k = k
        self.threads = threads
        self.grouping = grouping
        self.minblocks = minblocks
        self.perf = perf
        assert self.m * self.n <= self.threads

    @property
    def func_signature(self):
        return "cusmm_dnt_tiny<%(m)d,%(n)d,%(k)d,%(threads)d,%(grouping)d,%(minblocks)d>;\n" % self.__dict__

    @staticmethod
    def promising_parameters(m, n, k, gpu):

        # Shared memory buffer size
        buf_sz = k * (m + n)   # number of elements in the a_block buffer = mk, and in the b_block buffer = kn

        # Minimum number of threads required to cover the result matrix c
        min_threads = m*n 

        # Parameter space: 
        params = []
        for minblocks in range(1, gpu["maxBLOCKSperSM"] + 1):
            for grouping in range(2, 32 + 1, 1):  # heuristic: never seen optimal=1 hence start from 2
            
                # Max work ("operations")  which can be run concurrently
                max_concurrent_work = max(grouping, m*k, k*n, m*n)

                # Shared memory utilisation (bytes)
                smem_tot = buf_sz * cu.sizeof_double + cu.npar * grouping * cu.sizeof_int
                if smem_tot > gpu["SMEMperBLOCK"]:
                    continue
                if smem_tot * minblocks > gpu["SMEMperSM"]:
                    continue

                # Use all concurrency available: fill warps
                for threads in range(gpu["warp_size"], gpu["maxTHREADSperBLOCK"] + 1, gpu["warp_size"]):

                    if threads > cu.round_up_to_multiple(max_concurrent_work, gpu["warp_size"]): 
                        continue  # soft: too much concurrency harms performance
                    if threads * minblocks > gpu["maxTHREADSperSM"]:
                        continue
                    if threads < min_threads: 
                        continue

                    params.append({'m': m, 'n': n, 'k': k,
                                   'threads': threads,
                                   'grouping': grouping,
                                   'minblocks': minblocks})
        return params

#EOF
