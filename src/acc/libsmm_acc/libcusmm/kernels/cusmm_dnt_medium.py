# -*- coding: utf-8 -*-
from kernels import cusmm_dnt as cu


class Kernel_dnt_medium(cu.Kernel):

    algorithm = 'medium'
    algorithm_num = 3
    launch_parameters = ['m', 'n', 'k', 'tile_m', 'tile_n', 'threads', 'grouping', 'minblocks']

    def __init__(self, m, n, k, threads, tile_m, tile_n, grouping, minblocks, perf):
        self.m = m
        self.n = n
        self.k = k
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.threads = threads
        self.grouping = grouping
        self.minblocks = minblocks
        self.perf = perf
        assert self.threads * self.minblocks <= 2048
        min_threads = ((self.m + self.tile_m - 1) // self.tile_m) * ((self.n + self.tile_n - 1) // self.tile_n)
        assert min_threads <= self.threads

    @property
    def func_signature(self):
        return "cusmm_dnt_medium<%(m)d,%(n)d,%(k)d,%(tile_m)d,%(tile_n)d,%(threads)d,%(grouping)d,%(minblocks)d>;\n" \
               % self.__dict__

    @staticmethod
    def promising_parameters(m, n, k, gpu):
        params = []
        for minblocks in range(1, 28):  # for exhaustive search: range(1, gpu.maxBLOCKSperSM + 1):
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
                                           'tile_m': tm, 'tile_n': tn,
                                           'threads': threads,
                                           'grouping': grouping,
                                           'minblocks': minblocks})
        return params


#EOF
