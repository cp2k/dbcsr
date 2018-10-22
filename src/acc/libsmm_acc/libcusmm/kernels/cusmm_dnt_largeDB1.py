# -*- coding: utf-8 -*-
from kernels import cusmm_dnt as cu


class Kernel_dnt_largeDB1(cu.Kernel):

    algorithm = "largeDB1"
    algorithm_num = 1
    launch_parameters = ['m', 'n', 'k', 'tile_m', 'tile_n', 'w', 'v', 'threads', 'grouping', 'minblocks']

    def __init__(self, *, m, n, k, threads, tile_m, tile_n, w, v, grouping, minblocks, perf):
        self.m = m
        self.n = n
        self.k = k
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.w = w
        self.v = v
        self.threads = threads
        self.grouping = grouping
        self.minblocks = minblocks
        self.perf = perf
        assert self.threads * self.minblocks <= 2048
        min_threads = ((self.m+self.tile_m-1)//self.tile_m) * ((self.n+self.tile_n-1)//self.tile_n)
        assert min_threads <= self.threads
        assert self.tile_m <= self.v
        assert self.tile_n <= self.w

    @property
    def func_signature(self):
        return "cusmm_dnt_largeDB1<%(m)d,%(n)d,%(k)d,%(tile_m)d,%(tile_n)d,%(w)d,%(v)d,%(threads)d,%(grouping)d,%(minblocks)d>;\n" \
               % self.__dict__

    @staticmethod
    def promising_parameters(m, n, k, gpu):
        params = []
        grouping = 16

        for minblocks in (1, 2, 4, 8, 12):  # for exhaustive search, it should be: range(1, gpu.maxBLOCKSperSM + 1):
                                            # but heuristically reduce the search space
            for threads in range(gpu["warp_size"], gpu["maxTHREADSperBLOCK"] + 1, gpu["warp_size"]):

                if threads * minblocks > gpu["maxTHREADSperSM"]:
                    continue

                for tm in range(1, min(12, m + 1)):
                    for tn in range(1, min(12, n + 1)):

                        if tm * tn > 49:
                            continue  # heuristic: performance decreases for very large tiles

                        # Number of tiled columns, rows
                        cmax = (n + tn - 1) // tn
                        rmax = (m + tm - 1) // tm

                        # Minimum number of threads required to have one thread per tile,
                        # i.e., cover the result matrix
                        min_threads = cmax * rmax
                        if threads < min_threads:
                            continue
                        if min_threads < (threads - 32):
                            continue  # heuristic: too many threads unused during calculation

                        for w in range(4, (k + 1)//2, 2):  # heuristic: even numbers yield better performance
                            if w < tn:
                                continue  # invalid: input slap too small
                            if 2 * w > k:
                                continue  # heuristic: do at least one double-buffering step

                            for v in range(2, n + 1, 2):  # heuristic: even numbers yield better performance

                                if v < tm:
                                    continue  # invalid: output slab too small

                                # Number of registers
                                n_regs = tm * tn + (w * m + threads - 1) // threads + (w * n + threads - 1) // threads
                                if n_regs * threads * minblocks > 15000:
                                    continue  # heuristic: too many registers used

                                # Max work ("operations") which can be run concurrently
                                max_concurrent_work = max(grouping, m*w, w*n, m*v, cmax*rmax)
                                if threads > cu.round_up_to_multiple(max_concurrent_work, gpu["warp_size"]):
                                    continue  # heuristics: too much concurrency harms performance

                                # Shared memory buffer size
                                buf_sz = max((w - 1) * m + rmax * tm, m * w + (w - 1) * n + cmax * tn, v * m)
                                smem_tot = buf_sz * cu.sizeof_double + cu.npar * grouping * cu.sizeof_int
                                if smem_tot > gpu["SMEMperBLOCK"]:
                                    continue  # invalid: uses too much shared memory
                                if smem_tot * minblocks > gpu["SMEMperSM"]:
                                    continue  # invalid: uses too much shared memory

                                params.append({'m': m, 'n': n, 'k': k,
                                               'tile_m': tm, 'tile_n': tn,
                                               'w': w, 'v': v,
                                               'threads': threads,
                                               'grouping': grouping,
                                               'minblocks': minblocks})
        return params


#EOF
