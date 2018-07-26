# -*- coding: utf-8 -*-
from kernels import cusmm_dnt
import cusmm_P100 as gpu
import cusmm_common as cu


class Kernel_dnt_small(cusmm_dnt.Kernel):

    algorithm = "small"
    launch_parameters = ['m', 'n', 'k', 'tile_m', 'tile_n', 'threads', 'grouping', 'minblocks']

    def __init__(self, *, m, n, k, threads, tile_m, tile_n, grouping, minblocks, perf):
        self.m = m
        self.n = n
        self.k = k
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.threads = threads
        self.grouping = grouping
        self.minblocks = minblocks
        self.perf = perf

    @property
    def func_signature(self):
        return "cusmm_dnt_small<%(m)d,%(n)d,%(k)d,%(tile_m)d,%(tile_n)d,%(threads)d,%(grouping)d,%(minblocks)d>;\n" \
               % self.__dict__

    @staticmethod
    def promising_parameters(m, n, k):

        params = []
        for minblocks in range(1, gpu.maxBLOCKSperSM + 1):
            for grouping in range(1, 32 + 1, 1):
                for tm in range(1, min(32, m) + 1):
                    for tn in range(1, min(32, n) + 1):

                        # Number of tiled columns, rows
                        cmax = (n + tn - 1) // tn
                        rmax = (m + tm - 1) // tm

                        # Miniumum number of threads required to have one thread per tile
                        min_threads = cmax * rmax

                        # Max work ("operations") which can be run concurrently
                        max_concurrent_work = max(grouping, m*k, k*n, m*n, min_threads)

                        # Set shared memory buffer size
                        buf_sz = max(m * n, m * k + k * tn * cmax, tm * rmax * k + 1)
                        smem_tot = buf_sz * gpu.sizeof_double + 3 * grouping * gpu.sizeof_int
                        if (smem_tot > gpu.SMEMperBLOCK):
                            continue
                        if (smem_tot * minblocks > gpu.SMEMperSM):
                            continue

                        # Use all concurrency available: fill warps
                        for threads in range(gpu.warp_size, gpu.maxTHREADSperBLOCK + 1, gpu.warp_size):
        
                            if threads > gpu.round_up_to_multiple(max_concurrent_work, gpu.warp_size):
                                continue
                            if threads * minblocks > gpu.maxTHREADSperSM:
                                continue
                            if threads < min_threads:
                                continue

                            params.append({'m':m, 'n':n, 'k':k,
                                           'tile_m':tm, 'tile_n':tn,
                                           'threads':threads,
                                           'grouping':grouping,
                                           'minblocks':minblocks})
        return(params)

#EOF
