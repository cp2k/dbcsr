# -*- coding: utf-8 -*-
from kernels import cusmm_dnt


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
        assert(self.m * self.n <= self.threads)

    @property
    def func_signature(self):
        return "cusmm_dnt_tiny<%(m)d,%(n)d,%(k)d,%(threads)d,%(grouping)d,%(minblocks)d>;\n" % self.__dict__

    @staticmethod
    def promising_parameters(m, n, k):
        params = []
        for minblocks in (7, 8, 14, 28):              # heuristic: kernel dependent optimum
            for grouping in range(1, 33, 1):          # soft: divide stack work in chunks of grouping + the rest
                for threads in (16, 32, 64):          # heuristic: not more than 2 warps per SM (sm_60)
                    if (m * n > threads):
                        continue                       # hard: not enough threads to cover result matrix

                    buf_sz = k * (m + n)
                    sizeof_int = 4; sizeof_double = 8
                    smem_tot = buf_sz * sizeof_double + 3 * grouping * sizeof_int
                    if (smem_tot * minblocks > 48 * 1024): # hard: see cudaFuncSetCacheConfig() docu
                        continue                       # hard: uses too much shared memory

                    params.append({'m':m, 'n':n, 'k':k,
                                   'threads':threads,
                                   'grouping':grouping,
                                   'minblocks':minblocks})
        return(params)

#EOF
