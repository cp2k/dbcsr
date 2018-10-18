# -*- coding: utf-8 -*-
from kernels import cusmm_dnt


class Kernel_dnt_largeDB1(cusmm_dnt.Kernel):

    algorithm = "largeDB1"
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
        assert(self.threads * self.minblocks <= 2048)
        min_threads = ((self.m+self.tile_m-1)//self.tile_m) * ((self.n+self.tile_n-1)//self.tile_n)
        assert(min_threads <= self.threads)
        assert(self.tile_m <= self.v)
        assert(self.tile_n <= self.w)

    @property
    def func_signature(self):
        return "cusmm_dnt_largeDB1<%(m)d,%(n)d,%(k)d,%(tile_m)d,%(tile_n)d,%(w)d,%(v)d,%(threads)d,%(grouping)d,%(minblocks)d>;\n" \
               % self.__dict__

    @staticmethod
    def promising_parameters(m, n, k):
        params = []
        grouping = 16

        for minblocks in (1, 2, 4, 8, 12):
            for threads in range (96, 513, 32):

                # invalid: too many threads per SM
                if(threads * minblocks > 2048): continue

                for tm in range(1, 12):
                    for tn in range(1, 12):

                        # invalid: not enough threads to cover result matrix
                        min_threads = ((m + tm - 1) // tm) * ((n + tn - 1) // tn)
                        if (min_threads > threads): continue

                        # heuristic: too many threads unused during calculation
                        if (min_threads < (threads - 16)): continue

                        # heuristic: only even numbers
                        for w in range(2, k + 1, 2):

                            # invalid: input slap too small
                            if (w < tn): continue

                            # heuristic: do at least one double-buffering step
                            if (2 * w > k): continue

                            # heuristic: only even numbers
                            for v in range(2, n + 1, 2):

                                # invalid: output slap too small
                                if (v < tm): continue

                                #heuristic: too many registers used
                                n_regs = tm * tn + (w * m + threads - 1) // threads + (w * n + threads - 1) // threads
                                if (n_regs * threads * minblocks > 15000): continue

                                # invalid: uses too much shared memory
                                cmax = ((n + tn - 1) // tn)
                                rmax = ((m + tm - 1) // tm)
                                buf_sz = max((w - 1) * m + rmax * tm, m * w + (w - 1) * n + cmax * tn, v * m)
                                sizeof_int = 4; sizeof_double = 8
                                smem_tot = buf_sz * sizeof_double + 3 * grouping * sizeof_int
                                if (smem_tot * minblocks > 48 * 1024): continue # hard: see cudaFuncSetCacheConfig() docu

                                params.append({'m':m, 'n':n, 'k':k,
                                               'tile_m':tm, 'tile_n':tn,
                                               'w':w, 'v':v,
                                               'threads':threads,
                                               'grouping':grouping,
                                               'minblocks':minblocks})
        return(params)


#EOF
