# -*- coding: utf-8 -*-
from kernels import cusmm_dnt


class Kernel_dnt_medium(cusmm_dnt.Kernel):

    algorithm = 'medium'

    def __init__(self, **params):
        self.__dict__.update(params)
        self.name  = "cusmm_dnt_medium_"
        self.name += "_".join([str(params[k]) for k in sorted(super().naming_parameters) if k in params.keys()])
        assert(self.threads * self.minblocks <= 2048)
        min_threads = ((self.m + self.tile_m - 1) // self.tile_m) * ((self.n + self.tile_n - 1) // self.tile_n)
        assert(min_threads <= self.threads)

    def include(self):
        return("cusmm_dnt_medium.h")

    def to_dict(self):
        d = {**self.__dict__, **{'algorithm': self.algorithm}}
        return dict([(e, d[e]) for e in super().characteristic_parameters if e in d.keys()])

    def launcher_code(self):
        sign = "cusmm_dnt_medium<%(m)d,%(n)d,%(k)d,%(tile_m)d,%(tile_n)d,%(threads)d,%(grouping)d,%(minblocks)d>;\n" % self.__dict__
        return super().compose_launcher_code(sign)

    @staticmethod
    def promising_parameters(m, n, k):
        params = []
        for minblocks in range(1, 28, 1):
            for grouping in range(1, 33, 1):
                for threads in range (32, 257, 32):
                    if(threads * minblocks > 2048): # hard: too much concurrent threads per SM
                        continue
                    for tm in range(1, 7):
                        for tn in range(1, 7):
                            if (tm * tn > 16):
                                continue #heuristic:
                            min_threads = ((m + tm - 1) // tm) * ((n + tn - 1) // tn)
                            if (min_threads > threads):
                                continue #hard: not enough threads to cover result matrix
    
                            if (threads > 4 * min_threads):
                                continue #heuristic: too many threads unused during calculation
    
                            cmax = ((n + tn - 1) // tn)
                            rmax = ((m + tm - 1) // tm)
                            buf_sz = max(m * n, m * k + k * tn * cmax, tm * rmax * k + 1)
                            sizeof_int = 4; sizeof_double = 8
                            smem_tot = buf_sz * sizeof_double + 3 * grouping * sizeof_int
                            if(smem_tot * minblocks > 48 * 1024): # hard: see cudaFuncSetCacheConfig() docu
                                continue #hard: uses too much shared memory
    
                            params.append({'m':m, 'n':n, 'k':k,
                                           'tile_m':tm, 'tile_n':tn,
                                           'threads':threads,
                                           'grouping':grouping,
                                           'minblocks':minblocks})
        return(params)


#EOF
