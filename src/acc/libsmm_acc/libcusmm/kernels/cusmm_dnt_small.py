# -*- coding: utf-8 -*-
from kernels import cusmm_dnt


class Kernel_dnt_small(cusmm_dnt.Kernel):

    algorithm = "small"

    def __init__(self, **params):
        self.__dict__.update(params)
        self.name  = "cusmm_dnt_small_"
        self.name += "_".join([str(params[k]) for k in sorted(super().naming_parameters) if k in params.keys()])

    def include(self):
        return("cusmm_dnt_small.h")

    def to_dict(self):
        d = {**self.__dict__, **{'algorithm': self.algorithm}}
        return dict([(e, d[e]) for e in super().characteristic_parameters if e in d.keys()])

    def launcher_code(self):
        sign = "cusmm_dnt_small<%(m)d,%(n)d,%(k)d,%(tile_m)d,%(tile_n)d,%(threads)d,%(grouping)d,%(minblocks)d>;\n" % self.__dict__
        return super().compose_launcher_code(sign)

    @staticmethod
    def promising_parameters(m, n, k):
        params = []
        for minblocks in (7, 8, 14, 28):              # heuristic: kernel dependent optimum
            for grouping in range(1, 33, 1):          # soft: divide stack work in chunks of grouping + the rest
                for threads in (32, 48, 64):          # heuristic: not more than 2 warps per SM (sm_60)
                    if(threads * minblocks > 2048):   # hard: too much concurrent threads per SM
                        continue
                    for tm in (1, 2,):
                        for tn in (1, 2,):
                            min_threads = ((m + tm - 1) // tm) * ((n + tn - 1) // tn)
                            if (min_threads > threads):
                                continue              # hard: not enough threads to cover result matrix
        
                            cmax = ((n + tn - 1) // tn)
                            rmax = ((m + tm - 1) // tm)
                            buf_sz = max(m * n, m * k + k * tn * cmax, tm * rmax * k + 1)
                            sizeof_int = 4; sizeof_double = 8
                            smem_tot = buf_sz * sizeof_double + 3 * grouping * sizeof_int
                            if (smem_tot * minblocks > 48 * 1024): # hard: see cudaFuncSetCacheConfig() docu
                                continue              # hard: uses too much shared memory
        
                            params.append({'m':m, 'n':n, 'k':k,
                                           'tile_m':tm, 'tile_n':tn,
                                           'threads':threads,
                                           'grouping':grouping,
                                           'minblocks':minblocks})
        return(params)

#EOF
