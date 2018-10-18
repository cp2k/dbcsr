# -*- coding: utf-8 -*-


class Kernel:

    characteristic_parameters = [
        'm', 'n', 'k',
        'tile_m', 'tile_n',
        'w', 'v',
        'threads', 'grouping', 'minblocks',
        'algorithm', 'perf'
    ]

    naming_parameters = [
        'm', 'n', 'k',
        'tile_m', 'tile_n',
        'w', 'v',
        'threads', 'grouping', 'minblocks',
    ]

    def __init__(self, **params):
        pass

    def __repr__(self):
        return ("<%s>" % self.name)

    def can_handle(self, m, n, k):
        return (self.m == m and self.n == n and self.k == k)

    def include(self):
        pass

    def to_dict(self):
        pass

    def compose_launcher_code(self, func_signature):
        output  = "int launch_"+self.name+"(int *param_stack, int stack_size, "
        output += "cudaStream_t stream, int m_max, int n_max, int k_max, "
        output += "double *a_data, double *b_data, double *c_data){\n"
        output += "int shared_size = 0;\n"
        output += "//%s\n"%str(self.__dict__)
        output += "typedef void (*kernel)(const int*, int, const double*, const double*, double*);\n"
        output += "static kernel kern_func = " + func_signature
        output += "static bool configured = false;\n"
        output += "if(configured == false){\n"
        output += "  cudaError_t err = cudaFuncSetSharedMemConfig(kern_func, cudaSharedMemBankSizeEightByte);\n"
        output += "  if(err != cudaSuccess) return(-1);\n"
        output += "  configured = true;\n"
        output += "}\n"
        output += "kern_func<<< ((stack_size + %(grouping)d - 1) / %(grouping)d), %(threads)d, shared_size, stream >>>\n"%self.__dict__
        output += "(param_stack, stack_size, \n"
        output += "a_data, b_data, c_data);\n"
        output += "return(0);\n"
        output += "}\n"
        return(output)

    @staticmethod
    def promising_parameters(m, n, k):
        pass

#EOF