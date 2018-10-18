# -*- coding: utf-8 -*-


class Kernel:

    def __repr__(self):
        return "<%s>" % self.name

    def can_handle(self, m, n, k):
        return self.m == m and self.n == n and self.k == k

    @property
    def include(self):
        return "cusmm_dnt_" + self.algorithm + ".h"

    @property
    def name(self):
        return "cusmm_dnt_" + self.algorithm + "_" + \
               "_".join([str(self.__dict__[k]) for k in sorted(self.launch_parameters)])

    @property
    def as_dict(self):
        return {**self.__dict__, **{'algorithm': self.algorithm}}

    @property
    def launcher_code(self):
        output  = "int launch_" + self.name + "(int *param_stack, int stack_size, "
        output += "cudaStream_t stream, int m_max, int n_max, int k_max, "
        output += "double *a_data, double *b_data, double *c_data){\n"
        output += "int shared_size = 0;\n"
        output += "//%s\n" % str(self.__dict__)
        output += "typedef void (*kernel)(const int*, int, const double*, const double*, double*);\n"
        output += "static kernel kern_func = " + self.func_signature
        output += "static bool configured = false;\n"
        output += "if(configured == false){\n"
        output += "  cudaError_t err = cudaFuncSetSharedMemConfig(kern_func, cudaSharedMemBankSizeEightByte);\n"
        output += "  if(err != cudaSuccess) return(-1);\n"
        output += "  configured = true;\n"
        output += "}\n"
        output += "kern_func<<< ((stack_size + %(grouping)d - 1) / %(grouping)d), %(threads)d, shared_size, stream >>>\n" \
                  % self.__dict__
        output += "(param_stack, stack_size, \n"
        output += "a_data, b_data, c_data);\n"
        output += "return(0);\n"
        output += "}\n"
        return output

    @staticmethod
    def promising_parameters(m, n, k):
        raise NotImplementedError()


#EOF