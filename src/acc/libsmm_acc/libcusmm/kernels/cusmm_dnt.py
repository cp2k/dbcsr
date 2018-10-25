# -*- coding: utf-8 -*-
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################


#===============================================================================
def round_up_to_multiple(x, step):
    if x % step == 0:
        return x
    else:
        return x + step - x % step


#===============================================================================
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
    def autotuned(self):
        return True if self.source == 'autotuned' else False

    @property
    def as_dict(self):
        return dict(algorithm=self.algorithm, **self.__dict__)

    @property
    def as_dict_for_parameters_json(self):
        fields = ['m', 'n', 'k', 'tile_m', 'tile_n', 'w', 'v', 'threads', 'grouping', 'minblocks', 'algorithm', 'perf', 'source']
        d = dict()
        for f in fields:
            d[f] = self.as_dict[f] if f in self.as_dict.keys() else 0
        return d

    @property
    def as_dict_for_parameters_json(self):
        fields = ['m', 'n', 'k', 'tile_m', 'tile_n', 'w', 'v', 'threads', 'grouping', 'minblocks', 'algorithm', 'perf', 'source']
        d = dict()
        for f in fields:
            if f in self.as_dict.keys():
                d[f] = self.as_dict[f]
        return d

    @property
    def as_dict_for_parameters_h(self):
        fields = ['m', 'n', 'k', 'tile_m', 'tile_n', 'w', 'v', 'threads', 'grouping', 'minblocks', 'perf', 'source']
        d = dict()
        for f in fields:
            d[f] = self.as_dict[f] if f in self.as_dict.keys() else 0
        d['algorithm'] = self.algorithm_num
        d['source'] = '(predicted)' if not self.autotuned else ''
        return d

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
    def promising_parameters(m, n, k, gpu, autotuning):
        raise NotImplementedError()
