# -*- coding: utf-8 -*-
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

# ===============================================================================
#  Computing helpers


def round_up_to_nearest_multiple(x, step):
    import numpy as np

    result = np.where(x % step == 0, x, x + step - x % step).astype(float)
    if result.size == 1:
        result = result.item()  # extract single element of numpy array
    return result


def round_down_to_nearest_multiple(x, step):
    import numpy as np

    result = np.where(x % step == 0, x, x - x % step).astype(float)
    if result.size == 1:
        result = result.item()  # extract single element of numpy array
    return result


# ===============================================================================
class Kernel:
    """
    Base class for libsmm_acc's kernels
    """

    def __repr__(self):
        return "<%s>" % self.name

    def can_handle(self, m, n, k):
        return self.m == m and self.n == n and self.k == k

    @property
    def include(self):
        return "smm_acc_dnt_" + self.algorithm + ".h"

    @property
    def name(self):
        return (
            "smm_acc_dnt_"
            + self.algorithm
            + "_"
            + "_".join([str(self.__dict__[k]) for k in self.launch_parameters])
        )

    @property
    def autotuned(self):
        return True if self.source == "autotuned" else False

    @property
    def as_dict(self):
        return dict(algorithm=self.algorithm, **self.__dict__)

    @property
    def as_dict_for_parameters_json(self):
        """
        Return the kernel as a dictionary in such a way that it is convenient to write in a JSON file and parameters
        always appear in the same order
        """
        # Add common fields
        fields = [
            "m",
            "n",
            "k",
            "tile_m",
            "tile_n",
            "w",
            "v",
            "threads",
            "grouping",
            "minblocks",
            "algorithm",
        ]
        d = dict()
        for f in fields:
            if f in self.as_dict.keys():
                d[f] = self.as_dict[f]

        # Only add the performance if it was autotuned. If it was predicted, we only have a value scaled in (0,1)
        if self.as_dict["source"] == "autotuned":
            d["perf"] = self.as_dict["perf"]

        # Add the source of this parameter set (autotuned or predicted)
        d["source"] = self.as_dict["source"]

        return d

    @property
    def as_dict_for_parameters_h(self):
        """
        Return the kernel as a dictionary in such a way that it is convenient to write into the parameters.h file and
        parameters always appear in the same order
        """
        # Add common fields
        fields = [
            "m",
            "n",
            "k",
            "tile_m",
            "tile_n",
            "w",
            "v",
            "threads",
            "grouping",
            "minblocks",
            "perf",
            "source",
        ]
        d = dict()
        for f in fields:
            d[f] = self.as_dict[f] if f in self.as_dict.keys() else 0

        # Add algorithm and source
        d["algorithm"] = self.algorithm_num
        d["source"] = "(predicted)" if not self.autotuned else ""
        return d

    def launcher_code(self, compiler):
        """
        Compiler: either "nvcc" or "hipcc": determines the C++ dialect to use for kernel launching: either CUDA or HIP
        """
        indent = "  "
        output = "int launch_" + self.name + "(int *param_stack, int stack_size, "
        if compiler == "nvcc":
            output += "cudaStream_t stream, "
        else:  # i.e. compiler == "hipcc"
            output += "hipStream_t stream, "
        output += "int m_max, int n_max, int k_max, "
        output += "double *a_data, double *b_data, double *c_data){\n"
        output += indent + "int shared_size = 0;\n"
        output += indent + "//%s\n" % str(self.__dict__)
        output += (
            indent
            + "typedef void (*kernel)(const int*, int, const double*, const double*, double*);\n"
        )
        output += indent + "static kernel kern_func = " + self.func_signature

        # The syntax for kernel launching is different in CUDA and HIP
        if compiler == "nvcc":
            output += (
                indent
                + "kern_func<<< ((stack_size + %(grouping)d - 1) / %(grouping)d), %(threads)d, shared_size, stream >>>(\n"
                % self.__dict__
            )
        else:  # i.e. compiler == "hipcc"
            output += (
                indent
                + "hipLaunchKernelGGL"
                + "(kern_func, (stack_size + %(grouping)d - 1) / %(grouping)d, %(threads)d, shared_size, stream, \n"
                % self.__dict__
            )
        output += indent + "param_stack, stack_size, a_data, b_data, c_data);\n"
        output += indent + "return 0;\n"
        output += "}\n"
        return output

    @property
    def func_signature(self):
        raise NotImplementedError("func_signature must be implemented in subclass")

    @staticmethod
    def promising_parameters(m, n, k, gpu, autotuning):
        raise NotImplementedError(
            "promising_parameters must be implemented in subclass"
        )

    @staticmethod
    def baseline(m, n, k, gpu, autotuning):
        """Compute a baseline parameter set, whose performance can be compared against"""
        raise NotImplementedError("baseline must be implemented in subclass")

    @classmethod
    def parameter_set_distance(cls, par_set1, par_set2):
        """
        Compute a distance-score between two parameter sets.
        The lower the score, the closer the two parameter sets are
        par_set1, par_set2 are parameter set dictionaries of the form
        {m, n, k, algorithm, minblocks, grouping, threads, tile_m, tile_n, w, v}
        """
        # Check that ('m', 'n', 'k') are the same
        assert (
            par_set1["m"] == par_set2["m"]
        ), "The two parameter sets have different 'm'-parameters: {} and {}".format(
            par_set1["m"], par_set2["m"]
        )
        assert (
            par_set1["n"] == par_set2["n"]
        ), "The two parameter sets have different 'n'-parameters: {} and {}".format(
            par_set1["n"], par_set2["n"]
        )
        assert (
            par_set1["k"] == par_set2["k"]
        ), "The two parameter sets have different 'k'-parameters: {} and {}".format(
            par_set1["k"], par_set2["k"]
        )

        # Compute distance in number of threads
        score = abs(par_set1["threads"] - par_set2["threads"]) / 32
        par_entries = [
            p for p in cls.launch_parameters if p not in ("m", "n", "k", "threads")
        ]

        # Compute distance in other parameters
        for par in par_entries:
            score += abs(par_set1[par] - par_set2[par])

        return score
