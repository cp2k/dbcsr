# -*- coding: utf-8 -*-
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

import numpy as np
from kernels.cusmm_dnt_base import Kernel, round_up_to_nearest_multiple


class Kernel_dnt_tiny(Kernel):

    algorithm = "tiny"
    algorithm_num = 5
    launch_parameters = ["m", "n", "k", "threads", "grouping", "minblocks"]

    def __init__(self, m, n, k, threads, grouping, minblocks, perf, source):
        self.m = m
        self.n = n
        self.k = k
        self.threads = threads
        self.grouping = grouping
        self.minblocks = minblocks
        self.perf = perf
        self.source = source
        assert self.m * self.n <= self.threads

    @property
    def func_signature(self):
        return (
            "cusmm_dnt_tiny<%(m)d,%(n)d,%(k)d,%(threads)d,%(grouping)d,%(minblocks)d>;\n"
            % self.__dict__
        )

    @staticmethod
    def promising_parameters(
        m,
        n,
        k,
        gpu,
        autotuning,
        threads=None,
        grouping=None,
        minblocks=None,
        tile_m=None,
        tile_n=None,
        w=None,
        v=None,
    ):
        """
        Given a certain (m,n,k)-triplet, GPU properties and autotuning properties, return a list of all possible
        kernel parameters
        """

        # Shared memory buffer size
        buf_sz = k * (
            m + n
        )  # number of elements in the a_block buffer = mk, and in the b_block buffer = kn

        # Minimum number of threads required to cover the result matrix c
        min_threads = m * n

        # Parameter space:
        params = []
        for minblocks_ in (
            range(1, gpu["Thread_Blocks_/_Multiprocessor"] + 1)
            if minblocks is None
            else [minblocks]
        ):
            # heuristic: never seen optimal=1 hence start from 2
            for grouping_ in range(2, 32 + 1, 1) if grouping is None else [grouping]:

                # Max work ("operations")  which can be run concurrently
                max_concurrent_work = max(grouping_, m * k, k * n, m * n)

                # Shared memory utilisation (bytes)
                smem_tot = (
                    buf_sz * autotuning["sizeof_double"]
                    + autotuning["npars"] * grouping_ * autotuning["sizeof_int"]
                )
                if smem_tot > gpu["Max_Shared_Memory_/_Block_(bytes)"]:
                    continue
                if smem_tot * minblocks_ > gpu["Max_Shared_Memory_/_Block_(bytes)"]:
                    continue

                # Use all concurrency available: fill warps
                for threads_ in (
                    range(
                        gpu["Threads_/_Warp"],
                        gpu["Max_Thread_Block_Size"] + 1,
                        gpu["Threads_/_Warp"],
                    )
                    if threads is None
                    else [threads]
                ):

                    if threads_ > round_up_to_nearest_multiple(
                        max_concurrent_work, gpu["Threads_/_Warp"]
                    ):
                        continue  # soft: too much concurrency harms performance
                    if threads_ * minblocks_ > gpu["Threads_/_Multiprocessor"]:
                        continue
                    if threads_ < min_threads:
                        continue

                    params.append(
                        {
                            "m": m,
                            "n": n,
                            "k": k,
                            "threads": threads_,
                            "grouping": grouping_,
                            "minblocks": minblocks_,
                        }
                    )
        return params

    @staticmethod
    def baseline(m, n, k, gpu, autotuning):
        """
        Given an (m, n, k)-triplet and GPu and autotuning properties, return a set of parameters corresponding to a
        baseline ("educated guess") of the kernel's optimal parameters
        """

        grp = 16
        minblk = 2
        min_threads = m * n

        base = {
            "threads": round_up_to_nearest_multiple(min_threads, 32),
            "grouping": grp,
            "minblocks": minblk,
            "tile_m": np.NaN,
            "tile_n": np.NaN,
            "w": np.NaN,
            "v": np.NaN,
        }

        if (
            len(Kernel_dnt_tiny.promising_parameters(m, n, k, gpu, autotuning, **base))
            > 0
        ):
            base = Kernel_dnt_tiny.promising_parameters(m, n, k, gpu, autotuning)[0]
            base.update(
                dict(
                    [
                        ("tile_m", np.NaN),
                        ("tile_n", np.NaN),
                        ("w", np.NaN),
                        ("v", np.NaN),
                    ]
                )
            )

        base.update(
            dict(
                [
                    ("m", m),
                    ("n", n),
                    ("k", k),
                    ("algorithm", "tiny"),
                    ("perf", 0),
                    ("source", "predicted"),
                ]
            )
        )
        return base
