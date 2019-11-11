# -*- coding: utf-8 -*-
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

from kernels.smm_acc_dnt_base import Kernel


class Kernel_dnt_small(Kernel):

    algorithm = "small"
    algorithm_num = 4
    launch_parameters = [
        "m",
        "n",
        "k",
        "tile_m",
        "tile_n",
        "threads",
        "grouping",
        "minblocks",
    ]

    def __init__(
        self, m, n, k, threads, tile_m, tile_n, grouping, minblocks, perf, source
    ):
        self.m = m
        self.n = n
        self.k = k
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.threads = threads
        self.grouping = grouping
        self.minblocks = minblocks
        self.perf = perf
        self.source = source

    def func_signature(self):
        return "smm_acc_dnt_tiny< {m}, {n}, {k}, {threads}, {grouping}, {minblocks} >;\n".format(
            self.__dict__
        )

    @property
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
        from kernels.smm_acc_dnt_base import round_up_to_nearest_multiple

        # Parameter space:
        params = []
        for minblocks_ in (
            range(1, gpu["Thread_Blocks_/_Multiprocessor"] + 1)
            if minblocks is None
            else [minblocks]
        ):
            for grouping_ in (
                range(2, 32 + 1, 1) if grouping is None else [grouping]
            ):  # heuristic: never seen optimal=1 hence start from 2
                for tm in (
                    range(1, min(12, m) + 1) if tile_m is None else [tile_m]
                ):  # heuristic: the optimal tile_m is never above 12
                    for tn in (
                        range(1, min(12, n) + 1) if tile_n is None else [tile_n]
                    ):  # heuristic: the optimal tile_n is never above 12

                        if tm * tn > 16:
                            continue  # heuristic: performance decreases for very large tiles

                        # Number of tiled columns, rows
                        cmax = (n + tn - 1) // tn
                        rmax = (m + tm - 1) // tm

                        # Minimum number of threads required to have one thread per tile
                        min_threads = cmax * rmax

                        # Max work ("operations") which can be run concurrently
                        max_concurrent_work = max(
                            grouping_, m * k, k * n, m * n, min_threads
                        )

                        # Shared memory buffer size
                        buf_sz = max(m * n, m * k + k * tn * cmax, tm * rmax * k + 1)
                        smem_tot = (
                            buf_sz * autotuning["sizeof_double"]
                            + autotuning["npars"] * grouping_ * autotuning["sizeof_int"]
                        )
                        if smem_tot > gpu["Max_Shared_Memory_/_Block_(bytes)"]:
                            continue
                        if (
                            smem_tot * minblocks_
                            > gpu["Shared_Memory_/_Multiprocessor_(bytes)"]
                        ):
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
                                    "tile_m": tm,
                                    "tile_n": tn,
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
        from kernels.smm_acc_dnt_base import round_up_to_nearest_multiple

        grp = 16
        minblk = 2
        tm = 2
        tn = 2
        cmax = (n + tn - 1) // tn
        rmax = (m + tm - 1) // tm
        min_threads = cmax * rmax

        base = {
            "threads": round_up_to_nearest_multiple(min_threads, 32),
            "grouping": grp,
            "minblocks": minblk,
            "tile_m": tn,
            "tile_n": tn,
            "w": float("nan"),
            "v": float("nan"),
        }

        if (
            len(Kernel_dnt_small.promising_parameters(m, n, k, gpu, autotuning, **base))
            > 0
        ):
            base = Kernel_dnt_small.promising_parameters(m, n, k, gpu, autotuning)[0]
            base.update(
                dict(
                    [
                        ("tile_m", float("nan")),
                        ("tile_n", float("nan")),
                        ("w", float("nan")),
                        ("v", float("nan")),
                    ]
                )
            )

        base.update(
            dict(
                [
                    ("m", m),
                    ("n", n),
                    ("k", k),
                    ("algorithm", "small"),
                    ("perf", 0),
                    ("source", "predicted"),
                ]
            )
        )
        return base
