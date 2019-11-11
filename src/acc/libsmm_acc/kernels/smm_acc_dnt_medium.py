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


class Kernel_dnt_medium(Kernel):

    algorithm = "medium"
    algorithm_num = 3
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
        assert self.threads * self.minblocks <= 2048
        min_threads = ((self.m + self.tile_m - 1) // self.tile_m) * (
            (self.n + self.tile_n - 1) // self.tile_n
        )
        assert min_threads <= self.threads

    @property
    def func_signature(self):
        return (
            "smm_acc_dnt_medium"
            + "<{m}, {n}, {k}, {tile_m}, {tile_n}, {threads}, {grouping}, {minblocks} >;\n".format(
                self.__dict__
            )
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
        from kernels.smm_acc_dnt_base import round_up_to_nearest_multiple

        params = []
        for minblocks_ in range(1, 28) if minblocks is None else [minblocks]:
            # for exhaustive search: range(1, gpu["Thread_Blocks_/_Multiprocessor"] + 1):
            # heuristic: the optimal minblocks is never > 28
            for grouping_ in range(1, 32 + 1, 1) if grouping is None else [grouping]:

                if m >= 28 and grouping_ not in (3, 4, 5, 24, 26, 29, 32):
                    continue  # heuristic: investigate a smaller search space of grouping for large matrices

                for tm in (
                    range(1, min(12, m) + 1) if tile_m is None else [tile_m]
                ):  # heuristic: the optimal tile_m is never above 12
                    for tn in (
                        range(1, min(12, n) + 1) if tile_n is None else [tile_n]
                    ):  # heuristic: the optimal tile_m is never above 12

                        if tm * tn > 16:
                            continue  # heuristic: performance decreases for very large tiles

                        # Number of tiled columns, rows
                        cmax = (n + tn - 1) // tn
                        rmax = (m + tm - 1) // tm

                        # Max work ("operations") which can be run concurrently
                        max_concurrent_work = max(
                            grouping_, m * k, k * n, m * n, cmax * rmax
                        )

                        # Minimum number of threads required to have one thread per tile
                        # i.e., cover the result matrix
                        min_threads = cmax * rmax

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

        while True:

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
                len(
                    Kernel_dnt_medium.promising_parameters(
                        m, n, k, gpu, autotuning, **base
                    )
                )
                > 0
            ):
                break
            else:
                grp -= 1
                if grp == 0:
                    base = Kernel_dnt_medium.promising_parameters(
                        m, n, k, gpu, autotuning
                    )[0]
                    base.update(dict([("w", float("nan")), ("v", float("nan"))]))
                    break

        base.update(
            dict(
                [
                    ("m", m),
                    ("n", n),
                    ("k", k),
                    ("algorithm", "medium"),
                    ("perf", 0),
                    ("source", "predicted"),
                ]
            )
        )
        return base
