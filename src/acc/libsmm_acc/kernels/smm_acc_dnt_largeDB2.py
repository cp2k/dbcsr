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


class Kernel_dnt_largeDB2(Kernel):
    """Kernel 'large double-buffering' 2"""

    algorithm = "largeDB2"
    algorithm_num = 2
    launch_parameters = [
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
    ]

    def __init__(
        self, m, n, k, threads, tile_m, tile_n, w, v, grouping, minblocks, perf, source
    ):
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
        self.source = source
        assert self.threads * self.minblocks <= 2048
        min_threads = ((self.m + self.tile_m - 1) // self.tile_m) * (
            (self.n + self.tile_n - 1) // self.tile_n
        )
        assert min_threads <= self.threads
        assert self.tile_m <= self.v
        assert self.tile_n <= self.w

    @property
    def func_signature(self):
        return (
            "smm_acc_dnt_largeDB2"
            + "< {m}, {n}, {k}, {threads}, {grouping}, {minblocks} >;\n".format(
                **self.__dict__
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
        grouping = 16

        for minblocks_ in (1, 2, 4, 8, 12) if minblocks is None else [minblocks]:
            # for exhaustive search, it should be: range(1, gpu.maxBLOCKSperSM + 1):
            # but heuristically reduce the search space
            for threads_ in (
                range(
                    gpu["Threads_/_Warp"],
                    gpu["Max_Thread_Block_Size"] + 1,
                    gpu["Threads_/_Warp"],
                )
                if threads is None
                else [threads]
            ):

                if threads_ * minblocks_ > gpu["Threads_/_Multiprocessor"]:
                    continue

                for tm in range(1, min(12, m + 1)) if tile_m is None else [tile_m]:
                    for tn in range(1, min(12, n + 1)) if tile_n is None else [tile_n]:

                        if tm * tn > 49:
                            continue  # heuristic: performance decreases for very large tiles

                        # Number of tiled columns, rows
                        cmax = (n + tn - 1) // tn
                        rmax = (m + tm - 1) // tm

                        # Minimum number of threads required to have one thread per tile
                        # i.e., cover the result matrix
                        min_threads = cmax * rmax
                        if threads_ < min_threads:
                            continue
                        if min_threads < (threads_ - 32):
                            continue  # heuristic: too many threads unused during calculation

                        for w_ in range(4, (k + 1) // 2, 2) if w is None else [w]:
                            # heuristic: even numbers yield better performance

                            if w_ < tn:
                                continue  # invalid: input slap too small
                            if 2 * w_ > k:
                                continue  # heuristic: do at least one double-buffering step

                            for v_ in range(4, n + 1, 2) if v is None else [v]:
                                # heuristic: even numbers yield better performance

                                if v_ < tm:
                                    continue  # invalid: output slab too small

                                # Number of registers
                                n_regs = (
                                    tm * tn
                                    + (w_ * m + threads_ - 1) // threads_
                                    + (w_ * n + threads_ - 1) // threads_
                                )
                                if n_regs * threads_ * minblocks_ > 15000:
                                    continue  # heuristic: too many registers used

                                # Max work ("operations") which can be run concurrently
                                max_concurrent_work = max(
                                    grouping, m * w_, w_ * n, m * v_, cmax * rmax
                                )
                                if threads_ > round_up_to_nearest_multiple(
                                    max_concurrent_work, gpu["Threads_/_Warp"]
                                ):
                                    continue  # heuristics: too much concurrency harms performance

                                # Shared memory buffer size
                                buf_sz = max(
                                    (w_ - 1) * m + rmax * tm,
                                    m * w_ + (w_ - 1) * n + cmax * tn,
                                    v_ * m,
                                )
                                smem_tot = (
                                    buf_sz * autotuning["sizeof_double"]
                                    + autotuning["npars"]
                                    * grouping
                                    * autotuning["sizeof_int"]
                                )
                                if smem_tot > gpu["Max_Shared_Memory_/_Block_(bytes)"]:
                                    continue  # invalid: uses too much shared memory
                                if (
                                    smem_tot * minblocks_
                                    > gpu["Shared_Memory_/_Multiprocessor_(bytes)"]
                                ):
                                    continue  # invalid: uses too much shared memory

                                params.append(
                                    {
                                        "m": m,
                                        "n": n,
                                        "k": k,
                                        "tile_m": tm,
                                        "tile_n": tn,
                                        "w": w_,
                                        "v": v_,
                                        "threads": threads_,
                                        "grouping": grouping,
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

        grouping = 16
        minblk = 2
        tm = 2
        tn = 2
        cmax = (n + tn - 1) // tn
        rmax = (m + tm - 1) // tm
        min_threads = cmax * rmax
        w = 8
        v = 8

        while True:
            base = {
                "threads": round_up_to_nearest_multiple(min_threads, 32),
                "grouping": grouping,
                "minblocks": minblk,
                "tile_m": tn,
                "tile_n": tn,
                "w": w,
                "v": v,
            }
            if (
                len(
                    Kernel_dnt_largeDB2.promising_parameters(
                        m, n, k, gpu, autotuning, **base
                    )
                )
                > 0
            ):
                break
            else:
                if w > 1:
                    w /= 2
                else:
                    base = Kernel_dnt_largeDB2.promising_parameters(
                        m, n, k, gpu, autotuning
                    )[0]
                    break

        base.update(
            dict(
                [
                    ("m", m),
                    ("n", n),
                    ("k", k),
                    ("algorithm", "largeDB2"),
                    ("perf", 0),
                    ("source", "predicted"),
                ]
            )
        )
        return base
