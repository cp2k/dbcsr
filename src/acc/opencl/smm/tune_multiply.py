#!/usr/bin/env python3
###############################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved           #
# This file is part of the DBCSR library.                                     #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# For further information please visit https://dbcsr.cp2k.org                 #
# SPDX-License-Identifier: GPL-2.0+                                           #
###############################################################################
#
# This script is based on OpenTuner's tutorial
# "Optimizing Block Matrix Multiplication", and
# LIBXSMM's "xgemm" and "transpose" examples.
#
import opentuner
from opentuner import ConfigurationManipulator
from opentuner import MeasurementInterface
from opentuner import IntegerParameter
from opentuner import Result
import json
import time
import sys
import re


class SmmTuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        self.exepath = "../.."
        self.exename = "acc_bench_smm"
        run_result = self.call_program(self.exepath + "/" + self.exename + " 1 1 1")
        if (0 == run_result["returncode"]):
            match = re.search("element type:\\s+(\\w+)", str(run_result["stdout"]))
        if (match is not None):
            self.elemtype = match.group(1)
        else:
            sys.tracebacklimit = 0
            raise RuntimeError("Execution failed for \"" +
                               self.exepath + "/" +
                               self.exename + "\"!")
        # sanitize input arguments
        self.args.m = max(self.args.m, 1)
        self.args.n = [max(self.args.n, 1), self.args.m][0 == self.args.n]
        self.args.k = [max(self.args.k, 1), self.args.m][0 == self.args.k]
        self.args.mb = max(self.args.mb, 1)
        self.args.bs = max(min(self.args.bs, self.args.mb), 1)
        self.args.bm = [max(self.args.bm, 1), self.args.m][0 == self.args.bm]
        self.args.bn = [max(self.args.bn, 1), 1][0 == self.args.bn]
        # setup tunable parameters
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(IntegerParameter("BS", 1, self.args.mb))
        manipulator.add_parameter(IntegerParameter("BM", 1, self.args.m))
        manipulator.add_parameter(IntegerParameter("BN", 1, self.args.n))
        return manipulator

    def seed_configurations(self):
        return [{"BS": self.args.bs,
                 "BM": self.args.bm,
                 "BN": self.args.bn}]

    def objective(self):
        return opentuner.search.objective.MaximizeAccuracyMinimizeSize()

    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        cfg = desired_result.configuration.data
        run_cmd = (
            "OMP_PROC_BIND=TRUE CHECK=" + str(self.args.check) +
            " OPENCL_LIBSMM_SMM_BATCHSIZE=" + str(cfg["BS"]) +
            " OPENCL_LIBSMM_SMM_BLOCK_M=" + str(cfg["BM"]) +
            " OPENCL_LIBSMM_SMM_BLOCK_N=" + str(cfg["BN"]) +
            " " + self.exepath + "/" + self.exename + " 0 0" +
            " " + str(self.args.m) +
            " " + str(self.args.n) +
            " " + str(max(self.args.k, 1)))
        run_result = self.call_program(run_cmd)
        if (0 == run_result["returncode"]):
            match = re.search(
                "device:\\s+([0-9]+(\\.[0-9]*)*) ms\\s+([0-9]+(\\.[0-9]*)*)",
                str(run_result["stdout"]))
            assert(match is not None)
            mseconds = float(match.group(1))
            gflops = float(match.group(3))
            kernelreq = round((100.0 * cfg["BM"] * cfg["BN"])
                              / (self.args.m * self.args.n))
            # gflops are reported as "accuracy" (console output)
            return Result(time=mseconds, accuracy=gflops, size=kernelreq)

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        filename = ("tune_multiply-" + self.elemtype + "-" +
                    str(self.args.m) + "x" + str(self.args.n) + "x" + str(self.args.k) +
                    time.strftime("-%Y%m%d-%H%M%S") + ".json")
        print("Optimal block size written to " + filename + ": ", configuration.data)
        # self.manipulator().save_to_file(configuration.data, filename)
        with open(filename, 'w') as fd:
            json.dump(configuration.data, fd)


if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    argparser.add_argument(
        "m", type=int, default=23, nargs='?',
        help="Shape of SMM-kernel (M)")
    argparser.add_argument(
        "n", type=int, default=0, nargs='?',
        help="Shape of SMM-kernel (N)")
    argparser.add_argument(
        "k", type=int, default=0, nargs='?',
        help="Shape of SMM-kernel (K)")
    argparser.add_argument(
        "-bm", "--initial-bm", type=int, default=0, nargs='?',
        dest="bm", help="Initial block/tile size (BM)")
    argparser.add_argument(
        "-bn", "--initial-bn", type=int, default=0, nargs='?',
        dest="bn", help="Initial block/tile size (BN)")
    argparser.add_argument(
        "-bs", "--initial-bs", type=int, default=1, nargs='?',
        dest="bs", help="Initial (mini-)batch size (BS)")
    argparser.add_argument(
        "-mb", "--max-bs", type=int, default=128, nargs='?',
        dest="mb", help="Maximum (mini-)batch size (BS)")
    argparser.add_argument(
        "-v", "--check", type=float, default=0, nargs='?',
        dest="check", help="Validate kernel (epsilon)")
    SmmTuner.main(argparser.parse_args())
