#!/usr/bin/env python3
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################
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
from signal import signal, SIGINT
import json
import glob
import sys
import re


class SmmTuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        if self.args.merge:
            self.merge_into_csv(glob.glob("*.json"))
            exit(0)
        self.exepath = "../.."
        self.exename = "acc_bench_smm"
        run_result = self.call_program("{}/{} 1 1 1".format(self.exepath, self.exename))
        if 0 == run_result["returncode"]:
            match = re.search(
                "typename \\(id=([0-9]+)\\):\\s+(\\w+)", str(run_result["stdout"])
            )
        else:
            match = None
        if (match is not None) and match.group(1) and match.group(2):
            self.typename = match.group(2)
            self.typeid = int(match.group(1))
        else:
            sys.tracebacklimit = 0
            raise RuntimeError(
                "Setup failed for {}/{}!".format(self.exepath, self.exename)
            )
        # sanitize input arguments
        self.args.m = max(self.args.m, 1)
        self.args.n = [max(self.args.n, 1), self.args.m][0 == self.args.n]
        self.args.k = [max(self.args.k, 1), self.args.m][0 == self.args.k]
        self.args.mb = max(self.args.mb, 1)
        self.args.bs = max(min(self.args.bs, self.args.mb), 1)
        self.args.bm = [max(self.args.bm, 1), self.args.m][0 == self.args.bm]
        self.args.bn = [max(self.args.bn, 1), 1][0 == self.args.bn]
        self.gflops = 0
        # setup tunable parameters
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(IntegerParameter("BS", 1, self.args.mb))
        manipulator.add_parameter(IntegerParameter("BM", 1, self.args.m))
        manipulator.add_parameter(IntegerParameter("BN", 1, self.args.n))
        # register signal handler (CTRL-C)
        signal(SIGINT, self.handle_sigint)
        return manipulator

    def seed_configurations(self):
        return [{"BS": self.args.bs, "BM": self.args.bm, "BN": self.args.bn}]

    def objective(self):
        if not self.args.primary:
            return opentuner.search.objective.MaximizeAccuracyMinimizeSize()
        else:
            return opentuner.search.objective.MaximizeAccuracy()

    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        config = desired_result.configuration.data
        run_cmd = "{} CHECK={} {}={} {}={} {}={} {}/{} 0 0 {} {} {}".format(
            "OMP_PROC_BIND=TRUE",
            self.args.check,
            "OPENCL_LIBSMM_SMM_BATCHSIZE",
            config["BS"],
            "OPENCL_LIBSMM_SMM_BLOCK_M",
            config["BM"],
            "OPENCL_LIBSMM_SMM_BLOCK_N",
            config["BN"],
            self.exepath,
            self.exename,
            self.args.m,
            self.args.n,
            self.args.k,
        )
        run_result = self.call_program(run_cmd)
        if 0 == run_result["returncode"]:
            match = re.search(
                "device:\\s+([0-9]+(\\.[0-9]*)*) ms\\s+([0-9]+(\\.[0-9]*)*)",
                str(run_result["stdout"]),
            )
        else:
            match = None
        if (match is not None) and match.group(1) and match.group(3):
            mseconds = float(match.group(1))
            gflops = float(match.group(3))
            if self.gflops < gflops:
                # keep best configuration in case of an early exit
                self.config = desired_result.configuration
                self.gflops = gflops
            kernelreq = round(
                (100.0 * config["BM"] * config["BN"]) / (self.args.m * self.args.n)
            )
            # gflops are reported as "accuracy" (console output)
            return Result(time=mseconds, accuracy=gflops, size=kernelreq)
        else:  # return non-competitive/bad result in case of an error
            return Result(time=float("inf"), accuracy=0.0, size=100.0)

    def merge_into_csv(self, filenames):
        """merge all JSONs into a single CSV-file"""
        if self.args.csvfile:
            merged = dict()
            for ifilename in filenames:
                try:
                    data = dict()
                    with open(ifilename, "r") as ifile:
                        data = json.load(ifile)
                    key = (int(data["TYPEID"]), data["M"], data["N"], data["K"])
                    value = (
                        data["GFLOPS"],
                        data["BS"],
                        data["BM"],
                        data["BN"],
                        ifilename,
                    )
                    if key not in merged:
                        merged[key] = value
                    else:
                        if merged[key][0] < value[0]:
                            ifilename = merged[key][-1]
                            merged[key] = value
                        print(
                            "Worse result {} ignored when merging CSV-file.".format(
                                ifilename
                            )
                        )
                except (json.JSONDecodeError, KeyError):
                    print("Failed to merge {} into CSV-file.".format(ifilename))
            if bool(merged):
                with open(self.args.csvfile, "w") as ofile:
                    ofile.write(  # CSV header line
                        self.args.csvsep.join(
                            ["TYPEID", "M", "N", "K", "GFLOPS", "BS", "BM", "BN\n"]
                        )
                    )
                    for key, value in merged.items():  # CSV data lines
                        strkey = self.args.csvsep.join([str(k) for k in key])
                        strval = self.args.csvsep.join([str(v) for v in value[:-1]])
                        ofile.write(strkey)
                        ofile.write(self.args.csvsep)
                        ofile.write(strval)
                        ofile.write("\n")
                print(
                    "Merged {} of {} JSONs into {}.".format(
                        len(merged), len(filenames), self.args.csvfile
                    )
                )

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        if 0 < self.gflops:
            ofilename = "tune_multiply-{}-{}x{}x{}-{}gflops.json".format(
                self.typename, self.args.m, self.args.n, self.args.k, round(self.gflops)
            )
            # extend result for easier reuse later
            config = configuration.data
            config["GFLOPS"] = self.gflops
            config["TYPEID"] = self.typeid
            config["M"] = self.args.m
            config["N"] = self.args.n
            config["K"] = self.args.k
            filenames = glob.glob("*.json")
            if not filenames and glob.glob(self.args.csvfile):
                print(
                    "WARNING: no JSON file found but (unrelated?) {} exists!".format(
                        self.args.csvfile
                    )
                )
            # self.manipulator().save_to_file(config, ofilename)
            with open(ofilename, "w") as ofile:
                json.dump(config, ofile)
                ofile.write("\n")  # append newline at EOF
            print(
                "Result achieving {} GFLOPS/s ({}) was written to {}.".format(
                    self.gflops, self.typename, ofilename
                )
            )
            if ofilename not in filenames:
                filenames.append(ofilename)
                self.merge_into_csv(filenames)

    def handle_sigint(self, signum, frame):
        """handles SIGINT or CTRL-C"""
        print(
            "\nWARNING: tuning {}x{}x{}-kernel was interrupted.".format(
                self.args.m, self.args.n, self.args.k
            )
        )
        self.save_final_config(self.config)
        exit(1)


if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    argparser.add_argument(
        "m", type=int, default=23, nargs="?", help="Shape of SMM-kernel (M)"
    )
    argparser.add_argument(
        "n", type=int, default=0, nargs="?", help="Shape of SMM-kernel (N)"
    )
    argparser.add_argument(
        "k", type=int, default=0, nargs="?", help="Shape of SMM-kernel (K)"
    )
    argparser.add_argument(
        "-bm",
        "--initial-bm",
        type=int,
        default=0,
        nargs="?",
        dest="bm",
        help="Initial block/tile size (BM)",
    )
    argparser.add_argument(
        "-bn",
        "--initial-bn",
        type=int,
        default=0,
        nargs="?",
        dest="bn",
        help="Initial block/tile size (BN)",
    )
    argparser.add_argument(
        "-bs",
        "--initial-bs",
        type=int,
        default=32,
        nargs="?",
        dest="bs",
        help="Initial (mini-)batch size (BS)",
    )
    argparser.add_argument(
        "-mb",
        "--max-bs",
        type=int,
        default=256,
        nargs="?",
        dest="mb",
        help="Maximum (mini-)batch size (BS)",
    )
    argparser.add_argument(
        "-s",
        "--csv-separator",
        type=(lambda c: c if isinstance(c, str) and 1 == len(c) else False),
        default=";",
        nargs="?",
        dest="csvsep",
        help="Separator used in CSV-file",
    )
    argparser.add_argument(
        "-c",
        "--csv-filename",
        type=str,
        default="tune_multiply.csv",
        nargs="?",
        dest="csvfile",
        help="Generate CSV-file",
    )
    argparser.add_argument(
        "-m",
        "--csv-merge-only",
        action="store_true",
        default=False,
        dest="merge",
        help="Merge JSONs into CSV, and terminate",
    )
    argparser.add_argument(
        "-v",
        "--check",
        type=float,
        default=0,
        nargs="?",
        dest="check",
        help="Validate kernel (epsilon)",
    )
    argparser.add_argument(
        "-p",
        "--primary-objective",
        action="store_true",
        default=False,
        dest="primary",
        help="Primary objective only",
    )
    SmmTuner.main(argparser.parse_args())
