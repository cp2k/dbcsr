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
        self.exepath = "../.."
        self.exename = "acc_bench_smm"
        run_result = self.call_program(self.exepath + "/" + self.exename + " 1 1 1")
        if 0 == run_result["returncode"]:
            match = re.search(
                "typename \\(id=([0-9]+)\\):\\s+(\\w+)", str(run_result["stdout"])
            )
        else:
            match = None
        if (match is not None) and match.group(1) and match.group(2):
            self.typename = match.group(2)
            self.typeid = match.group(1)
        else:
            sys.tracebacklimit = 0
            raise RuntimeError(
                "Setup failed for " + self.exepath + "/" + self.exename + "!"
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
        run_cmd = (
            "OMP_PROC_BIND=TRUE CHECK="
            + str(self.args.check)
            + " OPENCL_LIBSMM_SMM_BATCHSIZE="
            + str(config["BS"])
            + " OPENCL_LIBSMM_SMM_BLOCK_M="
            + str(config["BM"])
            + " OPENCL_LIBSMM_SMM_BLOCK_N="
            + str(config["BN"])
            + " "
            + self.exepath
            + "/"
            + self.exename
            + " 0 0"
            + " "
            + str(self.args.m)
            + " "
            + str(self.args.n)
            + " "
            + str(self.args.k)
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

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        if 0 < self.gflops:
            ofilename = (
                "tune_multiply-"
                + self.typename
                + "-"
                + str(self.args.m)
                + "x"
                + str(self.args.n)
                + "x"
                + str(self.args.k)
                + "-"
                + str(round(self.gflops))
                + "gflops.json"
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
                    "WARNING: no JSON file found but (unrelated?) "
                    + self.args.csvfile
                    + " exists!"
                )
            # self.manipulator().save_to_file(config, ofilename)
            with open(ofilename, "w") as ofile:
                json.dump(config, ofile)
                ofile.write("\n")  # append newline at EOF
                print(
                    "Result achieving "
                    + str(self.gflops)
                    + " GFLOPS/s ("
                    + self.typename
                    + ") was written to "
                    + ofilename
                )
                if ofilename not in filenames:
                    filenames.append(ofilename)
            # merge all JSONs into a single CSV file
            if self.args.csvfile:
                merged = dict()
                for ifilename in filenames:
                    with open(ifilename, "r") as ifile:
                        data = json.load(ifile)
                        try:
                            key = (data["TYPEID"], data["M"], data["N"], data["K"])
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
                                    "Worse result "
                                    + ifilename
                                    + " ignored when merging CSV file"
                                )
                        except KeyError:
                            print(
                                "Malformed "
                                + ifilename
                                + " ignored when merging CSV file"
                            )
                            pass
                if bool(merged):
                    with open(self.args.csvfile, "w") as ofile:
                        ofile.write(  # CSV header line
                            self.args.csvsep.join(
                                ["TYPEID", "M", "N", "K", "GFLOPS", "BS", "BM", "BN"]
                            )
                            + "\n"
                        )
                        for key, value in merged.items():  # CSV data lines
                            strkey = self.args.csvsep.join([str(k) for k in key])
                            strval = self.args.csvsep.join([str(v) for v in value[:-1]])
                            ofile.write(strkey + self.args.csvsep + strval + "\n")
                    print(
                        "Merged "
                        + str(len(merged))
                        + " of "
                        + str(len(filenames))
                        + " JSONs into "
                        + self.args.csvfile
                    )

    def handle_sigint(self, signum, frame):
        """handles SIGINT or CTRL-C"""
        print(
            "\nWARNING: tuning "
            + str(self.args.m)
            + "x"
            + str(self.args.n)
            + "x"
            + str(self.args.k)
            + "-kernel was interrupted."
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
