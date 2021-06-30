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
from opentuner.search.manipulator import BooleanParameter
from opentuner.search.manipulator import IntegerParameter
from opentuner import ConfigurationManipulator
from opentuner import MeasurementInterface
from opentuner import Result
from signal import signal, SIGINT
import json
import glob
import sys
import re
import os


class SmmTuner(MeasurementInterface):
    def manipulator(self):
        """Setup common state and define search space"""
        # sanitize input arguments
        self.args.m = max(self.args.m, 1)
        self.args.n = [max(self.args.n, 1), self.args.m][0 == self.args.n]
        self.args.k = [max(self.args.k, 1), self.args.m][0 == self.args.k]
        self.args.mb = max(self.args.mb, 1)
        self.args.bs = max(min(self.args.bs, self.args.mb), 1)
        self.args.bm = [max(self.args.bm, 1), self.args.m][0 == self.args.bm]
        self.args.bn = [max(self.args.bn, 1), 1][0 == self.args.bn]
        self.gflops = 0
        self.exepath = "../.."
        self.exename = "acc_bench_smm"
        # verbosity level to capture device name and tuned parameters
        run_result = self.launch(["ACC_OPENCL_VERBOSE=2", "CHECK=0"], 1, 1)
        if 0 == run_result["returncode"]:
            typename = re.search(
                "typename \\(id=([0-9]+)\\):\\s+(\\w+)", str(run_result["stdout"])
            )
            self.typename = typename.group(2) if typename and typename.group(2) else ""
            self.typeid = (
                int(typename.group(1)) if typename and typename.group(1) else 0
            )
            device = re.search(
                'INFO ACC/OpenCL:\\s+ndevices=[0-9]+\\s+device[0-9]+="(.+)"',
                str(run_result["stderr"]),
            )
            self.device = device.group(1) if device and device.group(1) else ""
            params = re.search(
                "INFO ACC/OpenCL:\\s+{}x{}x{}\\s+{}SMM-kernel{}".format(
                    self.args.m,
                    self.args.n,
                    self.args.k,
                    {"float": "S", "double": "D"}.get(self.typename, ""),
                    "\\s+bs=([0-9]+)\\s+bm=([0-9]+)\\s+bn=([0-9]+)\\s+bc=([0-9]+)\\s+gen=",
                ),
                str(run_result["stderr"]),
            )
            self.bs = int(params.group(1)) if params and params.group(1) else None
            self.bm = int(params.group(2)) if params and params.group(2) else None
            self.bn = int(params.group(3)) if params and params.group(3) else None
            self.bc = (
                (0 != int(params.group(4))) if params and params.group(4) else None
            )
        else:
            self.typename = self.typeid = None
        if self.typename and self.typeid:
            # construct label used for the database session
            if not self.args.label:
                self.args.label = "multiply-{}x{}x{}-{}{}".format(
                    self.args.m,
                    self.args.n,
                    self.args.k,
                    self.typename,
                    ["", " " + self.device]["" != self.device],
                )
        elif not (self.args.merge or self.args.update):
            sys.tracebacklimit = 0
            raise RuntimeError(
                "Setup failed for {}/{}!".format(self.exepath, self.exename)
            )
        # consider to update and/or merge JSONS (update first)
        if self.args.update or self.args.merge:
            filenames = glob.glob("*.json")
            if self.args.update:
                self.update_jsons(filenames)
            if self.args.merge:
                self.merge_jsons(filenames)
            exit(0)
        # setup tunable parameters
        manipulator, params = ConfigurationManipulator(), []
        if not os.getenv("OPENCL_LIBSMM_SMM_BS"):
            params.append(IntegerParameter("BS", 1, self.args.mb))
        if not os.getenv("OPENCL_LIBSMM_SMM_BM"):
            params.append(IntegerParameter("BM", 1, self.args.m))
        if not os.getenv("OPENCL_LIBSMM_SMM_BN"):
            params.append(IntegerParameter("BN", 1, self.args.n))
        if not os.getenv("OPENCL_LIBSMM_SMM_BC"):
            params.append(BooleanParameter("BC"))
        if not params:
            sys.tracebacklimit = 0
            raise RuntimeError(
                "All tunable parameters are fixed with environment variables!"
            )
        for param in params:
            manipulator.add_parameter(param)
        # register signal handler (CTRL-C)
        signal(SIGINT, self.handle_sigint)
        return manipulator

    def launch(self, envs, nrep=None, size=None):
        """Launch executable supplying environment and arguments"""
        return self.call_program(
            "OMP_PROC_BIND=TRUE {} {} {} {}".format(
                " ".join(map(str, envs)),  # environment variables
                "{}/{}".format(self.exepath, self.exename),
                # executable's arguments
                "{} {}".format(
                    self.args.r if None else nrep, self.args.s if None else size
                ),
                "{} {} {}".format(self.args.m, self.args.n, self.args.k),
            )
        )

    def seed_configurations(self):
        return [
            {
                "BS": self.bs if self.bs is not None else self.args.bs,
                "BM": self.bm if self.bm is not None else self.args.bm,
                "BN": self.bn if self.bn is not None else self.args.bn,
                "BC": self.bc if self.bc is not None else self.args.bc,
            }
        ]

    def objective(self):
        if not self.args.primary:
            return opentuner.search.objective.MaximizeAccuracyMinimizeSize()
        else:
            return opentuner.search.objective.MaximizeAccuracy()

    def environment(self, config):
        return [
            "OPENCL_LIBSMM_SMM_BS={}".format(config["BS"]),
            "OPENCL_LIBSMM_SMM_BM={}".format(config["BM"]),
            "OPENCL_LIBSMM_SMM_BN={}".format(config["BN"]),
            "OPENCL_LIBSMM_SMM_BC={}".format(int(config["BC"])),
        ]

    def run(self, desired_result, input, limit):
        """Run a configuration and return performance"""
        config = desired_result.configuration.data
        run_result = self.launch(
            self.environment(config) + ["CHECK={}".format(self.args.check)]
        )
        if 0 == run_result["returncode"]:
            performance = re.search(
                "device:\\s+([0-9]+(\\.[0-9]*)*) ms\\s+([0-9]+(\\.[0-9]*)*)",
                str(run_result["stdout"]),
            )
        else:
            performance = None
        if performance and performance.group(1) and performance.group(3):
            mseconds = float(performance.group(1))
            gflops = float(performance.group(3))
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

    def update_jsons(self, filenames):
        """Update device name of all JSONs"""
        if self.device:
            updated = False
            for filename in filenames:
                try:
                    with open(filename, "r") as file:
                        data = json.load(file)
                        device = data["DEVICE"] if "DEVICE" in data else ""
                        if device != self.device:
                            data.update({"DEVICE": self.device})
                            file.close()
                            with open(filename, "w") as file:
                                json.dump(data, file)
                                file.write("\n")
                                print("Updated {} to {}.".format(filename, self.device))
                                updated = True
                except (json.JSONDecodeError, KeyError):
                    print("Failed to update {}.".format(filename))
            if not updated:
                print("All JSONs target {}.".format(self.device))
        else:
            print("Cannot determine device name.")

    def merge_jsons(self, filenames):
        """Merge all JSONs into a single CSV-file"""
        if self.args.csvfile:
            merged = dict()
            for filename in filenames:
                try:
                    data = dict()
                    with open(filename, "r") as file:
                        data = json.load(file)
                    key = (
                        data["DEVICE"] if "DEVICE" in data else self.device,
                        int(data["TYPEID"]),
                        data["M"],
                        data["N"],
                        data["K"],
                    )
                    value = (
                        data["GFLOPS"],
                        data["BS"],
                        data["BM"],
                        data["BN"],
                        int(data["BC"]) if "BC" in data else 0,
                        filename,
                    )
                    if key not in merged:
                        merged[key] = value
                    else:
                        if merged[key][0] < value[0]:
                            filename = merged[key][-1]
                            merged[key] = value
                        print(
                            "Worse result {} ignored when merging CSV-file".format(
                                filename
                            )
                        )
                except (json.JSONDecodeError, KeyError):
                    print("Failed to merge {} into CSV-file.".format(filename))
            if bool(merged):
                with open(self.args.csvfile, "w") as file:
                    file.write(  # CSV header line (key part)
                        self.args.csvsep.join(["DEVICE", "TYPEID", "M", "N", "K"])
                    )
                    file.write(self.args.csvsep)
                    file.write(
                        self.args.csvsep.join(["GFLOPS", "BS", "BM", "BN", "BC"])
                    )
                    file.write("\n")  # CSV header line (termination)
                    for key, value in merged.items():  # CSV data lines
                        strkey = self.args.csvsep.join([str(k) for k in key])
                        strval = self.args.csvsep.join([str(v) for v in value[:-1]])
                        file.write(strkey)
                        file.write(self.args.csvsep)
                        file.write(strval)
                        file.write("\n")
                print(
                    "Merged {} of {} JSONs into {}".format(
                        len(merged), len(filenames), self.args.csvfile
                    )
                )
            elif glob.glob(self.args.csvfile):
                backup = "{}.bak".format(self.args.csvfile)
                print("Renamed {} to {}.".format(self.args.csvfile, backup))
                os.rename(self.args.csvfile, backup)

    def save_final_config(self, configuration):
        """Called at termination"""
        if 0 < self.gflops:
            filename = "tune_multiply-{}-{}x{}x{}-{}gflops.json".format(
                self.typename, self.args.m, self.args.n, self.args.k, round(self.gflops)
            )
            # extend result for easier reuse later
            config = configuration.data
            config["DEVICE"] = self.device
            config["GFLOPS"] = self.gflops
            config["TYPEID"] = self.typeid
            config["M"] = self.args.m
            config["N"] = self.args.n
            config["K"] = self.args.k
            filenames = glob.glob("*.json")
            if not filenames and glob.glob(self.args.csvfile):
                print(
                    "WARNING: no JSON file found but (unrelated?) {}".format(
                        self.args.csvfile
                    )
                )
            # self.manipulator().save_to_file(config, filename)
            with open(filename, "w") as file:
                json.dump(config, file)
                file.write("\n")  # append newline at EOF
            if filename not in filenames:
                filenames.append(filename)
                self.merge_jsons(filenames)
            print(
                "Result achieving {} GFLOPS/s ({}) was written to {}".format(
                    self.gflops, self.typename, filename
                )
            )
            if 0 == self.args.check:
                run_result = self.launch(self.environment(config) + ["CHECK=1"])
                if 0 != run_result["returncode"]:
                    print("WARNING: tuned result seems to be incorrect!")

    def handle_sigint(self, signum, frame):
        """Handle SIGINT or CTRL-C"""
        print(
            "\nWARNING: tuning {}x{}x{}-kernel was interrupted".format(
                self.args.m, self.args.n, self.args.k
            )
        )
        self.save_final_config(self.config)
        exit(1)


if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    # additional arguments
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
        default=int(os.getenv("OPENCL_LIBSMM_SMM_BM", "0")),
        nargs="?",
        dest="bm",
        help="Initial block/tile size (BM)",
    )
    argparser.add_argument(
        "-bn",
        "--initial-bn",
        type=int,
        default=int(os.getenv("OPENCL_LIBSMM_SMM_BN", "0")),
        nargs="?",
        dest="bn",
        help="Initial block/tile size (BN)",
    )
    argparser.add_argument(
        "-bc",
        "--initial-bc",
        action="store_true",
        default=True if int(os.getenv("OPENCL_LIBSMM_SMM_BC", "0")) else False,
        dest="bc",
        help="Initial bank-conflict flag (BC)",
    )
    argparser.add_argument(
        "-bs",
        "--initial-bs",
        type=int,
        default=int(os.getenv("OPENCL_LIBSMM_SMM_BS", "24")),
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
        "-r",
        "--repetitions",
        type=int,
        default=0,
        nargs="?",
        dest="r",
        help="Repetitions per experiment",
    )
    argparser.add_argument(
        "-s",
        "--batchsize",
        type=int,
        default=0,
        nargs="?",
        dest="s",
        help='Size of batch ("stacksize")',
    )
    argparser.add_argument(
        "-c",
        "--csv-separator",
        type=(lambda c: c if isinstance(c, str) and 1 == len(c) else False),
        default=";",
        nargs="?",
        dest="csvsep",
        help="Separator used in CSV-file",
    )
    argparser.add_argument(
        "-o",
        "--csv-filename",
        type=str,
        default="tune_multiply.csv",
        nargs="?",
        dest="csvfile",
        help="Generate CSV-file",
    )
    argparser.add_argument(
        "-m",
        "--csv-merge-jsons",
        action="store_true",
        default=False,
        dest="merge",
        help="Merge JSONs into CSV, and terminate",
    )
    argparser.add_argument(
        "-u",
        "--update-jsons",
        action="store_true",
        default=False,
        dest="update",
        help="Update JSONs (device), and terminate",
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
    # adjust default value of existing arguments
    argparser.set_defaults(no_dups=True)
    SmmTuner.main(argparser.parse_args())
