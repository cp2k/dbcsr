#!/usr/bin/env python3
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################
import opentuner
from opentuner.search.manipulator import IntegerParameter
from opentuner import ConfigurationManipulator
from opentuner import MeasurementInterface
from opentuner import Result
from signal import signal, getsignal, SIGINT
import json
import glob
import sys
import re
import os

default_basename = "tune_multiply"
default_mnk = "23x23x23"


def env_isfixed(envname):
    strvalue = os.getenv(envname)
    if strvalue:
        try:
            ivalue = int(strvalue)
            return ivalue == ivalue
        except ValueError:
            pass
    return False


def env_value(envname, default):
    strvalue = os.getenv(envname, default)
    try:
        return int(strvalue)
    except ValueError:
        pass
    return int(default)


def ilog2(n):
    i, t = (0 if 1 != n else 1), 1
    while t < n:
        t <<= 1
        i += 1
    return i


class SmmTuner(MeasurementInterface):
    def manipulator(self):
        """Setup common state and define search space"""
        manipulator = ConfigurationManipulator()
        # parse and sanitize kernel shape argument
        if not self.args.mnk:
            self.args.mnk = default_mnk
        mnk = tuple(max(int(i), 1) for i in self.args.mnk.split("x"))
        self.mnk = (mnk + (mnk[0], mnk[0]))[:3]
        # sanitize input arguments
        self.args.mb = max(self.args.mb, 1)
        self.args.bs = max(min(self.args.bs, self.args.mb), 1)
        self.args.bm = [max(self.args.bm, 1), self.mnk[0]][0 == self.args.bm]
        self.args.bn = [max(self.args.bn, 1), 1][0 == self.args.bn]
        self.args.bk = [max(self.args.bk, 1), self.mnk[2]][0 == self.args.bk]
        self.args.ws = min(self.args.ws, self.mnk[0] * self.mnk[1])
        self.bs = self.bm = self.bn = self.bk = self.ws = self.wg = self.lu = None
        self.nz = self.al = self.tb = self.tc = None
        self.ap = self.aa = self.ab = self.ac = None
        self.typename = self.typeid = None
        self.device = self.size = None
        self.gfbase = self.gflops = 0
        self.config = None
        self.exename = "acc_bench_smm"
        self.exepath = os.path.join(
            os.path.dirname(sys.argv[0]), "..", "..", self.exename
        )
        run_result = (  # verbosity to capture device name and tuned parameters
            self.launch(["ACC_OPENCL_VERBOSE=2", "CHECK=0"], nrep=1)
            if (self.args.merge is None or 0 > self.args.merge)
            and (self.args.update is None or "" == self.args.update)
            else None
        )
        if run_result:
            stdout = str(run_result["stdout"])
            if 0 >= self.args.size:
                size = re.search(
                    "{}\\s+[0-9]+\\s+([0-9]+)".format(self.exepath),
                    stdout,
                )
                self.size = int(size.group(1)) if size and size.group(1) else 0
            else:
                self.size = self.args.size
            typename = re.search("typename \\(id=([0-9]+)\\):\\s+(\\w+)", stdout)
            self.typename = typename.group(2) if typename and typename.group(2) else ""
            self.typeid = (
                int(typename.group(1)) if typename and typename.group(1) else 0
            )
            device = re.search(
                'INFO ACC/OpenCL:\\s+ndevices=[0-9]+\\s+device[0-9]+="([^"]+)"',
                str(run_result["stderr"]),
            )
            self.device = device.group(1) if device and device.group(1) else ""
        elif self.args.update is not None and "" != self.args.update:
            self.device = self.args.update
        if run_result and 0 == run_result["returncode"]:
            seedpat = "INFO ACC/OpenCL:\\s+SMM-kernel\\s+{}={}\\s+gen=".format(
                "{t,m,n,k, bs,bm,bn,bk, ws,wg, lu,nz,al, tb,tc, ap,aa,ab,ac}",
                "{{{}, {}}}".format(  # key and value
                    "{},{},{},{}".format(  # t,m,n,k (key)
                        self.typeid, self.mnk[0], self.mnk[1], self.mnk[2]
                    ),
                    "{}, {}, {}, {}, {}".format(  # value: some can be neg. hence "-*[0-9]+"
                        "(-*[0-9]+),(-*[0-9]+),(-*[0-9]+),(-*[0-9]+)",  # bs,bm,bn,bk
                        "(-*[0-9]+),(-*[0-9]+)",  # ws,wg
                        "(-*[0-9]+),(-*[0-9]+),(-*[0-9]+)",  # lu,nz,al
                        "(-*[0-9]+),(-*[0-9]+)",  # tb,tc
                        "(-*[0-9]+),(-*[0-9]+),(-*[0-9]+),(-*[0-9]+)",  # ap,aa,ab,ac
                    ),
                ),
            )
            seed = re.search(seedpat, str(run_result["stderr"]))
            if 15 != (len(seed.groups()) if seed else 0):
                print("WARNING: missed to parse initial parameters!")
            # setup fixed and tunable parameters
            params, paramt = [], []
            self.create_param("BS", params, paramt, seed, 1, 1, self.args.mb)
            self.create_param("BM", params, paramt, seed, 2, 1, self.mnk[0])
            self.create_param("BN", params, paramt, seed, 3, 1, self.mnk[1])
            self.create_param("BK", params, paramt, seed, 4, 1, self.mnk[0])
            self.create_param(
                "WS", params, paramt, seed, 5, 1, self.mnk[0] * self.mnk[1]
            )
            self.create_param("WG", params, paramt, seed, 6, -2, 1)  # avoid WG=2
            self.create_param("LU", params, paramt, seed, 7, -2, 2)
            self.create_param("NZ", params, paramt, seed, 8, 0, 1)
            self.create_param("AL", params, paramt, seed, 9, 0, 1)
            self.create_param("TB", params, paramt, seed, 10, 0, 1)
            self.create_param("TC", params, paramt, seed, 11, 0, 1)
            self.create_param("AP", params, paramt, seed, 12, 0, 1)
            self.create_param("AA", params, paramt, seed, 13, 0, 3)
            self.create_param("AB", params, paramt, seed, 14, 0, 3)
            self.create_param("AC", params, paramt, seed, 15, 0, 2)
            if not paramt:
                sys.tracebacklimit = 0
                raise RuntimeError(
                    "All tunable parameters are fixed with environment variables!"
                )
            for param in params + paramt:
                manipulator.add_parameter(param)
        # consider to update and/or merge JSONS (update first)
        if (
            (self.args.merge is not None and (0 <= self.args.merge or self.typeid))
            or self.args.update is None
            or "" != self.args.update
        ):
            filepattern = "{}-*.json".format(default_basename)
            filenames = glob.glob(
                os.path.normpath(os.path.join(self.args.jsondir, filepattern))
            )
            if self.args.update is None or "" != self.args.update:
                self.update_jsons(filenames)
            if self.args.merge is not None:
                self.merge_jsons(filenames)
            exit(0)
        elif (
            (self.typename and "" != self.typename)
            and (self.device and "" != self.device)
            and (self.size and 0 < self.size)
            and self.typeid
        ):  # construct label used for the database session
            if not self.args.label:  # consider to include self.device
                self.args.label = "{}-{}-{}x{}x{}-s{}".format(
                    default_basename,
                    self.typename,
                    self.mnk[0],
                    self.mnk[1],
                    self.mnk[2],
                    ilog2(self.size),
                )
        else:
            sys.tracebacklimit = 0
            raise RuntimeError("Setup failed for {}!".format(self.exepath))
        # register signal handler (CTRL-C)
        signal(SIGINT, self.handle_sigint)
        return manipulator

    def create_param(self, name, params, paramt, match, match_id, value0, value1):
        """Append integer-parameter to either params or paramt list"""
        if env_isfixed("OPENCL_LIBSMM_SMM_{}".format(name)):
            value_fix = getattr(self.args, name.lower())
            params.append(IntegerParameter(name, value_fix, value_fix))
        else:
            value = (
                int(match.group(match_id)) if match and match.group(match_id) else None
            )
            setattr(self, name.lower(), value)
            paramt.append(IntegerParameter(name, value0, value1))

    def launch(self, envs, nrep=None, verbose=None):
        """Launch executable supplying environment and arguments"""
        envstrs = " ".join(map(str, envs))
        if verbose is not None and 0 != int(verbose):
            print(envstrs.replace("OPENCL_LIBSMM_SMM_", "").replace(" CHECK=0", ""))
        return self.call_program(
            "OMP_PROC_BIND=TRUE OPENCL_LIBSMM_SMM_S=0 {} {} {} {} {}".format(
                envstrs,  # environment variables
                "{}".format(self.exepath),
                # executable's arguments
                self.args.r if nrep is None else nrep,
                self.size if self.size else self.args.size,
                "{} {} {}".format(self.mnk[0], self.mnk[1], self.mnk[2]),
            )
        )

    def seed_configurations(self):
        return [
            {
                "BS": self.bs if self.bs is not None else self.args.bs,
                "BM": self.bm if self.bm is not None else self.args.bm,
                "BN": self.bn if self.bn is not None else self.args.bn,
                "BK": self.bk if self.bk is not None else self.args.bk,
                "WS": self.ws if self.ws is not None else self.args.ws,
                "WG": self.wg if self.wg is not None else self.args.wg,
                "NZ": self.nz if self.nz is not None else self.args.nz,
                "LU": self.lu if self.lu is not None else self.args.lu,
                "AL": self.al if self.al is not None else self.args.al,
                "TB": self.tb if self.tb is not None else self.args.tb,
                "TC": self.tc if self.tc is not None else self.args.tc,
                "AP": self.ap if self.ap is not None else self.args.ap,
                "AA": self.aa if self.aa is not None else self.args.aa,
                "AB": self.ab if self.ab is not None else self.args.ab,
                "AC": self.ac if self.ac is not None else self.args.ac,
            }
        ]

    def objective(self):
        if 0 == args.tlevel:
            return opentuner.search.objective.MaximizeAccuracyMinimizeSize()
        else:
            return opentuner.search.objective.MaximizeAccuracy()

    def environment(self, config):
        return [
            "OPENCL_LIBSMM_SMM_BS={}".format(config["BS"]),
            "OPENCL_LIBSMM_SMM_BM={}".format(config["BM"]),
            "OPENCL_LIBSMM_SMM_BN={}".format(config["BN"]),
            "OPENCL_LIBSMM_SMM_BK={}".format(config["BK"]),
            "OPENCL_LIBSMM_SMM_WS={}".format(config["WS"]),
            "OPENCL_LIBSMM_SMM_WG={}".format(config["WG"]),
            "OPENCL_LIBSMM_SMM_LU={}".format(config["LU"]),
            "OPENCL_LIBSMM_SMM_NZ={}".format(config["NZ"]),
            "OPENCL_LIBSMM_SMM_AL={}".format(config["AL"]),
            "OPENCL_LIBSMM_SMM_TB={}".format(config["TB"]),
            "OPENCL_LIBSMM_SMM_TC={}".format(config["TC"]),
            "OPENCL_LIBSMM_SMM_AP={}".format(config["AP"]),
            "OPENCL_LIBSMM_SMM_AA={}".format(config["AA"]),
            "OPENCL_LIBSMM_SMM_AB={}".format(config["AB"]),
            "OPENCL_LIBSMM_SMM_AC={}".format(config["AC"]),
        ]

    def run(self, desired_result, input, limit):
        """Run a configuration and return performance"""
        config = desired_result.configuration.data
        cfgenv = self.environment(config)
        run_result = self.launch(
            cfgenv + ["CHECK={}".format(self.args.check)],
            verbose=self.args.verbose,
        )
        if 0 == run_result["returncode"]:
            performance = re.search(
                "device:\\s+([0-9]+[^ ]*) ms\\s+([0-9]+[^ ]*)",
                str(run_result["stdout"]),
            )
        else:
            failed = " ".join(map(str, cfgenv)).replace("OPENCL_LIBSMM_SMM_", "")
            print("FAILED: {}".format(failed))
            performance = None
        if performance and performance.group(1) and performance.group(2):
            mseconds = float(performance.group(1))
            gflops = float(performance.group(2))
            if self.gflops < gflops:
                # keep best configuration in case of an early exit
                self.config = desired_result.configuration
                self.gflops = gflops
                if 0 == self.gfbase:  # seed configuration
                    self.gfbase = gflops
                self.save_final_config(desired_result.configuration, final=False)
            kernelreq = round(
                (100.0 * config["BM"] * config["BN"]) / (self.mnk[0] * self.mnk[1])
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
                            print("Updated {} to {}.".format(filename, self.device))
                            data.update({"DEVICE": self.device})
                            file.close()
                            updated = True
                        # rewrite JSON (in any case) with keys in order
                        with open(filename, "w") as file:
                            json.dump(data, file, sort_keys=True)
                            file.write("\n")
                except (json.JSONDecodeError, KeyError):
                    print("Failed to update {}.".format(filename))
            if not updated:
                print("All JSONs already target {}.".format(self.device))
        else:
            print("Cannot determine device name.")

    def merge_jsons(self, filenames):
        """Merge all JSONs into a single CSV-file"""
        if self.args.csvfile:
            merged, worse = dict(), dict()
            for filename in filenames:
                try:
                    data = dict()
                    with open(filename, "r") as file:
                        data = json.load(file)
                    if self.args.merge is not None and (
                        (0 > self.args.merge and self.typeid != data["TYPEID"])
                        or (1 == self.args.merge and 1 != data["TYPEID"])
                        or (2 == self.args.merge and 3 != data["TYPEID"])
                    ):
                        continue
                    device = data["DEVICE"] if "DEVICE" in data else self.device
                    key = (
                        device,
                        data["TYPEID"],
                        data["M"],
                        data["N"],
                        data["K"],
                    )
                    value = (
                        data["S"] if "S" in data else 0,  # pseudo key component
                        data["GFLOPS"]
                        if "GFLOPS" in data and not self.args.nogflops
                        else 0,
                        data["BS"],
                        data["BM"],
                        data["BN"],
                        data["BK"] if "BK" in data else 0,
                        data["WS"] if "WS" in data else 0,
                        data["WG"] if "WG" in data else 0,
                        data["LU"] if "LU" in data else 0,
                        data["NZ"] if "NZ" in data else 0,
                        data["AL"] if "AL" in data else 0,
                        data["TB"] if "TB" in data else 0,
                        data["TC"] if "TC" in data else 1,
                        data["AP"] if "AP" in data else 0,
                        data["AA"] if "AA" in data else 1,
                        data["AB"] if "AB" in data else 3,
                        data["AC"] if "AC" in data else 0,
                        filename,  # last entry
                    )
                    if key not in merged:
                        merged[key] = value
                    else:
                        filename2 = merged[key][-1]
                        if merged[key][1] <= value[1]:  # GFLOPS
                            merged[key] = value
                        else:
                            filename2 = filename
                        if key in worse:
                            worse[key].append(filename2)
                        else:
                            worse[key] = [filename2]
                except (json.JSONDecodeError, KeyError, TypeError):
                    print("Failed to merge {} into CSV-file.".format(filename))
            if bool(merged):
                with open(self.args.csvfile, "w") as file:
                    file.write(  # CSV header line with termination/newline
                        "{}{}{}{}{}{}{}{}{}\n".format(  # key-part
                            self.args.csvsep.join(["DEVICE", "TYPEID", "M", "N", "K"]),
                            self.args.csvsep,  # separator for value-part
                            "S",  # pseudo-key component
                            self.args.csvsep,
                            self.args.csvsep.join(["GFLOPS", "BS", "BM", "BN", "BK"]),
                            self.args.csvsep,
                            self.args.csvsep.join(["WS", "WG", "LU", "NZ", "AL"]),
                            self.args.csvsep,
                            self.args.csvsep.join(["TB", "TC", "AP", "AA", "AB", "AC"]),
                        )
                    )
                    for key, value in sorted(merged.items()):  # CSV data lines
                        strkey = self.args.csvsep.join([str(k) for k in key])
                        strval = self.args.csvsep.join([str(v) for v in value[:-1]])
                        file.write("{}{}{}\n".format(strkey, self.args.csvsep, strval))
                    retain, delete = [], []
                    for key, value in worse.items():
                        mtime = os.path.getmtime(merged[key][-1])
                        for filename in value:
                            if mtime < os.path.getmtime(filename):
                                retain.append(filename)
                            else:
                                delete.append(filename)
                    if retain:
                        print("Worse and newer (retain): {}".format(" ".join(retain)))
                    if delete:
                        print("Worse and older (delete): {}".format(" ".join(delete)))
                print(
                    "Merged {} of {} JSONs into {}".format(
                        len(merged), len(filenames), self.args.csvfile
                    )
                )
            elif glob.glob(self.args.csvfile):
                backup = "{}.bak".format(self.args.csvfile)
                print("Renamed {} to {}.".format(self.args.csvfile, backup))
                os.rename(self.args.csvfile, backup)

    def save_final_config(self, configuration, final=True):
        """Called at termination"""
        if 0 < self.gflops and configuration:
            # extend result for easier reuse later
            config = configuration.data
            config["DEVICE"] = self.device
            config["GFLOPS"] = self.gflops if not self.args.nogflops else 0
            config["TYPEID"] = self.typeid
            config["M"] = self.mnk[0]
            config["N"] = self.mnk[1]
            config["K"] = self.mnk[2]
            config["S"] = self.size
            filepattern = "{}-*.json".format(default_basename)
            filenames = (
                glob.glob(
                    os.path.normpath(os.path.join(self.args.jsondir, filepattern))
                )
                if final
                else None
            )
            # self.manipulator().save_to_file(config, filename)
            with open(
                os.path.join(self.args.jsondir, ".{}.json".format(self.args.label)),
                "w",
            ) as file:
                json.dump(config, file, sort_keys=True)
                file.write("\n")  # append newline at EOF
            if final:
                if not filenames and glob.glob(self.args.csvfile):
                    print(
                        "WARNING: no JSON file found but (unrelated?) {}".format(
                            self.args.csvfile
                        )
                    )
                filename = os.path.normpath(
                    os.path.join(
                        self.args.jsondir,
                        "{}-{}gflops.json".format(self.args.label, round(self.gflops)),
                    )
                )
                os.rename(
                    os.path.join(self.args.jsondir, ".{}.json".format(self.args.label)),
                    filename,
                )
                if filename not in filenames:
                    filenames.append(filename)
                    self.merge_jsons(filenames)
                speedup = round(
                    (self.gflops / self.gfbase) if 0 < self.gfbase else 0, 1
                )
                print(
                    "Result{} was written to {}".format(
                        " ({}x over seed)".format(speedup) if 1 < speedup else "",
                        filename,
                    )
                )
                # no validation in SIGINT (user may fired signal due to apps misbehavior)
                if 0 == self.args.check and self.handle_sigint != getsignal(SIGINT):
                    run_result = self.launch(self.environment(config) + ["CHECK=1"])
                    if 0 != run_result["returncode"]:
                        print("WARNING: tuned result seems to be incorrect!")

    def handle_sigint(self, signum, frame):
        """Handle SIGINT or CTRL-C"""
        print(
            "\nWARNING: tuning {}x{}x{}-kernel was interrupted".format(
                self.mnk[0], self.mnk[1], self.mnk[2]
            )
        )
        self.save_final_config(self.config)
        exit(1)


if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    # adjust default value of existing arguments
    argparser.set_defaults(no_dups=True)
    # add primary arguments (parsed first)
    argparser.add_argument(
        "mnk",
        type=str,
        default=default_mnk,
        nargs="?",
        help="Shape of SMM-kernel (MxNxK)",
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
        "-e",
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
        default="{}.csv".format(default_basename),
        nargs="?",
        dest="csvfile",
        help="Generate CSV-file",
    )
    argparser.add_argument(
        "-m",
        "--csv-merge-jsons",
        type=int,
        default=None,
        const=-1,
        nargs="?",
        dest="merge",
        help="Merge JSONs into CSV (-1: auto, 0: all, 1: SP, 2: DP)",
    )
    argparser.add_argument(
        "-x",
        "--csv-nogflops",
        action="store_true",
        default=False,
        dest="nogflops",
        help="Exclude real GFLOPS",
    )
    argparser.add_argument(
        "-p",
        "--jsons-dir",
        type=str,
        default=".",
        nargs="?",
        dest="jsondir",
        help="Directory to read/write JSONs",
    )
    argparser.add_argument(
        "-u",
        "--jsons-update",
        type=str,
        default="",
        nargs="?",
        dest="update",
        help="Update JSONs (device name optional)",
    )
    argparser.add_argument(
        "-c",
        "--check",
        type=float,
        default=0,
        nargs="?",
        dest="check",
        help="Validate kernel (epsilon, 0:off)",
    )
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        dest="verbose",
        help="Verbose output",
    )
    argparser.add_argument(
        "-a",
        "--tuning-level",
        type=int,
        default=-1,
        nargs="?",
        dest="tlevel",
        help="Tunables: (0) all, (1) most, (2) some, (3) least",
    )
    argparser.add_argument(
        "-bm",
        "--initial-bm",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_BM", "0"),
        nargs="?",
        dest="bm",
        help="Block/tile size (0:auto)",
    )
    argparser.add_argument(
        "-bn",
        "--initial-bn",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_BN", "0"),
        nargs="?",
        dest="bn",
        help="Block/tile size (0:auto)",
    )
    argparser.add_argument(
        "-bk",
        "--initial-bk",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_BK", "0"),
        nargs="?",
        dest="bk",
        help="Block size (0:auto)",
    )
    argparser.add_argument(
        "-ws",
        "--initial-ws",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_WS", "0"),
        nargs="?",
        dest="ws",
        help="Minimum WG-size (0:auto)",
    )
    argparser.add_argument(
        "-wg",
        "--initial-wg",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_WG", "0"),
        dest="wg",
        help="Size of WG: subgroups (-1), tight (0), round-up (1), PoT (2)",
    )
    argparser.add_argument(
        "-lu",
        "--initial-lu",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_LU", "-1"),
        dest="lu",
        help="Loop unroll (-2) full, (-1) no hints (default), (0) inner, (1) outer-dehint, (2) literal",
    )
    argparser.add_argument(
        "-nz",
        "--initial-nz",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_NZ", "0"),
        dest="nz",
        help="Check (1) atomic increment to be non-zero (0:off)",
    )
    argparser.add_argument(
        "-al",
        "--initial-al",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_AL", "0"),
        dest="al",
        help="Access: transposed (0), linear (1)",
    )
    argparser.add_argument(
        "-tb",
        "--initial-tb",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_TB", "0"),
        dest="tb",
        help="Matrix B: untracked (0), tracked (1)",
    )
    argparser.add_argument(
        "-tc",
        "--initial-tc",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_TC", "1"),
        dest="tc",
        help="Matrix C: untracked (0), tracked (1)",
    )
    argparser.add_argument(
        "-ap",
        "--initial-ap",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_AP", "0"),
        dest="ap",
        help="Params: global (0), shared (1)",
    )
    argparser.add_argument(
        "-aa",
        "--initial-aa",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_AA", "0"),
        dest="aa",
        help="Matrix A: global (0), shared (1), shared-bc (2), register (3)",
    )
    argparser.add_argument(
        "-ab",
        "--initial-ab",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_AB", "0"),
        dest="ab",
        help="Matrix B: global (0), shared (1), shared-bc (2), register (3)",
    )
    argparser.add_argument(
        "-ac",
        "--initial-ac",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_AC", "0"),
        dest="ac",
        help="Matrix C: register (0), shared (1), shared-bc (2)",
    )
    argparser.add_argument(
        "-bs",
        "--initial-bs",
        type=int,
        default=env_value("OPENCL_LIBSMM_SMM_BS", "0"),
        nargs="?",
        dest="bs",
        help="Minibatch size (0:auto)",
    )
    argparser.add_argument(
        "-mb",
        "--max-bs",
        type=int,
        default=0,
        nargs="?",
        dest="mb",
        help="Maximum (mini-)batch size (0:auto)",
    )
    argparser.add_argument(
        "-s",
        "--batchsize",
        type=int,
        default=0,
        nargs="?",
        dest="size",
        help="Size of batch (a.k.a. stacksize)",
    )
    args = argparser.parse_args()
    # OPENCL_LIBSMM_SMM_xx=tune|enabled|on must be given to permit tuning)
    if not os.getenv("OPENCL_LIBSMM_SMM_WS") in {"tune", "enabled", "on"}:
        os.environ["OPENCL_LIBSMM_SMM_WS"] = "{}".format(args.ws)
    # if not os.getenv("OPENCL_LIBSMM_SMM_AL") in {"tune", "enabled", "on"}:
    # os.environ["OPENCL_LIBSMM_SMM_AL"] = "{}".format(args.al)
    # fix tunables according to level of tuning
    if 1 <= args.tlevel or 0 > args.tlevel:
        os.environ["OPENCL_LIBSMM_SMM_BM"] = "{}".format(args.bm)
        os.environ["OPENCL_LIBSMM_SMM_BN"] = "{}".format(args.bn)
    if 2 <= args.tlevel or 0 > args.tlevel:
        os.environ["OPENCL_LIBSMM_SMM_TB"] = "{}".format(args.tb)
        os.environ["OPENCL_LIBSMM_SMM_TC"] = "{}".format(args.tc)
        os.environ["OPENCL_LIBSMM_SMM_AP"] = "{}".format(args.ap)
        os.environ["OPENCL_LIBSMM_SMM_AC"] = "{}".format(args.ac)
        os.environ["OPENCL_LIBSMM_SMM_NZ"] = "{}".format(args.nz)
    if 3 <= args.tlevel:
        os.environ["OPENCL_LIBSMM_SMM_BK"] = "{}".format(args.bk)
        os.environ["OPENCL_LIBSMM_SMM_WG"] = "{}".format(args.wg)
    if 4 <= args.tlevel:
        os.environ["OPENCL_LIBSMM_SMM_LU"] = "{}".format(args.lu)
    if 0 == args.mb:
        args.mb = 64
    # additional/depending arguments
    SmmTuner.main(args)
