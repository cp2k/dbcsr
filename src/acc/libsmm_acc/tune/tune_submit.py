#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

import os
import argparse

from subprocess import check_call, check_output
from pathlib import Path


def tune_sort_key(path: Path):
    try:
        _, triple = path.name.split("_")
        m, n, k = triple.split("x")
    except ValueError:
        return (0, 0, 0)  # sort non-matching dirs as they come at the beginning

    return (int(m), int(n), int(k))


# ===============================================================================
def main(submit_jobs, num_jobs, tune_dir: Path, sbatch_args):
    cmd = ["squeue", "--user", os.environ["USER"], "--format=%j", "--nohead"]
    submitted = check_output(cmd, encoding="utf-8")

    n_submits = 0
    for dir in sorted(tune_dir.glob("tune_*"), key=tune_sort_key):
        if not dir.is_dir():
            continue

        if list(dir.glob("slurm-*.out")):
            print(f"{dir.name:20}: Found slurm file(s)")
            continue

        if dir.name in submitted:
            print(f"{dir.name:20}: Found submitted job")
            continue

        n_submits += 1
        if submit_jobs:
            print(f"{dir.name:20}: Submitting")
            check_call(f"cd {dir}; sbatch {sbatch_args} *.job", shell=True)
        else:
            jobfiles = list(dir.glob("*.job"))
            if len(jobfiles) == 1:
                print(f'{dir.name:20}: Would submit, run with "doit!"')
            elif len(jobfiles) == 0:
                print(
                    f"{dir.name:20}: Cannot find jobfile, delete this folder and re-create with tune_setup.py"
                )
            else:
                print(
                    f"{dir.name:20}: Found multiple jobfiles, delete this folder and re-create with tune_setup.py"
                )

        if num_jobs > 0:
            if n_submits >= num_jobs:
                break

    print(f"Number of jobs submitted: {n_submits}")


# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Submit autotuning jobs: Each tune-directory contains a job file. Since there might be many tune-directories, the
        convenience script tune_submit.py can be used. It will go through all the tune_*-directories and check if it has
        already been submitted or run. For this the script calls squeue in the background and it searches for
        slurm-*.out files.

        This script is part of the workflow for autotuning optimal libsmm_acc parameters.
        For more details, see README.md.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("doit", metavar="doit!", nargs="?", type=str)
    parser.add_argument(
        "-j",
        "--num_jobs",
        metavar="INT",
        default=0,
        type=int,
        help="Maximum number of jobs to submit. 0: submit all",
    )
    parser.add_argument(
        "-d",
        "--dir",
        metavar="tune_directory",
        default=".",
        type=Path,
        help="Path from which to read already-existing tune-folders and write new tune-folders",
    )
    parser.add_argument(
        "--sbatch-args",
        metavar="sbatch_args",
        default="",
        type=str,
        help="Additional arguments passed to sbatch",
    )

    args = parser.parse_args()
    submit_jobs = True if args.doit == "doit!" else False
    main(submit_jobs, args.num_jobs, args.dir, args.sbatch_args)
