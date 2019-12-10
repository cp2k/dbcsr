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
from os import path
from glob import glob

from subprocess import Popen, PIPE


# ===============================================================================
def main(submit_jobs, num_jobs):

    cmd = ["squeue", "--user", os.environ["USER"], "--format=%j", "--nohead"]
    p = Popen(cmd, stdout=PIPE)
    submitted = p.stdout.read()
    submitted = submitted.decode("utf-8")

    n_submits = 0
    for d in glob("tune_*"):
        if not path.isdir(d):
            continue

        if len(glob(d + "/slurm-*.out")) > 0:
            print("%20s: Found slurm file(s)" % d)
            continue

        if d in submitted:
            print("%20s: Found submitted job" % d)
            continue

        n_submits += 1
        if submit_jobs:
            print("%20s: Submitting" % d)
            assert os.system("cd %s; sbatch *.job" % d) == 0
        else:
            if len(glob(d + "/*.job")) == 1:
                print('%20s: Would submit, run with "doit!"' % d)
            elif len(glob(d + "/*.job")) == 0:
                print(
                    '%20s: Cannot find jobfile, delete this folder and re-create with tune_setup.py"'
                    % d
                )
            else:
                print(
                    '%20s: Found multiple jobfiles, delete this folder and re-create with tune_setup.py"'
                    % d
                )

        if num_jobs > 0:
            if n_submits >= num_jobs:
                break

    print("Number of jobs submitted: %d" % n_submits)


# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Submit autotuning jobs: Each tune-directory contains a job file. Since there might be many tune-directories, the
        convenience script submit.py can be used. It will go through all the tune_*-directories and check if it has
        already been submitted or run. For this the script calls squeue in the background and it searches for
        slurm-*.out files.

        This script is part of the workflow for autotuning optimal libsmm_acc parameters.
        For more details, see README.md#autotuning-procedure.
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

    args = parser.parse_args()
    submit_jobs = True if args.doit == "doit!" else False
    main(submit_jobs, args.num_jobs)
