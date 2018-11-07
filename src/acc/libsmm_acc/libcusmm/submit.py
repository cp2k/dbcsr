#!/usr/bin/env python
# -*- coding: utf-8 -*-
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

import sys, os
from os import path
from glob import glob

from subprocess import Popen, PIPE


#===============================================================================
def main():
	do_it = sys.argv[-1] == "doit!"

	cmd = ["squeue", "--user", os.environ['USER'], "--format=%j", "--nohead"]
	p = Popen(cmd, stdout=PIPE)
	submitted = p.stdout.read()

	n_submits = 0
	for d in glob("tune_*"):
		if not path.isdir(d):
			continue
		
		if len(glob(d+"/slurm-*.out"))>0:
			print("%20s: Found slurm file(s)"%d)
			continue

		if d in submitted:
			print("%20s: Found submitted job"%d)
			continue
	
		n_submits += 1
		if do_it:
			print("%20s: Submitting"%d)
			assert os.system("cd %s; sbatch *.job"%d)==0
		else:
			print('%20s: Would submit, run with "doit!"'%d)

	print("Number of jobs submitted: %d"%n_submits)

#===============================================================================
if len(sys.argv)==2 and sys.argv[-1]=="--selftest":
    pass #TODO implement selftest
else:
    main()
