#!/bin/bash -l
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

#SBATCH --export=ALL
#SBATCH --constraint="gpu"
#SBATCH --partition="cscsci"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread

set -o errexit
set -o nounset
set -o pipefail

module load daint-gpu cudatoolkit cdt-cuda
module unload cray-libsci_acc
module list

export PATH=/project/cray/alazzaro/cmake/bin:$PATH

set -o xtrace  # do not set earlier to avoid noise from module

umask 0002  # make sure group members can access the data

mkdir -p "${SCRATCH}/${BUILD_TAG}.cray"
chmod 0775 "${SCRATCH}/${BUILD_TAG}.cray"
cd "${SCRATCH}/${BUILD_TAG}.cray"

export CRAY_CUDA_MPS=1 # enable the CUDA proxy for MPI+CUDA
export OMP_PROC_BIND=TRUE # set thread affinity
# OMP_NUM_THREADS is set by cmake

# document the current environment
env |& tee -a "${STAGE_NAME}.out"

ulimit -s 256000
env CTEST_OUTPUT_ON_FAILURE=1 make test ARGS="--timeout 1200" |& tee -a "${STAGE_NAME}.out"
