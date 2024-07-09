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
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread

#set -o nounset
set -o errexit
set -o pipefail

source /opt/intel/oneapi/mkl/latest/env/vars.sh
module swap PrgEnv-cray PrgEnv-gnu
module load daint-gpu cudatoolkit cdt-cuda
module unload cray-libsci_acc cray-libsci
module list

export PATH=/project/cray/alazzaro/cmake/bin:${PATH}

set -o xtrace  # do not set earlier to avoid noise from module

umask 0002  # make sure group members can access the data

mkdir -p "${SCRATCH}/${BUILD_TAG}.ocl"
chmod 0775 "${SCRATCH}/${BUILD_TAG}.ocl"
cd "${SCRATCH}/${BUILD_TAG}.ocl"

export LD_LIBRARY_PATH=${HOME}/libxsmm/lib:${LD_LIBRARY_PATH}
export OMP_PROC_BIND=TRUE # set thread affinity
# OMP_NUM_THREADS is set by cmake

# use default parameters (omit loading tuned parameters)
#export OPENCL_LIBSMM_SMM_PARAMS=0

# document the current environment
env |& tee -a "${STAGE_NAME}.out"

env CTEST_OUTPUT_ON_FAILURE=1 make test ARGS="--timeout 1200" |& tee -a "${STAGE_NAME}.out"
