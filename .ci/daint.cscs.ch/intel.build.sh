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
#SBATCH --constraint="mc"
#SBATCH --partition="cscsci"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread

set -o errexit
set -o nounset
set -o pipefail

module swap PrgEnv-cray PrgEnv-intel
module load daint-gpu cudatoolkit cdt-cuda
module unload cray-libsci_acc
module list

export PATH=/project/cray/alazzaro/cmake/bin:$PATH

set -o xtrace  # do not set earlier to avoid noise from module

umask 0002  # make sure group members can access the data

mkdir -p "${SCRATCH}/${BUILD_TAG}.intel"
chmod 0775 "${SCRATCH}/${BUILD_TAG}.intel"
cd "${SCRATCH}/${BUILD_TAG}.intel"

cmake \
    -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment \
    -DUSE_ACCEL=cuda \
    -DWITH_GPU=P100 \
    -DBLAS_FOUND=ON -DBLAS_LIBRARIES="-lsci_intel_mpi_mp" \
    -DLAPACK_FOUND=ON -DLAPACK_LIBRARIES="-lsci_intel_mpi_mp" \
    -DMPIEXEC_EXECUTABLE="$(command -v srun)" \
    -DTEST_MPI_RANKS="${SLURM_NTASKS}" \
    -DTEST_OMP_THREADS="${SLURM_CPUS_PER_TASK}" \
    "${WORKSPACE}" |& tee -a "${STAGE_NAME}.out"

make VERBOSE=1 -j |& tee -a "${STAGE_NAME}.out"
