#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH --constraint="gpu"
#SBATCH --partition="cscsci"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-core=1 # 1=no HT, 2=HT

set -o errexit
set -o nounset
set -o pipefail

module load daint-gpu cudatoolkit CMake/3.18.4
module unload cray-libsci_acc
module list

set -o xtrace  # do not set earlier to avoid noise from module

umask 0002  # make sure group members can access the data

mkdir --mode=0775 -p "${SCRATCH}/${BUILD_TAG}.cray"
cd "${SCRATCH}/${BUILD_TAG}.cray"

export CRAY_CUDA_MPS=1 # enable the CUDA proxy for MPI+CUDA
export OMP_PROC_BIND=TRUE # set thread affinity
# OMP_NUM_THREADS is set by cmake

# document the current environment
env |& tee -a "${STAGE_NAME}.out"

ulimit -s 256000
env CTEST_OUTPUT_ON_FAILURE=1 make test ARGS="--timeout 900" |& tee -a "${STAGE_NAME}.out"
