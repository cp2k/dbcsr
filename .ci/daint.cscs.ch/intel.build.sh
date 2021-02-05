#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH --constraint="mc"
#SBATCH --partition="cscsci"
#SBATCH --nodes=1

set -o errexit
set -o nounset
set -o pipefail

module swap PrgEnv-cray PrgEnv-intel
module load daint-gpu cudatoolkit CMake/3.18.4
module unload cray-libsci_acc
# make sure a recent GCC is available as NVCC backend:
#   nvcc does not automatically use Cray's CC as backend
#   and the OS' default gcc-4.8 hangs when building libcusmm_benchmark
module load gcc
module list

set -o xtrace  # do not set earlier to avoid noise from module

umask 0002  # make sure group members can access the data

mkdir --mode=0775 -p "${SCRATCH}/${BUILD_TAG}.intel"
cd "${SCRATCH}/${BUILD_TAG}.intel"

cmake \
    -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment \
    -DUSE_ACCEL=cuda \
    -DWITH_GPU=P100 \
    -DBLAS_FOUND=ON -DBLAS_LIBRARIES="-lsci_intel_mpi_mp" \
    -DLAPACK_FOUND=ON -DLAPACK_LIBRARIES="-lsci_intel_mpi_mp" \
    -DMPIEXEC_EXECUTABLE="$(command -v srun)" \
    -DTEST_MPI_RANKS=${SLURM_NTASKS} \
    -DTEST_OMP_THREADS=${SLURM_CPUS_PER_TASK} \
    "${WORKSPACE}" |& tee -a "${STAGE_NAME}.out"

make VERBOSE=1 -j |& tee -a "${STAGE_NAME}.out"
