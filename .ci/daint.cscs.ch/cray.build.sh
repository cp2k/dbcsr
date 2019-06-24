#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH --constraint="mc"
#SBATCH --partition="cscsci"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-core=1 # 1=no HT, 2=HT

set -o errexit
set -o nounset
set -o pipefail

module load daint-gpu cudatoolkit CMake/3.14.5
module unload cray-libsci_acc
# make sure a recent GCC is available as NVCC backend:
#   nvcc does not automatically use Cray's CC as backend
#   and the OS' default gcc-4.8 hangs when building libcusmm_benchmark
module load gcc
module list

set -o xtrace  # do not set earlier to avoid noise from module

umask 0002  # make sure group members can access the data

mkdir --mode=0775 -p "${SCRATCH}/${BUILD_TAG}.cray"
cd "${SCRATCH}/${BUILD_TAG}.cray"

cmake \
    -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment \
    -DUSE_CUDA=ON \
    -DUSE_CUBLAS=ON \
    -DWITH_GPU=P100 \
    -DBLAS_FOUND=ON -DBLAS_LIBRARIES="-lsci_cray_mpi_mp" \
    -DLAPACK_FOUND=ON -DLAPACK_LIBRARIES="-lsci_cray_mpi_mp" \
    -DMPIEXEC_EXECUTABLE="$(command -v srun)" \
    -DTEST_MPI_RANKS=${SLURM_NTASKS} \
    -DTEST_OMP_THREADS=${SLURM_CPUS_PER_TASK} \
    "${WORKSPACE}" |& tee -a "${STAGE_NAME}.out"

make VERBOSE=1 -j |& tee -a "${STAGE_NAME}.out"
