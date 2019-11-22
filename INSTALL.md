# DBCSR CMake Build Recipes

Following are recipes for different combinations of compilers, platforms and libraries.
Unless otherwise noted, the examples assume that after fetching/unpacking DBCSR you created
a directory `build/` inside the DBCSR directory and switched into it using `cd build/`.

The listed examples can usually be combined with other build options like *libxsmm* or *CUDA*
even if the examples are not explicitly given.

The instructions used for building in the Continuous Integration can be found in
the `.ci/` folder or in the `.travis.yml`.


## GNU


### GNU compiler, system MPI and system-provided OpenBLAS

Most Linux systems provide the GNU compiler, a system MPI (OpenMPI or MPICH) using the
the GNU compiler as a backend and OpenBLAS for BLAS/LAPACK:

    cmake ..


### GNU compiler, system MPI and Intel MKL

To use the Intel MKL together with the GNU compiler and possibly a system-MPI,
assuming that MKL is installed in `/sw/intel/mkl`.

Verified with MKL provided as part of the Intel Parallel Studio XE 2019.5 installed in `/sw/intel`
with an OS-provided GCC 7.4.1 on Linux openSUSE Leap 15.1, using CMake 3.10.2.

1. Make sure the MKL environment is properly loaded:

       source /sw/intel/mkl/bin/mklvars.sh intel64

2. Make sure CMake picks the Intel MKL over any system-provided BLAS library:

       cmake -DBLA_VENDOR=Intel10_64lp_seq ..


## Intel

Instructions for using Intel compiler or libraries for different parts on non-Cray systems.
For Cray systems, please check further below.

*Note*: in Intel Parallel Studio 2019 there is a potential issue that `mpirun` fails with
the error `OFI addrinfo() failed` on local (non-cluster) installations.
This can be worked around by setting `export I_MPI_FABRICS=shm`.

### Intel MPI, GNU Compiler and system-provided OpenBLAS

Verified with Intel Parallel Studio XE 2019.5 installed in `/sw/intel`
with an OS-provided GCC 7.4.1 on Linux openSUSE Leap 15.1, using CMake 3.10.2.

1. Make sure that the Intel environment is properly loaded:

       source /sw/intel/bin/compilervars.sh intel64

2. Use the Intel-provided MPI compiler wrappers for the GNU toolchain,
   to override CMake's auto-detection which may pick up the system MPI:

       CC=mpicc FC=mpifc CXX=mpicxx cmake ..


### Intel MPI, GNU Compiler and Intel MKL

Verified with Intel Parallel Studio XE 2019.5 installed in `/sw/intel`
with an OS-provided GCC 7.4.1 on Linux openSUSE Leap 15.1, using CMake 3.10.2.

1. Make sure that the Intel environment is properly loaded:

       source /sw/intel/bin/compilervars.sh intel64

2. Use the Intel-provided MPI compiler wrappers for the GNU toolchain:

       CC=mpicc FC=mpifc CXX=mpicxx cmake -DBLA_VENDOR=Intel10_64lp_seq ..


### Intel MPI, Intel Compiler and Intel MKL

Verified with Intel Parallel Studio XE 2019.5 installed in `/sw/intel`
on Linux openSUSE Leap 15.1, using CMake 3.10.2.

1. Make sure that the Intel environment is properly loaded:

       source /sw/intel/bin/compilervars.sh intel64

2. Use the Intel-provided MPI compiler wrappers:

       CC=mpiicc FC=mpiifort CXX=mpiicxx cmake -DBLA_VENDOR=Intel10_64lp_seq ..

## MacOS

Follow what is descibed in the previous sections. 
For GNU, if you have installed Command Line Tools by Apple and GCC with homebrew that can lead to a 
conflict in which compiler cmake will use. Therefore, we suggest to specify GCC, for example

    CC=gcc-9 CXX=g++-9 cmake ..

where `-9` can be adapted to your version.

## Cray

Some machines require additional environments to be loaded to either provide
the modules specified below or to be able to properly build with the loaded modules.

Please contact your cluster/datacenter administrator for more information.

Example for the CSCS' Piz Daint:

    module load daint-mc  # to build for the non-GPU partition
    module load daint-gpu  # to build for the GPU partition

*Note*: the `libsci-cray` has different variants for MPI or OpenMP.
When disabling either MPI or OpenMP support in DBCSR you might want to adjust the
selected BLAS/LAPACK library accordingly (e.g. drop the `_mpi`, or `_mp`).


### CCE and libsci-cray

Verified on CSCS' Piz Daint with CCE 9.0.2 and cray-libsci 19.06.1,
using CMake 3.14.5.

1. Make sure that the `PrgEnv-cray` module is loaded:

       module load PrgEnv-cray

2. While the MPI wrapper/compiler will be detected automatically,
   must the BLAS/LAPACK libraries be specified manually:

       cmake \
         -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment \
         -DBLAS_LIBRARIES="-lsci_cray_mpi_mp -lhugetlbfs" \
         -DLAPACK_LIBRARIES="-lsci_cray_mpi_mp" \
         ..


### Intel Compiler and libsci-cray

Verified on CSCS' Piz Daint with Intel 19.01 and cray-libsci 19.06.1,
using CMake 3.14.5.

1. Make sure that the `PrgEnv-intel` module is loaded:

       module load PrgEnv-intel

2. While the MPI wrapper/compiler will be detected automatically,
   must the BLAS/LAPACK libraries be specified manually:

       cmake \
         -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment \
         -DBLAS_LIBRARIES="-lsci_intel_mpi_mp -lhugetlbfs" \
         -DLAPACK_LIBRARIES="-lsci_intel_mpi_mp" \
         ..


### GNU Compiler and libsci-cray

Verified on CSCS' Piz Daint with GNU 8.3.0 and cray-libsci 19.06.1,
using CMake 3.14.5.

1. Make sure that the `PrgEnv-gnu` module is loaded:

       module load PrgEnv-gnu

2. While the MPI wrapper/compiler will be detected automatically,
   must the BLAS/LAPACK libraries be specified manually:

       cmake \
         -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment \
         -DBLAS_LIBRARIES="-lsci_gnu_mpi_mp -lhugetlbfs" \
         -DLAPACK_LIBRARIES="-lsci_gnu_mpi_mp" \
         ..
