title: CMake Build Recipes

# DBCSR CMake Build Recipes

Following are recipes for different combinations of compilers, platforms and libraries.
Unless otherwise noted, the examples assume that after fetching/unpacking DBCSR you created
a directory `build/` inside the DBCSR directory and switched into it using `cd build/`.

The listed examples can usually be combined with other build options like *libxsmm* or *CUDA*
even if the examples are not explicitly given.

The instructions used for building in the Continuous Integration can be found in
the `.ci/` folder or in the `.github/workflows/`.

## GNU

### GNU compiler, system MPI and system-provided OpenBLAS

Most Linux systems provide the GNU compiler, a system MPI (OpenMPI or MPICH) using the
GNU compiler as a backend and OpenBLAS for BLAS/LAPACK:

```bash
    cmake ..
```

### GNU compiler, system MPI and Intel MKL

To use the Intel MKL together with the GNU compiler and possibly a system-MPI,
assuming that MKL is installed in `/sw/intel/mkl`.

Verified with MKL provided as part of the Intel Parallel Studio XE 2019.5 installed in `/sw/intel`
with an OS-provided GCC 7.4.1 on Linux openSUSE Leap 15.1, using CMake 3.12.0.

1. Make sure the MKL environment is properly loaded:

```bash
       source /sw/intel/mkl/bin/mklvars.sh intel64
```

2. Make sure CMake picks the Intel MKL over any system-provided BLAS library:

```bash
       cmake -DBLA_VENDOR=Intel10_64lp_seq ..
```

## Intel

Instructions for using Intel compiler or libraries for different parts on non-Cray systems.
For Cray systems, please check further below.

*Note*: in Intel Parallel Studio 2019 there is a potential issue that `mpirun` fails with
the error `OFI addrinfo() failed` on local (non-cluster) installations.
This can be worked around by setting `export I_MPI_FABRICS=shm`.

### Intel MPI, GNU Compiler and system-provided OpenBLAS

Verified with Intel Parallel Studio XE 2019.5 installed in `/sw/intel`
with an OS-provided GCC 7.4.1 on Linux openSUSE Leap 15.1, using CMake 3.12.0.

1. Make sure that the Intel environment is properly loaded:

```bash
       source /sw/intel/bin/compilervars.sh intel64
```

2. Use the Intel-provided MPI compiler wrappers for the GNU toolchain,
   to override CMake's auto-detection which may pick up the system MPI:

```bash
       CC=mpicc FC=mpifc CXX=mpicxx cmake ..
```

### Intel MPI, GNU Compiler and Intel MKL

Verified with Intel Parallel Studio XE 2019.5 installed in `/sw/intel`
with an OS-provided GCC 7.4.1 on Linux openSUSE Leap 15.1, using CMake 3.12.0.

1. Make sure that the Intel environment is properly loaded:

```bash
       source /sw/intel/bin/compilervars.sh intel64
```

2. Use the Intel-provided MPI compiler wrappers for the GNU toolchain:

```bash
       CC=mpicc FC=mpifc CXX=mpicxx cmake -DBLA_VENDOR=Intel10_64lp_seq ..
```

### Intel MPI, Intel Compiler and Intel MKL

Verified with Intel Parallel Studio XE 2019.5 installed in `/sw/intel`
on Linux openSUSE Leap 15.1, using CMake 3.12.0.

1. Make sure that the Intel environment is properly loaded:

```bash
       source /sw/intel/bin/compilervars.sh intel64
```

2. Use the Intel-provided MPI compiler wrappers:

```bash
       CC=mpiicc FC=mpiifort CXX=mpiicxx cmake -DBLA_VENDOR=Intel10_64lp_seq ..
```

## MacOS

Follow what is described in the previous sections.
For GNU, if you have installed Command Line Tools by Apple and GCC with Homebrew that can lead to a
conflict in which compiler CMake will use. Therefore, we suggest specifying GCC, for example

```bash
    CC=gcc-9 CXX=g++-9 cmake ..
```

where `-9` can be adapted to your version.

### PGI

Please note that you need at least PGI >= 19.11.

Assuming that your `$PATH` is set correctly such that `pgcc`, `pgc++` and `pgfortran` can be found,
run the following to get a DBCSR version without MPI:

```bash
    CC=pgcc CXX=pgc++ FC=pgfortran cmake -DUSE_MPI=OFF ..
```

the `-DUSE_MPI=OFF` is needed here to avoid that CMake picks up any MPI installation, for example from Homebrew.

To build with MPI you need an MPI implementation built for/with the PGI compiler, for example the MPICH
usually bundled with the PGI installation.

Make sure that `$PATH` is correctly set to include `mpicc` and `mpifort` from the PGI MPICH installation, then run:

```bash
    CC=mpicc CXX=mpicxx FC=mpifort MPICH_CC=pgcc cmake ..
```

## Cray

Some machines require additional environments to be loaded to either provide
the modules specified below or to be able to properly build with the loaded modules.

Please contact your cluster/datacenter administrator for more information.

Example for the CSCS' Piz Daint:

```bash
    module load daint-mc  # to build for the non-GPU partition
    module load daint-gpu  # to build for the GPU partition
```

*Note*: the `libsci-cray` has different variants for MPI or OpenMP.
When disabling either MPI or OpenMP support in DBCSR you might want to adjust the
selected BLAS/LAPACK library accordingly (e.g. drop the `_mpi`, or `_mp`).

### CCE and libsci-cray

Verified on CSCS' Piz Daint with CCE 10.0.2 and cray-libsci 20.06.1,
using CMake 3.18.4.

1. Make sure that the `PrgEnv-cray` module is loaded:

```bash
       module load PrgEnv-cray
```

2. While the MPI wrapper/compiler will be detected automatically,
   must the BLAS/LAPACK libraries be specified manually:

```bash
       cmake \
         -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment \
         -DBLAS_LIBRARIES="-lsci_cray_mpi_mp -lhugetlbfs" \
         -DLAPACK_LIBRARIES="-lsci_cray_mpi_mp" \
         ..
```

### Intel Compiler and libsci-cray

Verified on CSCS' Piz Daint with Intel 19.1 and cray-libsci 20.06.1,
using CMake 3.18.4.

1. Make sure that the `PrgEnv-intel` module is loaded:

```bash
       module load PrgEnv-intel
```

2. While the MPI wrapper/compiler will be detected automatically,
   must the BLAS/LAPACK libraries be specified manually:

```bash
       cmake \
         -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment \
         -DBLAS_LIBRARIES="-lsci_intel_mpi_mp -lhugetlbfs" \
         -DLAPACK_LIBRARIES="-lsci_intel_mpi_mp" \
         ..
```

### GNU Compiler and libsci-cray

Verified on CSCS' Piz Daint with GNU 8.3.0 and cray-libsci 20.06.1,
using CMake 3.18.4.

1. Make sure that the `PrgEnv-gnu` module is loaded:

```bash
       module load PrgEnv-gnu
```

2. While the MPI wrapper/compiler will be detected automatically,
   must the BLAS/LAPACK libraries be specified manually:

```bash
       cmake \
         -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment \
         -DBLAS_LIBRARIES="-lsci_gnu_mpi_mp -lhugetlbfs" \
         -DLAPACK_LIBRARIES="-lsci_gnu_mpi_mp" \
         ..
```

## Any compiler

### Custom compiler flags

In the DBCSR build system we preload the default compiler flags (especially the ones for Fortran) with flags
required to build the code with a specific compiler, while additional optimization flags are added based on the
CMake build type.

This allows the user to override optimization flags by setting a custom build type and providing optimization flags
for that build type as follows:

```bash
       cmake \
         -DCMAKE_BUILD_TYPE=custom \
         -DCMAKE_C_FLAGS_CUSTOM="-O3 -march=native" \
         -DCMAKE_CXX_FLAGS_CUSTOM="-O3 -march=native" \
         -DCMAKE_Fortran_FLAGS_CUSTOM="-O3 -march=native" \
         ..
```
