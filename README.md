# DBCSR: Distributed Block Compressed Sparse Row matrix library 

[![Build Status](https://travis-ci.org/cp2k/dbcsr.svg?branch=develop)](https://travis-ci.org/cp2k/dbcsr) [![codecov](https://codecov.io/gh/cp2k/dbcsr/branch/develop/graph/badge.svg)](https://codecov.io/gh/cp2k/dbcsr)
[![Licence](https://img.shields.io/badge/license-GPL%20v2.0-blue.svg)](./LICENSE)
[![GitHub Releases](https://img.shields.io/github/release-pre/cp2k/dbcsr.svg)](https://github.com/cp2k/dbcsr/releases)

DBCSR is a library designed to efficiently perform sparse matrix matrix multiplication, among other operations.
It is MPI and OpenMP parallel and can exploit GPUs via CUDA.

<p align="center">
<img src="docs/logo/logo.png" width="500">
</p>

## Prerequisites

You absolutely need:

* GNU make
* a Fortran compiler which supports at least Fortran 2008 (including the TS 29113 when using the C-bindings)
* a LAPACK implementation (reference, OpenBLAS-bundled and MKL have been tested. Note: DBCSR linked to OpenBLAS 0.3.6 gives wrong results on Power9 architectures.)
* a BLAS implementation (reference, OpenBLAS-bundled and MKL have been tested)
* a Python version installed (2.7 or 3.6+ have been tested) with Numpy

Optionally:

* [libxsmm](https://github.com/hfp/libxsmm) (1.8.2+ with make-only, 1.10+ with cmake) for Small Matrix Multiplication acceleration
* [CMake](https://cmake.org/) (3.10+)

To build [libcusmm](src/acc/libsmm_acc/libcusmm) (DBCSR's CUDA backend), you further need:

* CUDA Toolkit
* a C++ compiler which supports at least C++11 standard

We test against GNU and Intel compilers on Linux systems, GNU compiler on MacOS systems.

## Getting started

Download either a release tarball or clone the latest version from Git using:

    git clone --recursive https://github.com/cp2k/dbcsr.git

Run

    make help

to list all possible targets.

Update the provided [Makefile.inc](Makefile.inc) to fit your needs 
(read the documentation inside the file for further explanations) and then run

    make <target>

Some examples on how to use the library (which is the only current documentation) are available under the examples directory (see [readme](examples/README.md)).

## C/C++ Interface

You can compile with

    make CINT=1

to generate the C interface. Make sure your Fortran compiler supports F2008
standard (including the TS) by updating the flag in the Makefile.inc.

## CMake

Building with CMake is also supported:

    mkdir build
    cd build
    cmake ..
    
The configuration flags are (default first):

    -DUSE_MPI=<ON|OFF>
    -DUSE_OPENMP=<ON|OFF>
    -DUSE_SMM=<blas|libxsmm>
    -DUSE_CUDA=<OFF|ON>
    -DUSE_CUBLAS=<OFF|ON>
    -DWITH_C_API=<ON|OFF>
    -DWITH_EXAMPLES=<ON|OFF>
    -DWITH_GPU=<P100|K20X|K40|K80|V100>
    -DTEST_MPI_RANKS=<auto,N>
    -DTEST_OMP_THREADS=<2,N>
    -DCMAKE_BUILD_TYPE=<Release|Debug|Coverage>

## Contributing to DBCSR

Your contribution to the project is welcome! 
Please see [Contributing.md](./CONTRIBUTING.md) and this [wiki page](https://github.com/cp2k/dbcsr/wiki/Development). For any help, please notify the other developers.
