# DBCSR: Distributed Block Compressed Sparse Row matrix library 

[![Build Status](https://travis-ci.org/cp2k/dbcsr.svg?branch=develop)](https://travis-ci.org/cp2k/dbcsr) [![codecov](https://codecov.io/gh/cp2k/dbcsr/branch/develop/graph/badge.svg)](https://codecov.io/gh/cp2k/dbcsr)

DBCSR is a library designed to efficiently perform sparse matrix matrix multiplication, among other operations.
It is MPI and OpenMP parallel and can exploit GPUs via CUDA.

## Prerequisites

You absolutely need:

* GNU make
* a Fortran compiler which supports at least Fortran 2003 (respectively 2008+TS when using the C-bindings)
* a companion C compiler which supports at least C99
* a LAPACK implementation (reference, OpenBLAS-bundled and MKL have been tested)
* a SCALAPACK implementation (reference or MKL have been tested)
* a BLAS implementation or [libxsmm](https://github.com/hfp/libxsmm)

To build with CUDA support you further need:

* CUDA Toolkit
* C++ STL implementation
* pybind11 library (header-only, available from: https://github.com/pybind/pybind11)

## Getting started

Download either a release tarball or clone the latest version from Git using:

    git clone --recursive https://github.com/cp2k/dbcsr.git

Run

    make help

to list all possible targets.

If you want to change the compiler, you can either specify it directly:

    make CC=clang FC=flang

or update the provided `Makefile.inc` to fit your needs.

Support for `cmake` is still considered experimental and may not cover all functionalities yet.
If you are using `cmake` to build DBCSR, please make sure you can reproduce any errors using the plain `Makefile` before reporting them.

Some examples on how to use the library (which is the only current documentation) are available under the examples directory (see [readme](examples/README.md)).

## C/C++ Interface

You can compile with

    make CINT=1

to generate the C interface. Make sure your Fortran compiler supports F2008
standard (including the TS) by updating the flag in the Makefile.inc.
