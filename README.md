# DBCSR: Distributed Block Compressed Sparse Row matrix library 

[![Build Status](https://travis-ci.org/cp2k/dbcsr.svg?branch=develop)](https://travis-ci.org/cp2k/dbcsr) [![codecov](https://codecov.io/gh/cp2k/dbcsr/branch/develop/graph/badge.svg)](https://codecov.io/gh/cp2k/dbcsr)

DBCSR is a library designed to efficiently perform sparse matrix matrix multiplication, among other operations.
It is MPI and OpenMP parallel and can exploit GPUs via CUDA.

## Prerequisites

You absolutely need:

* GNU make
* a Fortran compiler which supports at least Fortran 2003 (respectively 2008+TS when using the C-bindings)
* a LAPACK implementation (reference, OpenBLAS-bundled and MKL have been tested)
* a BLAS implementation (reference, OpenBLAS-bundled and MKL have been tested)
* a Python version installed (2.x and 3.x have been tested)

Optionally, you can install [libxsmm](https://github.com/hfp/libxsmm).

To build with CUDA support you further need:

* CUDA Toolkit
* a C++ compiler which supports at least C++11 standard

We test against GNU and Intel compilers.

## Getting started

Download either a release tarball or clone the latest version from Git using:

    git clone --recursive https://github.com/cp2k/dbcsr.git

Run

    make help

to list all possible targets.

Update the provided `Makefile.inc` to fit your needs 
(read the documentation inside the file for further explanations) and then run

    make <target>

Support for `cmake` is still considered experimental and may not cover all functionalities yet.
If you are using `cmake` to build DBCSR, please make sure you can reproduce any errors using the plain `Makefile` before reporting them.

Some examples on how to use the library (which is the only current documentation) are available under the examples directory (see [readme](examples/README.md)).

## C/C++ Interface

You can compile with

    make CINT=1

to generate the C interface. Make sure your Fortran compiler supports F2008
standard (including the TS) by updating the flag in the Makefile.inc.
