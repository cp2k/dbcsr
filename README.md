# DBCSR: Distributed Block Compressed Sparse Row matrix library

DBCSR is a library designed to efficiently perform sparse matrix matrix multiplication, among other operations.
It is MPI and OpenMP parallel and can exploit GPUs via CUDA.

## Prerequisites

You absolutely need:

* a Fortran compiler which supports at least Fortran 2003
* a companion C compiler
* a LAPACK implementation (reference, OpenBLAS-bundled and MKL have been tested)
* a SCALAPACK implementation (reference or MKL have been tested)
* a BLAS implementation or [libxsmm](https://github.com/hfp/libxsmm)

To build with CUDA support you further need:

* CUDA Toolkit
* C++ STL implementation

## Getting started

Run

    make help

to list all possible targets.

If you want to change the compiler, you can either specify it directly:

    make CC=clang FC=flang

or update the provided `Makefile.inc` to fit your needs.
