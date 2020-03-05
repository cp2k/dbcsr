title: Overview

# Code Architecture

![DBCSR code architecture](./dbcsr_mm_overview.png)

```
dbcsr/
-- src/
---- acc/: contains all code related to accelerators
---- base/: base routines needed to abstract away some machine/compiler dependent functionality
---- block/: block level routines
---- core/: core matrix data structure
---- data/: data handling
---- dist/: data distribution and message passing
---- mm/: matrix-matrix multiplication
---- mpi/: wrappers of the MPI routines
---- ops/: high level operations
---- tas/: tall-and-skinny matrices
---- tensors/: block-sparse tensor framework
---- utils/: utilities
---- work/
```

# Distribution Scheme

Assumed square matrix with 20x20 matrix with 5x5 blocks and a 2x2 processor grid

![DBCSR distribution over processors](./dbcsr_dist.png)

![DBCSR block scheme](./dbcsr_blocks.png)

# List of standard compiler flags

* OpenMP flag to enable multi-threaded parallelization, e.g. `-fopenmp` for GNU and Intel compilers.
* Warnings
* Error checkings

# List of Macros used in the code

| Macro | Explanation | Language |
|-|-|-|
| `__parallel` | Enable MPI runs | Fortran |
| `__MPI_VERSION=N` | DBCSR assumes that the MPI library implements MPI version 3. If you have an older version of MPI (e.g. MPI 2.0) available you must define `-D__MPI_VERSION=2` | Fortran |
| `__NO_MPI_THREAD_SUPPORT_CHECK` | Workaround for MPI libraries that do not declare they are thread safe (funneled) but you want to use them with OpenMP code anyways | Fortran |
| `__MKL` | Enable use of optimized Intel MKL functions | Fortran
| `__NO_STATM_ACCESS`, `__STATM_RESIDENT` or `__STATM_TOTAL` | Toggle memory usage reporting between resident memory and total memory. In particular, macOS users must use `-D__NO_STATM_ACCESS` | Fortran |
| `__NO_ABORT` | Avoid calling abort, but STOP instead (useful for coverage testing, and to avoid core dumps on some systems) | Fortran |
| `__LIBXSMM` | Enable [LIBXSMM](https://github.com/hfp/libxsmm/) link for optimized small matrix multiplications on CPU | Fortran |
| `__ACCELERATE` | Must be defined on macOS when Apple's Accelerate framework is used for BLAS and LAPACK (this is due to some interface incompatibilities between Accelerate and reference BLAS/LAPACK) | Fortran |
| `NDEBUG`       | Assertions are stripped ("compiled out"), `NDEBUG` is the ANSI-conforming symbol name (not `__NDEBUG`). Regular release builds may carry assertions for safety | Fortran, C, C++ |
| `__CRAY_PM_ACCEL_ENERGY` or `__CRAY_PM_ENERGY` | Switch on collectin energy profiling on Cray systems | Fortran |
| `__DBCSR_ACC` | Enable Accelerator compilation | Fortran, C, C++ |
| `__CUDA_PROFILING`  | To turn on Nvidia Tools Extensions. It requires to link `-lnvToolsExt` | Fortran, C, C++ |
| `__CUDA` | Enable CUDA acceleration | C, C++ |
| `__HIP`  | Enable HIP acceleration | C, C++ |




