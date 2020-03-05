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

| `__parallel` | enable MPI runs | Fortran |
| `__MPI_VERSION=N` | DBCSR assumes that the MPI library implements MPI version 3. If you have an older version of MPI (e.g. MPI 2.0) available you must define -D__MPI_VERSION=2` |


