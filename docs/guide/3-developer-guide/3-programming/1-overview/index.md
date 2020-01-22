title: Overview

# Code Architecture

![DBCSR code architecture](./code-architecture.xxx)

```
dbcsr/
-- src/
---- acc/: contains all code related to accelerators
------ include/: contains interfaces to acc and acc_libsmm
------ cuda/: cuda interface
------ hip/: hip interface
------ openmp/ (PR #260): openmp offloading interface
------ libsmm_acc/: small matrix-matrix operations implementation on GPU (can use either cuda or hip interface)
------ libsmm_omp/ (PR #260): small matrix-matrix operations implementation on GPU (uses necessarily the openmp interface)
---- base/: base routines needed to abstract away some machine/compiler dependent functionality
---- block/: block level routines
---- core/: core matrix data structure
---- data/: data handling
---- dist/: data distribution and message passing
---- mm/: matrix matrix multiplication
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


