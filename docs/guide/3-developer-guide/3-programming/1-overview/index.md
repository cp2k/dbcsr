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
