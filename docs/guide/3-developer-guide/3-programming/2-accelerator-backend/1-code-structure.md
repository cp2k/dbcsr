title: Code Structure

# GPU Backend Code Architecture

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
```
