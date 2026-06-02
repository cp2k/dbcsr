title: Code Structure

# GPU Backend Code Architecture

```
dbcsr/
-- src/
---- acc/: contains interfaces to ACC and LIBSMM (top-level) as well as backends (subdirectories)
------ cuda/: CUDA backend
------ hip/: HIP backend
------ cuda_hip/: common code for CUDA and HIP
------ libsmm_acc/: small matrix-matrix operations on GPU (can use either cuda or hip interface)
------ smm_*.c, smm_*.h: OpenCL SMM integration using LIBXSTREAM and LIBXS
```
