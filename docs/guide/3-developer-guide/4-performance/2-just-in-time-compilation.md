title: Just-In-Time Compilation

# Just-In-Time (JIT) Compilation in libsmm_acc

DBCSR's GPU backend, libsmm_acc, uses heavily templated cuda/hip kernels for its batched multiplication and transpose.

If DBCSR were to compile kernels for all possible `m, n, k`s (or, in the case of the transpose, for all possible `m, n`s) ahead-of-time (AOT), this would bloat the library and the compilation time would be much longer.
Instead, kernels are JIT-ed on the fly, at runtime, as they are requested by the user. `libsmm_acc`'s JIT infrastructure is based on the CUDA library [NVRTC](https://docs.nvidia.com/cuda/nvrtc/), a runtime compilation library for CUDA C++.

On NVIDIA's P100, the overhead of JIT has been found to be around 400ms for one kernel - a negligible overhead for typical DBCSR (and CP2K) runs.
On AMD GPUs however, the overhead has been found to be of several seconds, a real hinderance to performance.

For performance debugging and in order to check how much time a program spends doing JIT, look for the functions `jit_kernel_multiply` and `jit_kernel_transpose` in the [timings report](./1-insights.md) at the end of the output file.

