# Just-In-Time (JIT) Compilation in libsmm_acc

DBCSR's GPU backend, libsmm_acc, uses heavily templated cuda/hip kernels for its batched multiplication and transpose. These kernels are templated on the characteristic dimensions of the multiplication `m, n, k`, among other things. If DBCSR were to compile kernels for all possible `m, n, k`s ahead-of-time (AOT), this would bloat the library and would compilation time much longer. Instead, only the necessary kernels are JIT-ed on the fly, at runtime, as they are needed.

On NVIDIA's P100, the overhead of JIT has been found to be around 400ms for one kernel - a negligible overhead for typical DBCSR runs. On AMD GPUs however, the overhead has been found to be of several seconds, a real hinderance to performance.

For performance debugging and in order to check how much time a program spends doing JIT, look for the functions `jit_kernel_multiply` and `jit_kernel_transpose` in the timings report at the end of the output file.

