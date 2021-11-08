title: Just-In-Time Compilation

# Just-In-Time (JIT) Compilation

DBCSR's GPU backends rely on templated kernels for batched matrix multiplication and matrix transpose. If DBCSR were to compile kernels for all possible `m, n, k`s (or, in the case of the transpose, for all possible `m, n`s) ahead-of-time (AOT), it would bloat the library and the compilation time would require to account for a large but still limited set of kernels.

Instead, kernels are JIT-ed on the fly, at runtime, as they are requested by the application and workload. For `libsmm_acc`, the JIT infrastructure is based on the CUDA library [NVRTC](https://docs.nvidia.com/cuda/nvrtc/), a runtime compilation library for CUDA C++. For OpenCL based LIBSMM, the JIT compilation relies on the OpenCL runtime library.

No matter which runtime is used and whether JIT compilation is in the order of ~500ms per kernel or longer, the compilation time becomes significant during the process of auto-tuning a set of kernels. Therefore extra documentation is provided in either case on how to collect tuned parameters and eventually submit them for reuse.

For performance debugging of the CUDA/HIP based GPU-support (and in order to check how much time a program spends doing JIT), look at `jit_kernel_multiply` and `jit_kernel_transpose` of the [timings report](1-insights.html) at the end of the output file. For the OpenCL based GPU-support, the [Runtime Settings](../../3-developer-guide/3-programming/2-accelerator-backend/3-opencl-backend.html) (`ACC_OPENCL_VERBOSE`) can be used to collect the time needed for JIT compilation.
