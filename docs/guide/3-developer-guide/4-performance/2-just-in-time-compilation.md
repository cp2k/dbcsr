title: JIT

# Just-In-Time (JIT) Compilation

DBCSR's GPU backends rely on templated kernels for batched matrix multiplication and matrix transpose (CUDA/HIP as well as OpenCL backend). If DBCSR were to compile kernels for all possible triplets or MxNxKs (in the case of transposes, for all possible MxNs) ahead-of-time (AOT), it would not only bloat the size of the library but also take a long time to compile the code. Reducing the number of triplets to a "practical set" would either sacrifice performance or limit potential acceleration to certain workloads.

Instead, kernels are generated Just-In-Time (JIT) or "on the fly", i.e., at runtime, as they are requested by the application and workload. For LIBSMM_ACC, the JIT infrastructure is based on [CUDA NVRTC](https://docs.nvidia.com/cuda/nvrtc/), a runtime compilation library for CUDA C++. The OpenCL based LIBSMM relies on the OpenCL runtime library to perform JIT compilation.

No matter which runtime is used and whether JIT compilation is in the order of ~500ms per kernel or not, the compilation time becomes significant during the process of auto-tuning a set of kernels. Therefore extra documentation is provided in either case (CUDA/HIP or OpenCL) on how to collect tuned parameters or to eventually submit a set of tuned parameters for the benefit of others.

To check how much time a program spends for JIT compilation, GPU-backends contribute `jit_kernel_multiply` and `jit_kernel_transpose` entries to the [timings report](1-insights.html). This report appears when the application terminates (final output). The OpenCL backend supports additional [Runtime Settings](../../3-developer-guide/3-programming/2-accelerator-backend/3-opencl-backend.html), e.g., to report compilation time on a per-kernel basis (`ACC_OPENCL_VERBOSE=2`).
