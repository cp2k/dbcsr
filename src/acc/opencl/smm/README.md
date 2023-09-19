# LIBSMM

The LIBSMM library implements the [ACC LIBSMM interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_libsmm.h), and depends on the [OpenCL backend](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/README.md).

Compile-time settings are (implicitly) documented and can be adjusted by editing [opencl_libsmm.h](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/opencl_libsmm.h), e.g., `OPENCL_LIBSMM_VALIDATE` is disabled by default but can be enabled for debug purpose. The `OPENCL_LIBSMM_VALIDATE` compile-time setting enables side-by-side validation of matrix transpose and multiply operations between device and host. For example, running DBCSR's unit tests with `OPENCL_LIBSMM_VALIDATE` enabled produces console output that allows to pin-point a kernel which misses validation. Runtime settings are made by the means of environment variables. The OpenCL backend provides `acc_getenv.sh` to list all occurrences of `getenv` categorized into "OpenCL Backend environment variables" and "OpenCL LIBSMM environment variables".

There are two categories for the two domains in LIBSMM, i.e., matrix transpose (`OPENCL_LIBSMM_TRANS_*`) and matrix multiplication (`OPENCL_LIBSMM_SMM_*`). For transposing matrices, the settings are:

* `OPENCL_LIBSMM_TRANS_BUILDOPTS`: character string with build options (compile and link) supplied to the OpenCL runtime compiler.
* `OPENCL_LIBSMM_TRANS_INPLACE`: Boolean value (zero or non-zero integer) for in-place matrix transpose (no local memory needed).
* `OPENCL_LIBSMM_TRANS_BM`: non-negative integer number (less/equal than the M-extent) denoting the blocksize in M-direction.

The most common settings for multiplying matrices are:

* `OPENCL_LIBSMM_SMM_BUILDOPTS`: character string with build options (compile and link) supplied to the OpenCL runtime compiler.
* `OPENCL_LIBSMM_SMM_ATOMICS`: selects the kind of atomic operation used for global memory updates (`xchg`, `cmpxchg`, `cmpxchg2`), attempts to force atomic instructions, or disables atomic instructions (`0`). The latter is for instance to quantify the impact of atomic operations.
* `OPENCL_LIBSMM_SMM_PARAMS`: Disable embedded/auto-tuned parameters (`0`), or load CSV-file (e.g., `path/to/tune_multiply.csv`).
* `OPENCL_LIBSMM_SMM_BS`: non-negative integer number denoting the intra-kernel (mini-)batchsize mainly used to amortize atomic updates of data in global/main memory. The remainder with respect to the "stacksize" is handled by the kernel.
* `OPENCL_LIBSMM_SMM_BM`: non-negative integer number (less/equal than the M-extent) denoting the blocksize in M-direction.
* `OPENCL_LIBSMM_SMM_BN`: non-negative integer number (less/equal than the N-extent) denoting the blocksize in N-direction.
* `OPENCL_LIBSMM_SMM_AP`: specifies access to array of parameters (batch or "stack").
* `OPENCL_LIBSMM_SMM_AA`: specifies access to array of A-matrices.
* `OPENCL_LIBSMM_SMM_AB`: specifies access to array of B-matrices.
* `OPENCL_LIBSMM_SMM_AC`: specifies access to array of C-matrices.

The full list of tunable parameters and some explanation can be received with `smm/tune_multiply.py --help`, i.e., short description, default settings, and accepted values.

**NOTE**: LIBSMM's tunable runtime settings can be non-smooth like producing distinct code-paths, e.g., `OPENCL_LIBSMM_SMM_BS=1` vs. `OPENCL_LIBSMM_SMM_BS=2`.

# Auto Tuning

To tune and optimize a kernel and generating kernel parameters, please refer to the [Auto Tuning](README-autotune.md) guide. To update or retune an entire set of kernels (optimized parameters), please refer to the [Bulk Tuning](README-bulktune.md) guide.
