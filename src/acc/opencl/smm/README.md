# LIBSMM (OpenCL)

## Overview

The LIBSMM library implements the [ACC LIBSMM interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_libsmm.h), and depends on the [OpenCL backend](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/README.md). At least the compile-time settings below are typically for development, e.g., when attempting to contribute new functionality or features, or meant for debug purpose (and not necessarily settings to be made when using DBCSR or CP2K).

## Customization

### Compile-time Settings

Compile-time settings are (implicitly) documented and can be adjusted by editing [opencl_libsmm.h](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/opencl_libsmm.h) (adjusting the build-line as per `-D` is possible as well but less convenient). For example, `OPENCL_LIBSMM_F32` is enabled by default but can be disabled, or `OPENCL_LIBSMM_DEBUG` (which is disabled by default) can be enabled for debug purpose.

### Runtime Settings

Runtime settings are made by the means of environment variables (implemented in `opencl_libsmm.c`). There are two categories (for the two major functions) like matrix transpose (`OPENCL_LIBSMM_TRANS_*`) and matrix multiplication (`OPENCL_LIBSMM_SMM_*`). For tranposing matrices:

* `OPENCL_LIBSMM_TRANS_BUILDOPTS`: character string with build options (compile and link) that are directly supplied to the OpenCL runtime compiler.
* `OPENCL_LIBSMM_TRANS_INPLACE`: Boolean value (zero or non-zero integer) for inplace matrix transpose not relying on local/shared memory (GPU).
* `OPENCL_LIBSMM_TRANS_BLOCK_M`: non-negative integer number (less/equal than the M-extent) denoting the blocksize in M-direction.

For multiplying matrices:

* `OPENCL_LIBSMM_SMM_BUILDOPTS`: character string with build options (compile and link) that are directly supplied to the OpenCL runtime compiler.
* `OPENCL_LIBSMM_SMM_ATOMICS`: selects the kind of atomic operation used for global memory updates ("cmpxchg", "xchg"), or disables atomic updates ("0"). The latter is to quantify the impact of atomic operations rather than for achieving correct results.
* `OPENCL_LIBSMM_SMM_BATCHSIZE`: non-negative integer number denoting the intr-kernel (mini-)batchsize mainly used to amortize atomic updates of data in global/main memory. The remainder with respect to the "stacksize" is handled by the kernel.
* `OPENCL_LIBSMM_SMM_BLOCK_M`: non-negative integer number (less/equal than the M-extent) denoting the blocksize in M-direction.
* `OPENCL_LIBSMM_SMM_BLOCK_N`: non-negative integer number (less/equal than the N-extent) denoting the blocksize in N-direction.

Please note: some of the above runtime settings may be non-smooth in the sense of enabling a distinct code-path depending on a specific value, e.g., `OPENCL_LIBSMM_SMM_BATCHSIZE=1`.

## Auto Tuning

TODO.
