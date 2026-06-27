# ACCelerator Interface

## Backends

The accelerator interface (ACC) consists of ISO_C_BINDING based Fortran code of DBCSR's [ACC-backend interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc.h) and [LIBSMM/ACC-interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_libsmm.h). The interface is implemented by CUDA (for Nvidia GPUs), HIP (for AMD GPUs), and the OpenCL accelerator backend.

The code for the CUDA and HIP backends is unified, and can be found in the `cuda`, `hip`, `cuda_hip`, and `libsmm_acc` directories. At compile-time either one or the other backend is chosen per macro (`__CUDA` or `__HIP`). The OpenCL backend is activated by the build-time macro `__OPENCL` and uses [LIBXSTREAM](https://libxstream.readthedocs.io/) for the OpenCL SMM implementation. [LIBXS](https://libxs.readthedocs.io/) is used underneath and also matters for DBCSR's host-side batched SMM path.

## Miniapp

There is one stand-alone sample code or driver exercising the ACC-interface for CUDA/HIP. The driver code only depends on the interfaces mentioned above and can be built locally without the DBCSR library, except for runtime libraries such as CUDA or HIP. For timing, comparison, and validation, the CUDA/HIP driver can use LIBXSMM if a prebuilt `libxsmm` folder is available next to DBCSR's root directory (`dbcsr`).

```bash
git clone -b main https://github.com/libxsmm/libxsmm.git
cd libxsmm
make GNU=1 -j
```

To build the CUDA/HIP driver code, change into the respective backend folder (`cuda` in the CUDA example below), and invoke `make` (`DBG=0|1|2` is supported among other optional key-value pairs).

```bash
git clone https://github.com/cp2k/dbcsr.git
cd dbcsr/src/acc/cuda
make
```

The OpenCL SMM sample code and benchmark driver that used to live below DBCSR's accelerator tree are now maintained by LIBXSTREAM. Build and tune that sample from LIBXSTREAM's `samples/smm` directory; see [LIBXSTREAM's documentation](https://libxstream.readthedocs.io/) for current commands and environment variables.

**NOTE**: To activate a certain device, the CUDA/HIP driver considers an environment variable called `DEVICE`. For example, `DEVICE=1 ./acc_bench` activates the second device (at least two devices must be discovered). The LIBXSTREAM SMM sample documents the corresponding OpenCL sample controls.

The driver supports command line options (_nrepeat_, _stack_size_, _m_, _n_, _k_, ...). Command line arguments are positional but allow `0` as placeholder to refer to the default value (`acc_bench 0 0 5 13 5` performs the default number of repetitions with the default stacksize when running the 5x13x5-kernel). For example, running the transpose benchmark may look like:

```bash
$ OMP_PROC_BIND=TRUE ./acc_bench 3 30000 23 23 23
Activated device0 (ndevices=8)
acc_bench 3 30000 23 23 23 1875 18750 18750
typename (id=3): double
copy-in (2058 MB): 92 ms 21.9 GB/s
transpose: 0.23 ms 3187.9 GFLOPS/s
device: 0.18 ms 4122.8 GFLOPS/s
host: 0.57 ms 1278.1 GFLOPS/s
diff.cur: 3.20547e-15 (|36.6983-36.6983|=5.47118e-13)
```

For timing, comparison (host code), and validation, LIBXSMM is used by the CUDA/HIP driver when available. For example with the CUDA backend:

```bash
cd src/acc/cuda
make WITH_GPU=P100
../acc_bench
```

In this case, `acc_bench` is built using the CUDA backend. The driver code can be built for double-precision (default) or single-precision using a build-time macro (`make ELEM_TYPE=float` or `-DELEM_TYPE=float` in general). For OpenCL benchmarking and tuning, use the LIBXSTREAM SMM sample.
