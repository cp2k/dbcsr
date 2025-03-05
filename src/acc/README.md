# ACCelerator Interface

## Backends

The accelerator interface (ACC) consists of ISO_C_BINDING based Fortran code of DBCSR's [ACC-backend interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc.h) and [LIBSMM/ACC-interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_libsmm.h). The interface is implemented by CUDA (for Nvidia GPUs), the HIP (for AMD GPUs), and the OpenCL accelerator backends.

The code for both the CUDA and the HIP backend is unified, and can be found in the `cuda` directory. At compile-time either one or the other backend is chosen per macro (`__CUDA` or `__HIP`). Similarly, the code for the OpenCL backend is activated by a build-time macro (`__OPENCL`).

## Miniapp

There is one stand-alone sample code or driver exercising the ACC-interface. The driver code (only depending on above mentioned interfaces) can be built locally and in a rather self-contained fashion, i.e., no DBCSR library is needed (except runtime libraries such as CUDA, HIP, OpenCL). For OpenCL, the LIBXSMM library is mandatory and preferred as baseline and for validation in any case. To build LIBXSMM, a folder `libxsmm` in parallel to DBCSR's root directory (`dbcsr`) is expected to be present and prebuilt.

```bash
git clone -b main https://github.com/libxsmm/libxsmm.git
cd libxsmm
make GNU=1 -j
```

To build the driver code (`opencl` in below example), change into the respective backend folder (`cuda` or `opencl`), and invoke `make` (`DBG=0|1|2` is supported among other optional key-value pairs).

```bash
git clone https://github.com/cp2k/dbcsr.git
cd dbcsr/src/acc/opencl
make
```

**NOTE**: To activate a certain device, the driver considers an environment variable called `DEVICE`. For example, `DEVICE=1 ./acc_bench` activates the second device (at least two devices must be discovered). This environment variable is implemented by the driver code and meant to work across backends, i.e., the OpenCL backend also supports `ACC_OPENCL_DEVICE=1` (see Developer Guide for the OpenCL backend).

The driver supports command line options (_nrepeat_, _stack_size_, _m_, _n_, _k_, ...). Command line arguments are positional but allow `0` as placeholder to refer to the default value (`acc_bench 0 0 5 13 5` performs the default number of repetitions with the default stacksize when running the 5x13x5-kernel). For example, running the tranpose benchmark may look like:

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

For timing, comparison (host code), and validation, LIBXSMM is required. The driver exercises the respective backend. For example with the CUDA backend:

```bash
cd src/acc/cuda
make WITH_GPU=P100
../acc_bench
```

For the OpenCL backend:

```bash
cd src/acc/opencl
make
../acc_bench
```

In above cases, `acc_bench` is built using the respective backend. The driver code can be built for double-precision (default) or single-precision using a build-time macro (`make ELEM_TYPE=float` or `-DELEM_TYPE=float` in general).
