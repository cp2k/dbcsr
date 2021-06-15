# ACCelerator Interfaces

## Overview

This folder contains the ISO_C_BINDING based Fortran code of DBCSR's [ACC-backend interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc.h) and [LIBSMM/ACC-interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_libsmm.h). It also contains the CUDA (for Nvidia GPUs), the HIP (for AMD GPUs), and the OpenCL accelerator backends.

Further, two stand-alone sample codes are given exercising both interfaces (benchmarks).

## CUDA and HIP backends

The code for both the CUDA and HIP backends is unified, and can be found in the `cuda` directory.
At compile-time either one or the other backend is chosen per macro (`__CUDA` or `__HIP`).

## OpenCL backend

The code for both the OpenCL backends is enabled with a build-time macro (`__OPENCL`).

## Benchmarks

Two stand-alone drivers (only depending on above mentioned interfaces) can be built locally and in a rather self-contained fashion, i.e., no DBCSR library is needed (except runtime libraries such as CUDA, HIP, OpenCL/LIBXSMM). For OpenCL, a folder `libxsmm` parallel to DBCSR's root directory (`dbcsr`) is expected to be present and prebuilt (`make` in LIBXSMM's root directory is enough). To build the driver code, change into the respective backend folder (`cuda` or `opencl`), and invoke `make` (`DBG=0|1|2`, and a few other key-value pairs are optional). When building the code is completed, change back into the parent folder and invoke either `acc_bench_trans` or `acc_bench_smm`.

**NOTE**: To activate a certain device, an environment variable `DEVICE` can be used. For example, `DEVICE=1 ./acc_bench_trans` activates the second device (at least two devices must be discovered).

The drivers support a few command line options (_nrepeat_, _stack_size_, _m_, _n_, ...). Command line arguments are positional but allow `0` as placeholder to access the default value (`acc_bench_smm 0 0 5 13 5` performs the default number of repetitions with the default stacksize when running the 5x13x5-kernel). For example, running the tranpose benchmark may look like:

```bash
$ OMP_PROC_BIND=TRUE ./acc_bench_trans 5 30000 23 23
./acc_bench_trans 5 30000 23 23
typename (id=3): double
copy-in: 17.2 ms 7.2 GB/s
device: 8.7 ms 14.2 GB/s
host: 8.4 ms 14.6 GB/s
errors: 0
```

For timing, comparison (host code), and validation, LIBXSMM is required. The drivers exercise the respective backend. For example with the CUDA backend:

```bash
cd cuda
make DBG=0 WITH_GPU=P100
cd ..
```

For the OpenCL backend:

```bash
cd opencl
make DBG=0
cd ..
```

In either of the above cases, `acc_bench_trans` and `acc_bench_smm` are built using the respective backends.
Both driver codes can be instantiated for at least double- and single-precision using a build-time macro (`ELEM_TYPE`).
Several build-time settings can be made on the build-line (`-D`) or inside of the source files (`acc_bench_trans.c` or `acc_bench_smm.c`).
