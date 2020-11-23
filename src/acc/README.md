# ACCerator Interfaces

## Overview

This folder contains the ISO_C_BINDING based Fortran code of DBCSR's [ACC-backend interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc.h) and [LIBSMM/ACC-interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_libsmm.h). Further, two stand-alone sample codes are given exercising both interfaces (benchmarks).

## Benchmarks

Two stand-alone drivers (only depending on above mentioned interfaces) can be built locally and in a rather self-contained fashion, i.e., no DBCSR library is needed (except for runtime libraries such as OpenCL, and LIBXSMM for some auxiliary functionality). For LIBXSMM, a folder `libxsmm` parallel to DBCSR's root directory (`dbcsr`) is expected to be present and prebuilt (`make` in LIBXSMM's root directory is enough). To build the driver code, change into the respective backend folder (`cuda` or `opencl`), and invoke `make` (`DBG=0|1|2`, and a few other key-value pairs are optional). When building the code is completed, change back into the parent folder and invoke either `acc_bench_trans` or `acc_bench_smm`.

The drivers support a few command line options (_nrepeat_, _stack_size_, _m_, _n_, ...); running the tranpose benchmark may look like:

```bash
$ OMP_PROC_BIND=TRUE ./acc_bench_trans 5 30000 23 23
./acc_bench_trans 5 30000 23 23
copy-in: 16.8 ms 7.4 GB/s
device: 8.7 ms 14.2 GB/s
host: 8.5 ms 14.5 GB/s
errors: 0
```

For timing, comparison (host code), and validation, LIBXSMM is expected. The drivers exercise the respective backend as chosen to build the code.
