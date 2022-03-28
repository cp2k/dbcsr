# OpenCL Backend

## Overview

The OpenCL backend implements the [ACC interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc.h), which is exposed in Fortran and used throughout DBCSR's code base to drive (GPU-)acceleration based on ACC's device enumeration, data movement, and synchronization functionality. The customizations below are typically for development, e.g., when attempting to contribute new functionality or features, or meant for debug purpose (and not necessarily settings to be made when using DBCSR or CP2K).

## Customization

### Compile-time Settings

Compile-time settings are (implicitly) documented and can be adjusted by editing [acc_opencl.h](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/acc_opencl.h) (adjusting the build-line as per `-D` is possible as well but less convenient). For example, `ACC_OPENCL_STREAM_PRIORITIES` is enabled by default (and further confirmed at runtime/build-time) but can be disabled, or `ACC_OPENCL_DEBUG` (which is disabled by default) can be enabled for debug purpose.

### Runtime Settings

Runtime settings are made by the means of environment variables. The OpenCL backend provides `acc_getenv.sh` to list all occurrences of `getenv` categorized into "OpenCL Backend environment variables" and "OpenCL LIBSMM environment variables". Common backend related settings are:

* `ACC_OPENCL_DEVSPLIT`: integer enabling devices to be split into subdevices (non-zero/default: subdevices, zero: aggregated).
* `ACC_OPENCL_DEVTYPE`: character string selecting "cpu", "gpu", "all" (unfiltered), or any other string (neither CPU or GPU).
* `ACC_OPENCL_DEVICE`: non-negative integer number to select a device from the (internally enumerated) list of devices.
* `ACC_OPENCL_VENDOR`: character string matching the vendor of the OpenCL device in a case-insensitive fashion, e.g., "intel".
* `ACC_OPENCL_VERBOSE`: verbosity level (integer) with console output on `stderr`.
    * `ACC_OPENCL_VERBOSE=1`: outputs the number of devices found and the name of the selected device.
    * `ACC_OPENCL_VERBOSE=2`: outputs the duration needed to generate a requested kernel.
    * `ACC_OPENCL_VERBOSE=3`: outputs device-side measured performance of kernels (geometric mean).
    * `ACC_OPENCL_VERBOSE=4`: outputs device-side performance of kernels (every launch profiled).
* `ACC_OPENCL_DUMP`: dump preprocessed kernel source code (1) or dump compiled OpenCL kernels (2).
    * `ACC_OPENCL_DUMP=1`: dump preprocessed kernel source code and use it for JIT compilation. Instantiates the original source code using preprocessor definitions (`-D`) and collapses the code accordingly.
    * `ACC_OPENCL_DUMP=2`: dump compiled OpenCL kernels (depends on OpenCL implementation), e.g., PTX code on Nvidia.

The OpenCL backend enumerates and orders devices by device-kind, i.e., GPU, CPU, and "other" (primary criterion) and by memory capacity (secondary criterion). Device IDs are zero-based as defined by the ACC interface (and less than what is permitted/returned by `acc_get_ndevices`).
