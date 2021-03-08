# OpenCL Backend

## Overview

The OpenCL backend implements the [ACC interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc.h), which is also exposed in Fortran and used throughout DBCSR's code base to drive (GPU-)accelation based on ACC's device enumeration, data movement, and synchronization functionality. The customizations below below are typically for development, e.g., when attempting to contribute new functionality or features, or meant for debug purpose (and not necessarily settings to be made when using DBCSR or CP2K).

## Customization

### Compile-time Settings

Compile-time settings are (implicitly) documented and can be adjusted by editing [acc_opencl.h](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/acc_opencl.h) (adjusting the build-line as per `-D` is possible as well but less convenient). For example, `ACC_OPENCL_STREAM_PRIORITIES` is enabled by default (and further confirmed at runtime/build-time) but can be disabled, or `ACC_OPENCL_DEBUG` (which is disabled by default) can be enabled for debug purpose. More sensitive/private compile-time settings may be available within particular translation units like in `acc_opencl_mem.c`.

An application of compile-time settings (and perhaps a valuable contribution) might be to call a GPU library in OpenCL-based LIBSMM. In such case, Shared Virtual Memory support (SVM) in OpenCL comes handy and can be enabled per `ACC_OPENCL_SVM`. The latter allows then to simply take the raw pointer out of an `cl_mem` object, and pass it into such library/function (which in turn can work across language borders, etc.).

### Runtime Settings

Runtime settings are made by the means of environment variables (implemented in `acc_opencl.c`). There are variables for chosing an OpenCL device:

* `ACC_OPENCL_DEVSPLIT`: integer enabling devices to be split into subdevices (non-zero: enabled, default/zero: disabled).
* `ACC_OPENCL_DEVTYPE`: character string matching the device-kind like "cpu", "gpu", or another kind if neither CPU or GPU.
* `ACC_OPENCL_DEVICE`: non-negative integer number to select a device from the (internally enumerated) list of devices.
* `ACC_OPENCL_VENDOR`: character string matching the vendor of the OpenCL device in an case-insensitive fashion, e.g., "intel".
* `ACC_OPENCL_VERBOSE`: verbosity level (integer) with console output on `stderr`.
    * `ACC_OPENCL_VERBOSE=1`: outputs the number of devices found and the name of the selected device.
    * `ACC_OPENCL_VERBOSE=2`: outputs the duration needed to generate a requested kernel.
    * `ACC_OPENCL_VERBOSE=3`: outputs device-side performance of kernels (geometric mean).
    * `ACC_OPENCL_VERBOSE=4`: outputs device-side performance of kernels (every execution).

The OpenCL backend enumerates and orders devices primarily by device-kind (GPU, CPU, and others in that order) and by memory capacity (secondary criterion). Device IDs are zero-based as per ACC interface (and less than what is permitted/returned by `acc_get_ndevices`).

Other runtime settings include:

* `ACC_OPENCL_ASYNC_MEMOPS`: Boolean value (zero or non-zero integer) for asynchronous data movements.
* `ACC_OPENCL_SVM`: Boolean value (zero or non-zero integer) for Shared Virtual Memory (SVM).

Please note: some of the above runtime settings depend on compile-time settings in the first place in order to be effective.
