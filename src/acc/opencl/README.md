# Backend

The OpenCL backend implements the [ACC interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc.h), which is exposed in Fortran and used throughout DBCSR's code base to drive (GPU-)acceleration based on ACC's device enumeration, data movement, and synchronization functionality. By design, DBCSR activates one device per rank (process). For instance, multiple GPUs can be used by the means of multiple ranks per system or at least one rank per device. The LIBSMM library complements the backend and implements the [ACC LIBSMM interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_libsmm.h).

All major GPU vendors support OpenCL even if the vendor-preferred programming model suggests otherwise. On Nvidia GPUs, the OpenCL backend can be used with CUDA based GPU-code in other portions of CP2K. The OpenCL based backend provides the following benefits:

* Code portability between GPU vendors (if not performance portability). For instance, performance of the OpenCL backend matches the performance of the CUDA backend or exceeds it.
* Acceptable performance for kernels not covered by specifically tuned parameters, and the ability to run on GPU if no tuned parameters are present.
* Auto-tuning kernels within an acceptable time limit along with handy scripts to retune parameters and to carry forward an existing set (new GPU).

Runtime settings are made by the means of environment variables. The OpenCL backend provides `acc_getenv.sh` to list all occurrences of `getenv` categorized into "OpenCL Backend environment variables" and "OpenCL LIBSMM environment variables". Common backend related settings are:

* `ACC_OPENCL_DEVSPLIT`: integer enabling devices to be split into subdevices (non-zero/default: subdevices, zero: aggregated).
* `ACC_OPENCL_DEVTYPE`: character string selecting "cpu", "gpu", "all" (unfiltered), or any other string (neither CPU or GPU).
* `ACC_OPENCL_DEVICE`: non-negative integer number to select a device from the (internally enumerated) list of devices.
* `ACC_OPENCL_VENDOR`: character string matching the vendor of the OpenCL device in a case-insensitive fashion, e.g., "intel".
* `ACC_OPENCL_VERBOSE`: verbosity level (integer) with console output on `stderr`.
    * `ACC_OPENCL_VERBOSE=1`: outputs the number of devices found and the name of the selected device.
    * `ACC_OPENCL_VERBOSE=2`: outputs the duration needed to generate a requested kernel.
    * `ACC_OPENCL_VERBOSE=3`: outputs device-side performance of kernels (every launch profiled).
* `ACC_OPENCL_DUMP`: dump preprocessed kernel source code (1) or dump compiled OpenCL kernels (2).
    * `ACC_OPENCL_DUMP=1`: dump preprocessed kernel source code and use it for JIT compilation. Instantiates the original source code using preprocessor definitions (`-D`) and collapses the code accordingly.
    * `ACC_OPENCL_DUMP=2`: dump compiled OpenCL kernels (depends on OpenCL implementation), e.g., PTX code on Nvidia.

The OpenCL backend enumerates and orders devices by kind, i.e., GPU, CPU, and "other" (primary criterion) and by memory capacity (secondary criterion). Device IDs are zero-based as defined by the ACC interface (and less than what is permitted by `acc_get_ndevices`).
