title: Install

# Install

## Prerequisites

You need:

* [CMake](https://cmake.org/) (3.17+)
* GNU make or Ninja
* Fortran compiler which supports at least Fortran 2008 (including the TS 29113 when using the C-bindings)
* BLAS+LAPACK implementation (reference, OpenBLAS and MKL have been tested. Note: DBCSR linked to OpenBLAS 0.3.6 gives wrong results on Power9 architectures.)
* Python version installed (2.7 or 3.6+ have been tested)

Optional:

* [LIBXSMM](https://github.com/hfp/libxsmm) (1.10+, and `pkg-config`) for Small Matrix Multiplication acceleration
* LAPACK implementation (reference, OpenBLAS-bundled and MKL have been tested), required when building the tests

To build DBCSR's GPU backend:

* CUDA Toolkit (targets NVIDIA GPUs, minimal version required: 5.5) with cuBLAS
    * Host C++ compiler which supports at least C++11 standard
* or HIP compiler (targets NVIDIA or AMD GPUs) and hipBLAS (ROCm 3.8 was tested)
    * Host C++ compiler which supports at least C++11 standard
* or OpenCL, i.e., development headers (`opencl-headers`), generic loader "ocl-icd" (`ocl-icd-opencl-dev`),
    * Vendor specific OpenCL package, e.g., [Intel Compute Runtime](https://github.com/intel/compute-runtime/releases/latest),
      or CUDA Toolkit (includes OpenCL)
    * For the OpenCL backend, a plain C compiler is sufficient (C90 standard),
    * Optionally `clinfo` (can be useful to show available devices)

DBCSR is tested against GNU and Intel compilers on Linux systems, and GNU compiler on MacOS systems.
See a list of supported compilers [here](./3-supported-compilers.html).

## Get DBCSR

Download either a [release tarball](https://github.com/cp2k/dbcsr/releases) or clone the latest version from Git using:

```bash
git clone --recursive https://github.com/cp2k/dbcsr.git
```

## Build

DBCSR can be compiled in four main variants:
* Serial, i.e., no OpenMP and no MPI
* OpenMP
* MPI
* OpenMP+MPI
In addition, the variants can support accelerators.

Run inside the `dbcsr` directory:

```bash
mkdir build
cd build
cmake ..
make
```

 The configuration flags for the CMake command are (default first):

```
-DUSE_MPI=<ON|OFF>
-DUSE_OPENMP=<ON|OFF>
-DUSE_SMM=<blas|libxsmm>
-DUSE_ACCEL=<opencl|cuda|hip>
-DWITH_CUDA_PROFILING=<OFF|ON>
-DWITH_C_API=<ON|OFF>
-DWITH_EXAMPLES=<ON|OFF>
-DWITH_GPU=<P100|K20X|K40|K80|V100|Mi50>
-DCMAKE_BUILD_TYPE=<Release|Debug|Coverage>
-DBUILD_TESTING=<ON|OFF>
-DTEST_MPI_RANKS=<auto,N>
-DTEST_OMP_THREADS=<2,N>
```

When providing a build of LIBXSMM, make sure the `lib` directory is added to the `PKG_CONFIG_PATH` variable prior
to running `cmake`. For example, if LIBXSMM was checked out using Git to your home folder:

```bash
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${HOME}/libxsmm/lib"
```

### CMake Build Recipes

For build recipes on different platforms, make sure to also read the [CMake Build Recipes](./1-cmake-build-recipes.html).

### Using Python in a virtual environment

If Python is desired from a virtual environment and the CMake version below v3.15, then the python interpreter shall be specified manually using `cmake -DPython_EXECUTABLE=/path/to/python`.

### C/C++ Interface

If MPI support is enabled (the default), the C API is automatically built.

### Workaround issue in HIP

For custom installs of HIP 3.9.0 and above, some paths have to be configured to ensure the JIT compiler can locate the HIP runtime and compiler tools

```bash
export ROCM_PATH=/path/to/hip-3.9.0
export HIP_PATH=$ROCM_PATH
export LLVM_PATH=/path/to/llvm-amdgpu-3.9.0
export HIP_DEVICE_LIB_PATH=/path/to/rocm-device-libs-3.9.0/amdgcn/bitcode
```

before running on an AMD GPU.
