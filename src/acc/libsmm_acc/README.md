# GPU Accelerated Small Matrix Multiplications

`libsmm_acc` is a **lib**rary for **s**mall **m**atrix-**m**atrix multiplication on a GPU-**acc**elerator. Stacks of matrix-matrix multiplication indices are passed from DBCSR to `libsmm_acc` which performs the multiplications on the GPU.

For a description of the library (some details are outdated, but this nevertheless provides a very good introduction), see Chapter 8.4 of:

> WALKER, R. C., & GOETZ, A. W. (2016). [Electronic structure calculations on graphics processing units: from quantum chemistry to condensed matter physics](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118670712).

### Compilation

`libsmm_acc` is compiled from within DBCSR, there is no separate compilation.

## Directory Organization

- [`kernels/`](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/kernels/): GPU kernels (CUDA- and HIP-compatible) for matrix-matrix multiplication and Python interface to autotuning code.
- `generate_*.py`: utility scripts for `libsmm_acc` compilation
- `libsmm_acc*`: libsmm_acc C++ and CUDA / HIP code
- [`parameters/`](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/parameters/): contains `parameters_GPU.json` files. These are sets of matrix-matrix multiplication parameters for different (m, n, k)-triplets optimized for a given GPU card.
- [`tune/`](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/tune/): scripts for autotuning of optimal parameter sets, see [autotuning of kernel parameters](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/tune/README.md)

## Matrix-matrix Multiplication Kernels and Parameters

For a given matrix-matrix multiplication **triplet** characterized by dimensions

- **m**
- **n**
- **k**,

`libsmm_acc` can run 5 different matrix-matrix multiplication **kernels**:

- [tiny](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/kernels/smm_acc_dnt_tiny.h)
- [small](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/kernels/smm_acc_dnt_small.h)
- [medium](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/kernels/smm_acc_dnt_medium.h)
- [largeDB1](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/kernels/smm_acc_dnt_largeDB1.h) ("large double-buffering 1")
- [largeDB2](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/kernels/smm_acc_dnt_largeDB2.h) ("large double-buffering 2")

which take between 3 - 7 **parameters** (see figure at the top):

- **threads**: number of threads per block in the execution configuration of the CUDA/HIP kernels
- **grouping**: how many stack entries are grouped together into a CUDA/HIP thread block (if `grouping` is bigger, less blocks are launched)
- **minblocks**: specifies the desired minimum number of resident blocks per multiprocessor
- **tile_m**: (on the figure: **M**), `tile_m` * `tile_n` = dimensions of the result block `T`
- **tile_n** : (on the figure: **N**)
- **w**: input slab width (width of slab `P_A` and `P_B`)
- **v**: output slab width (width of slab `P_C`)

The performance of the matrix-matrix multiplication kernels is highly dependent on the choice of algorithm and parameters. For this reason, `libsmm_acc` provides lists of optimal parameters for different GPU cards and different (m, n, k)-triplets.

## Contributing to libsmm_acc

We expect users to contribute to the library by providing new optimized kernels and support for new GPUs.

#### Autotuning procedure

Follow the [autotuning procedure](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/tune/README.md)

#### Adding a new kernel

1. Choose a kernel `name`

2. Add the kernel's code (must be able to compile by both `nvcc` and `hip`) in file `kernels/smm_acc_dnt_name.h`

3. Add Python kernel class inheriting from base class `kernels/smm_acc_dnt_name.py`

#### Adding support for a new GPU card

1. Add the GPU's compute architecture properties to [`kernels/gpu_properties.json`](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/kernels/gpu_properties.json). For more information on where to find these properties, please refer to the "info" field of [`kernels/gpu_properties.json`](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/kernels/gpu_properties.json).

2. Add the GPU to the `gpu_architectures` data structure in [`kernels/smm_acc.py`](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/kernels/smm_acc.py).

3. Add the necessary code for setting `ARCH_NUMBER` correctly in the [`CMakeLists`](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/CMakeLists.txt). Also add this GPU to the list of `SUPPORTED_CUDA_ARCHITECTURES` or `SUPPORTED_HIP_ARCHITECTURES` in the [`CMakeLists`](https://github.com/cp2k/dbcsr/blob/develop/src/acc/libsmm_acc/CMakeLists.txt).

4. Add a minimal JSON file `parameters_GPU.json`, containing:

```json
{
}
```

then add matrix-matrix multiplication parameters for this GPU using *autotuning*.
