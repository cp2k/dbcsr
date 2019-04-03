# libcusmm: GPU Accelerated Small Matrix Multiplications

`libcusmm` is a **lib**rary using **cu**da for **s**mall **m**atrix-**m**atrix multiplication on the GPU. Stacks of matrix-matrix multiplication indices are passed from DBCSR to `libcusmm` which performs the multiplications on the GPU.

![libcusmm parameters](../../../../docs/images/libcusmm_parameters_and_memory.png)

For a description of the library (some details are outdated, but this nevertheless provides a very good introduction), see Chapter 8.4 of:

> WALKER, R. C., & GOETZ, A. W. (2016). Electronic structure calculations on graphics processing units: from quantum chemistry to condensed matter physics.
> 
> Available at https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118670712.

### Compilation

`libcusmm` is compiled from within DBCSR, there is no separate compilation.

## Directory Organization

- [`kernels/`](kernels/): CUDA kernels for matrix-matrix multiplication and python interface to autotuning and predictive code.
- [`notebooks/`](notebooks/): jupyter notebooks for exploring data generated from autotuning and prediction.
- `generate_*.py`: utility scripts for `libcusmm` compilation
- `libcusmm_*`: libcusmm C++ and CUDA code
- `parameters_*.json`: sets of matrix-matrix multiplication parameters for different (m, n, k)-triplets optimized for a given GPU card. You can explore these parameters interactively using the [provided jupyter notebook](notebooks/inspect_autotuned_parameters.ipynb)
- `predict_*.py`: scripts for prediction of optimal parameter sets, see [predictive modelling of kernel parameters](#predictive-modelling-of-kernel-parameters)
- `tune_*.py`: scripts for autotuning of optimal parameter sets, see [autotuning of kernel parameters](#autotuning-procedure)

## Matrix-matrix Multiplication Kernels and Parameters

For a given matrix-matrix multiplication **triplet** characterized by dimensions

- **m**
- **n**
- **k**,

`libcusmm` can run 5 different matrix-matrix multiplication **kernels**:

- [tiny](kernels/cusmm_dnt_tiny.h)
- [small](kernels/cusmm_dnt_small.h)
- [medium](kernels/cusmm_dnt_medium.h)
- [largeDB1](kernels/cusmm_dnt_largeDB1.h) ("large double-buffering 1")
- [largeDB2](kernels/cusmm_dnt_largeDB2.h) ("large double-buffering 2")

which take between 3 - 7 **parameters** (see figure at the top):

- **threads**: number of threads per block in the execution configuration of the CUDA kernels
- **grouping**: how many stack entries are grouped together into a CUDA thread block (if `grouping` is bigger, less blocks are launched)
- **minblocks**: specifies the desired minimum number of resident blocks per multiprocessor
- **tile_m**: (on the figure: **M**), tile_m * tile_n = dimensions of the result block `T`
- **tile_n** : (on the figure: **N**)
- **w**: input slab width (width of slab `P_A` and `P_B`)
- **v**: output slab width (width of slab `P_C`)

The performance of the matrix-matrix multiplication kernels is highly dependant on the choice of algorithm and  parameters. For this reason, `libcusmm` provides lists of optimal parameters for different GPU cards and different (m, n, k)-triplets. These sets of optimal parameters can be found either through *autotuning* or *predictive modelling*.

## Contributing to libcusmm

#### Autotuning procedure

Follow the [autotuning procedure](https://www.cp2k.org/howto:libcusmm)

#### Predictive modelling of kernel parameters

Follow the [predictive modelling procedure](predict.md)

#### Adding a new kernel

1. Choose a kernel `name`

2. Add the kernel's CUDA code in file `kernels/cusmm_dnt_name.h`

3. Add python kernel class inheriting from base class `kernels/cusmm_dnt_name.py`

4. Add the new kernel to the `kernel_algorithm` data structure in [`kernels/cusmm_predict.py`](kernels/cusmm_predict.py)

#### Adding support for a new GPU card

1. Add the GPU's compute architecture properties to [`kernels/gpu_properties.json`](kernels/gpu_properties.json). For more information on where to find these properties, please refer to the "info" field of [`kernels/gpu_properties.json`](kernels/gpu_properties.json).

2. Add the GPU to the `arch_number` data structure in [`kernels/cusmm_predict.py`](kernels/cusmm_predict.py)

4. Add the necessary code for setting `ARCH_NUMBER` correctly in the [`Makefile`](../../../../Makefile) and in the [`CMakeListst`](CMakeLists.txt)

5. Add a minimal JSON file `parameters_GPU.json`, containing:

```json
{
}
```

then add matrix-matrix multiplication parameters for this GPU using *autotuning* and *predictive modelling*
