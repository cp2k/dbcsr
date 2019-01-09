# libcusmm: GPU Accelerated Small Matrix Multiplications

`libcusmm` is a **lib**rary using **cu**da for **s**mall **m**atrix-**m**atrix multiplication on the GPU. Stacks of matrix-matrix multiplication indices are passed from DBCSR to `libcusmm` which performs the multiplications on the GPU. 

![libcusmm parameters](../../../../docs/images/libcusmm_parameters_and_memory.png)

For a description of the library (some details are outdated, but provides a very good introduction), see 
Chapter 8.4 of: 

> WALKER, R. C., & GOETZ, A. W. (2016). Electronic structure calculations on graphics processing units: from quantum chemistry to condensed matter physics. Available at https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118670712.

### Compilation

`libcusmm` is compiled from within DBCSR, there is no separate compilation.  

## Directory Organization 

- [kernels/](kernels/): CUDA kernels for matrix-matrix multiplication and python interface to autotuning and predictive code. 
- [notebooks/](notebooks/): notebooks for exploring data generated from autotuning and prediction. 
- `generate_*.py`: utility scripts for `libcusmm` compilation 
- `libcusmm_*`: libcusmm C++ and CUDA code 
- `parameters_*.json`: sets of parameters for different (m, n, k)-triplets optimized for a given GPU card 
- `predict_*.py`: scripts for prediction of optimal parameter sets, see [Autotuning of Kernel Parameters](#autotuning-of-kernel-parameters) 
- `tune_*.py`: scripts for autotuning of optimal parameter sets, see [Autotuning of Kernel Parameters](#autotuning-of-kernel-parameters)  

## Matrix-matrix Multiplication Kernels and Parameters  

For a given matrix-matrix multiplication **triplet** characterized by

- **m** 
- **n** 
- **k**

`libcusmm` can run 5 different matrix-matrix multiplication **kernels**: 

- [tiny](kernels/cusmm_dnt_tiny.h)
- [small](kernels/cusmm_dnt_small.h)
- [medium](kernels/cusmm_dnt_medium.h) 
- [largeDB1](kernels/cusmm_dnt_largeDB1.h) ("large double-buffering 1")
- [largeDB2](kernels/cusmm_dnt_largeDB2.h) ("large double-buffering 2")

which take between 3 - 7 **parameters** (see schema at the top): 

- **threads**: number of threads per block in the execution configuration of the CUDA kernels
- **grouping**: how many stack entries are grouped together into a CUDA thread block
- **minblocks**: specifies the desired minimum number of resident blocks per multiprocessor
- **tile_m**: tile_m * tile_n = result block dimensions 
- **tile_n** 
- **w**: input slab width (width of slab `P_A` and `P_B`)
- **v**: output slab width (width of slab `P_C`) 

### To add a new kernel 

1.

2.

### Kernel Parameters 
The performance is of matrix-matrix multiplication kernels is highly dependant on the choice of parameters. 
For this reason, `libcusmm` provides lists of optimal parameters for different GPU cards and different(m, n, k)-triplets 
found through autotuning.

#### Autotuning Procedure
how to do 
if GPU already there 
if not

#### Predictive Modelling of Kernel Parameters 

already existing data

or not

To add a new engineered feature  

requirements
packages
python version

explain derived parameters  
 
## How to add support for a new GPU card


