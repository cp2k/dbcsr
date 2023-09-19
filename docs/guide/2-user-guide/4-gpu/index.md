title: GPUs

# Introduction

[CP2K](https://github.com/cp2k/cp2k/) was initially enabled for GPUs by the means of the DBCSR library. The original development focused on scalability and an assumption of a `1:1`-relationship between CPUs and GPUs (one CPU-socket drives one GPU). Multi-GPU asks for associating CPU-ranks with the closest GPU (affinity), but is usually a desparture in terms of algorithms as well (GPU to GPU communication). DBCSR associates ranks with GPUs based on a round-robin scheme using the rank-ID, i.e., GPU-affinity is only achieved with the help of the underlying MPI implementation or support from other runtimes. Aggregating GPU acceleration in as little as possible systems is contrary to the original design of DBCSR (and CP2K at that time). CP2K is a versatile toolbox covering a variety of workloads (input language), which imposes several hotspots beyond DBCSR ([status](https://www.cp2k.org/gpu)).

CP2K or DBCSR can scale to thousands of nodes and furter benefit from thread-scalability once communication starts to dominate (due to higher total rank-counts). Thread-scalability (OpenMP) in DBCSR if not CP2K is not equally developed when compared to process scalability (MPI), i.e., higher rank-counts tend to yield better performance on smaller number of systems or nodes. With multiple ranks per GPU, context switches and other overhead can negatively impact performance. However, more ranks are needed to best drive the CPU-dominated portion of the code, and hence GPU and in particular multi-GPU acceleration poses a challenge.

CP2K almost exclusively uses double-precision calculations on CPUs and GPUs (along with DBCSR's need for atomic update instructions for GPUs). Consumer focused GPU offerings often deliver a FLOP-rate ratio between single and double precision up to `SP:DP = 64:1`, which renders them unsuitable for CP2K like not beneficial when compared to modestly many CPU cores. Further, GPU accleration hinges on memory bandwidth rather than compute which further limits the benefit.

# CUDA/HIP Backend

Users interested to tune kernels for the CUDA/HIP backend and LIBSMM_ACC, can take a look at the [Developer Guide](../../3-developer-guide/3-programming/2-accelerator-backend/2-libsmm_acc/3-tune.html). Following the guide, [tuned parameters](https://github.com/cp2k/dbcsr/tree/develop/src/acc/libsmm_acc/parameters) can be collected for the desired GPU and potentially submitted for the benefit of others.

# OpenCL Backend

This section shows how to auto-tune a kernel for the OpenCL based LIBSMM library. The process builds a stand-alone driver program which is then driven by an [OpenTuner](https://opentuner.org/) based script guiding the auto-tuning of the desired kernel. The [Developer Guide](../../3-developer-guide/3-programming/2-accelerator-backend/3-libsmm_ocl/1-autotune.html) provides more information, e.g., about constraining execution time or parallelizing the tuning-process as well as how to select and tune an entire set of kernels.

For simplicity, the GNU Compiler is used to build the afore mentioned driver program, both DBCSR and LIBXSMM are Git-cloned into the same common directory, e.g., the user's `HOME` directory, and the driver is built for tuning double-precision kernels (DP).

```bash
cd ${HOME}
git clone https://github.com/hfp/libxsmm.git
cd libxsmm
make GNU=1 -j

cd ${HOME}
git clone https://github.com/cp2k/dbcsr.git
cd dbcsr/src/acc/opencl
make
```

Tuning the 23x23x23-kernel for example is the default case. However, to better illustrate the options, M, N, and K are given explicitly. The `tune_multiply.py` script can be used interactively for example, and terminated with CTRL-C which writes a JSON-file with tuned parameters (note a file `.tune_multiply-double-32x32x32.json` is quietly written every time a better set of parameters is found), and then aggregates all JSON-files in the directory into a CSV-file (`tune_multiply.csv`).

```bash
cd ${HOME}/dbcsr/src/acc/opencl/smm
./tune_multiply.py 23x23x23
```

Beside of interactive termination, above process would also terminate based on OpenTuner's default or can be constrained by the number of steps (experiments), time to be spent, or a combination of both. Details can be found in the [Developer Guide](../../3-developer-guide/3-programming/2-accelerator-backend/2-libsmm_acc/3-tune.html).

Suppose the 23x23x23-kernel was tuned for some time (e.g., 5-10 minutes), tuned parameters can be incorporated into the backend. The aggregated parameters (`tune_multiply.csv`) are automatically embedded when rebuilding the library and driver.

```bash
cd ${HOME}/dbcsr/src/acc/opencl
make
```

Important kernels can be further tuned (in addition to spending more time for the process) by widening the set of tuned parameters (`--tuning-level` or `-a` with "0" denoting an unrestricted set of tunables).

```bash
cd ${HOME}/dbcsr/src/acc/opencl/smm
./tune_multiply.py 23x23x23 -a 0
```

To "continue" tuning beyond the default level, the previously found parameters must be embedded (by rebuilding the library and driver program) or can be alternatively specified at runtime (`OPENCL_LIBSMM_SMM_PARAMS=/path/to/tune_multiply.csv`).
