title: GPUs

# Introduction

[CP2K](https://github.com/cp2k/cp2k/) was initially enabled for GPUs by the means of the DBCSR library. The original development focused on scalability and an assumption of a `1:1`-relationship between CPUs and GPUs (one CPU-socket drives one GPU). Multi-GPU asks for associating CPU-ranks with the closest GPU (affinity), but is usually a desparture in terms of algorithms as well (GPU to GPU communication). DBCSR associates ranks with GPUs based on a round-robin scheme using the rank-ID, i.e., GPU-affinity is only achieved with the help of the underlying MPI implementation or support from other runtimes. Aggregating GPU acceleration in as little as possible systems is contrary to the original design of DBCSR (and CP2K at that time). CP2K is a versatile toolbox covering a variety of workloads (input language), which imposes several hotspots beyond DBCSR ([status](https://www.cp2k.org/gpu)).

CP2K or DBCSR can scale to thousands of nodes and furter benefit from thread-scalability once communication starts to dominate (due to higher total rank-counts). Thread-scalability (OpenMP) in DBCSR if not CP2K is not equally developed when compared to process scalability (MPI), i.e., higher rank-counts tend to yield better performance on smaller number of systems or nodes. With multiple ranks per GPU, context switches and other overhead can negatively impact performance. However, more ranks are needed to best drive the CPU-dominated portion of the code, and hence GPU and in particular multi-GPU acceleration poses a challenge.

CP2K almost exclusively uses double-precision calculations on CPUs and GPUs (along with DBCSR's need for atomic update instructions for GPUs). Consumer focused GPU offerings often deliver a FLOP-rate ratio between single and double precision up to `SP:DP = 64:1`, which renders them unsuitable for CP2K like not beneficial when compared to modestly many CPU cores. Further, GPU accleration hinges on memory bandwidth rather than compute which further limits the benefit.

# CUDA/HIP Backend

Users interested to tune kernels for the CUDA/HIP backend and LIBSMM_ACC, can take a look at the [Developer Guide](../../3-developer-guide/3-programming/2-accelerator-backend/2-libsmm_acc/3-tune.html). Following the guide, [tuned parameters](https://github.com/cp2k/dbcsr/tree/develop/src/acc/libsmm_acc/parameters) can be collected for the desired GPU and potentially submitted for the benefit of others.

# OpenCL Backend

DBCSR's OpenCL backend is built on [LIBXSTREAM](https://libxstream.readthedocs.io/), with [LIBXS](https://libxs.readthedocs.io/) providing host-side batched small matrix multiplication support. The OpenCL SMM kernel sample, benchmark driver, tuned-parameter files, and OpenTuner workflow that used to live in DBCSR are now maintained as the SMM sample in LIBXSTREAM.

For DBCSR users, the immediate build knob is `-DUSE_ACCEL=opencl`. CMake then requires OpenCL headers/runtime support, LIBXS, and LIBXSTREAM. It can discover prebuilt installations through `pkg-config`, `LIBXSROOT`, and `LIBXSTREAMROOT`, or build local/downloaded sources as described in the installation guide.

Use [LIBXSTREAM's documentation](https://libxstream.readthedocs.io/) for OpenCL SMM benchmarking, tuning, and parameter management.
