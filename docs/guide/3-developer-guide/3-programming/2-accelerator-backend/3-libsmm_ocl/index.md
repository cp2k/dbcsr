title: OpenCL / LIBXSTREAM

# OpenCL / LIBXSTREAM

DBCSR's OpenCL accelerator path no longer carries a separate `src/acc/opencl/smm` implementation in the documentation tree. The OpenCL SMM code, sample benchmark, tuned-parameter files, and OpenTuner workflow are maintained in [LIBXSTREAM](https://libxstream.readthedocs.io/). LIBXSTREAM in turn uses [LIBXS](https://libxs.readthedocs.io/), which also provides host-side batched small matrix multiplication support relevant to DBCSR's CPU path.

Inside DBCSR, `-DUSE_ACCEL=opencl` enables the OpenCL accelerator backend. The build requires OpenCL headers/runtime support, LIBXS, and LIBXSTREAM; CMake can use prebuilt installations through `pkg-config`, `LIBXSROOT`, and `LIBXSTREAMROOT`, or obtain sources through the configured dependency mechanism.

For kernel-level experimentation, benchmarking, tuned-parameter management, and auto-tuning, use the LIBXSTREAM SMM sample documentation rather than the historical DBCSR OpenCL tuning notes.
