title: Autotune

# Autotune

OpenCL SMM auto-tuning has moved out of DBCSR. The benchmark driver, `tune_multiply.py` workflow, tuned-parameter CSV files, and related OpenTuner setup are maintained with the SMM sample in [LIBXSTREAM](https://libxstream.readthedocs.io/).

DBCSR consumes this functionality through its OpenCL backend when built with `-DUSE_ACCEL=opencl` and linked with LIBXS and LIBXSTREAM. Use the LIBXSTREAM documentation for current tuning commands and parameter-management details.
