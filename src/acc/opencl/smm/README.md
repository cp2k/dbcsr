# LIBSMM (OpenCL)

## Overview

The LIBSMM library implements the [ACC LIBSMM interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_libsmm.h), and depends on the [OpenCL backend](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/README.md). At least the compile-time settings below are typically for development, e.g., when attempting to contribute new functionality or features, or meant for debug purpose (and not necessarily settings to be made when using DBCSR or CP2K).

## Customization

### Compile-time Settings

Compile-time settings are (implicitly) documented and can be adjusted by editing [opencl_libsmm.h](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/opencl_libsmm.h) (adjusting the build-line as per `-D` is possible as well but less convenient). For example, `OPENCL_LIBSMM_F32` is enabled by default but can be disabled, or `OPENCL_LIBSMM_DEBUG` (which is disabled by default) can be enabled for debug purpose.

The `OPENCL_LIBSMM_DEBUG` compile-time setting enables side-by-side validation of matrix transpose and multiply operations on GPU against a built-in CPU implementation. For example, running DBCSR's unit tests with this setting produces useful console output that allows to pin-point the exact call arguments causing a validation error.

### Runtime Settings

Runtime settings are made by the means of environment variables (implemented in `acc_opencl.c`). There are two categories (for the two major functions) like matrix transpose (`OPENCL_LIBSMM_TRANS_*`) and matrix multiplication (`OPENCL_LIBSMM_SMM_*`). Common settings are (see OpenCL backend documentation for more details):

* `ACC_OPENCL_DEVSPLIT`: integer enabling devices to be split into subdevices (non-zero/default: enabled, zero: disabled).
* `ACC_OPENCL_DEVTYPE`: character string selecting "cpu", "gpu", "all" (unfiltered), or any other string (neither CPU or GPU).
* `ACC_OPENCL_DEVICE`: non-negative integer number to select a device from the (internally enumerated) list of devices.
* `ACC_OPENCL_VENDOR`: character string matching the vendor of the OpenCL device in an case-insensitive fashion, e.g., "intel".
* `ACC_OPENCL_VERBOSE`: verbosity level (integer) with console output on `stderr`.
    * `ACC_OPENCL_VERBOSE=1`: outputs the number of devices found and the name of the selected device.
    * `ACC_OPENCL_VERBOSE=2`: outputs the duration needed to generate a requested kernel.
    * `ACC_OPENCL_VERBOSE=3`: outputs device-side measured performance of kernels (geometric mean).
    * `ACC_OPENCL_VERBOSE=4`: outputs device-side performance of kernels (every launch profiled).

For transposing matrices (implemented in `opencl_libsmm.c`):

* `OPENCL_LIBSMM_TRANS_BUILDOPTS`: character string with build options (compile and link) supplied to the OpenCL runtime compiler.
* `OPENCL_LIBSMM_TRANS_INPLACE`: Boolean value (zero or non-zero integer) for in-place matrix transpose (no local memory needed).
* `OPENCL_LIBSMM_TRANS_BLOCK_M`: non-negative integer number (less/equal than the M-extent) denoting the blocksize in M-direction.

For multiplying matrices (implemented in `opencl_libsmm.c`):

* `OPENCL_LIBSMM_SMM_BUILDOPTS`: character string with build options (compile and link) supplied to the OpenCL runtime compiler.
* `OPENCL_LIBSMM_SMM_ATOMICS`: selects the kind of atomic operation used for global memory updates (`xchg`, `cmpxchg`, `cmpxchg2`), or disables atomic updates (`0`). The latter is to quantify the impact of atomic operations rather than for achieving correct results.
* `OPENCL_LIBSMM_SMM_BATCHSIZE`: non-negative integer number denoting the intra-kernel (mini-)batchsize mainly used to amortize atomic updates of data in global/main memory. The remainder with respect to the "stacksize" is handled by the kernel.
* `OPENCL_LIBSMM_SMM_BLOCK_M`: non-negative integer number (less/equal than the M-extent) denoting the blocksize in M-direction.
* `OPENCL_LIBSMM_SMM_BLOCK_N`: non-negative integer number (less/equal than the N-extent) denoting the blocksize in N-direction.
* `OPENCL_LIBSMM_SMM_PARAMS`: Disable embedded/auto-tuned parameters (`0`), or load CSV-file (e.g., `path/to/tune_multiply.csv`).

**NOTE**: LIBSMM's tunable runtime settings may be non-smooth in the sense of enabling a distinct code-path depending on a specific value, e.g., `OPENCL_LIBSMM_SMM_BATCHSIZE=1` vs. `OPENCL_LIBSMM_SMM_BATCHSIZE=2`.

## Auto Tuning

Auto tuning code for performance is a practical way to find the "best" setting for parameterized code (e.g., GPU kernels). Introducing effective parameters is a prerequisite, and exploring the (potentially) high-dimensional parameter space in an efficient way is an art. It is desirable to have reasonable defaults even without auto-tuning the parameters. It would be even better to avoid auto-tuning if best performance was possible right away, i.e., if auto-tuning is not able to find better settings.

For the OpenCL based LIBSMM, `OPENCL_LIBSMM_SMM_BATCHSIZE` (`OPENCL_LIBSMM_SMM_BS`), `OPENCL_LIBSMM_SMM_BLOCK_M` (`OPENCL_LIBSMM_SMM_BM`), and `OPENCL_LIBSMM_SMM_BLOCK_N` (`OPENCL_LIBSMM_SMM_BN`) are explored using [OpenTuner](http://opentuner.org/). The script [tune_multiply.py](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/tune_multiply.py) (or tune_multiply.sh) leverages the `acc_bench_smm` benchmark by parsing console output (timing, data type, etc.). This way, the tuning is implemented without being intermingled with the subject being tuned.

**NOTE**: If `tune_multiply.py` (or `tune_multiply.sh`) are called with `OPENCL_LIBSMM_SMM_BATCHSIZE` (`OPENCL_LIBSMM_SMM_BS`), `OPENCL_LIBSMM_SMM_BLOCK_M` (`OPENCL_LIBSMM_SMM_BM`), or `OPENCL_LIBSMM_SMM_BLOCK_N` (`OPENCL_LIBSMM_SMM_BN`) already set, the respective parameter is considered fixed (and not auto-tuned).

To toggle the benchmarks between tuning single-precision (SP) and double-precision (DP), `make ELEM_TYPE=float` can be used when building the benchmark drivers (backend). However, the `ELEM_TYPE` can be also directly edited in [acc_bench_smm.c](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_bench_smm.c#L26). Auto-tuned parameters for SP and DP can be embedded into the same final application and are considered correctly at runtime.

To build the benchmarks in double-precision (`ELEM_TYPE=double` is default):

```bash
cd src/acc/opencl
make DBG=0
```

To build the benchmarks in single-precision (SP):

```bash
cd src/acc/opencl
make DBG=0 ELEM_TYPE=float
```

To auto-tune, please install the Python `wheel` and `opentuner` packages:

```bash
cd src/acc/opencl/smm
pip install -r requirements.txt
```

The OpenTuner script supports several command line arguments (`tune_multiply.py --help`). For example, `--stop-after=300` can be of interest to finish in five minutes (without limit, OpenTuner decides when the auto-tuning process is finished). A single kernel can be selected by M, N, and K parameters (GEMM), e.g., `M=15`, `N=5`, and `K=7`:

```bash
./tune_multiply.py 13 5 7
```

**NOTE**: If multiple different kernels are tuned using `tune_multiply.py`, it is advisable to delete the `opentuner.db` directory prior to tuning a different kernel since otherwise auto-tuning is potentially (mis-)guided by information which was collected for a different kernel (`tune_multiply.sh` does this automatically).

The OpenTuner script implements multiple objectives ("cost"), primarily "accuracy" (maximized) and a secondary objective "size" (minimized). The former represents the achieved performance (GFLOPS/s) while the latter represents an artificial kernel requirement (just to prefer one parameter set over another in case of similar performance). The console output looks like:

```text
[    15s]    INFO opentuner.search.plugin.DisplayPlugin: tests=8, best {'BS': 32, 'BM': 6, 'BN': 1}, cost accuracy=28.80000000, size=1.0, found by UniformGreedyMutation
[    27s]    INFO opentuner.search.plugin.DisplayPlugin: tests=19, best {'BS': 48, 'BM': 8, 'BN': 1}, cost accuracy=32.20000000, size=1.0, found by UniformGreedyMutation
[    40s]    INFO opentuner.search.plugin.DisplayPlugin: tests=31, best {'BS': 48, 'BM': 8, 'BN': 1}, cost accuracy=32.20000000, size=1.0, found by UniformGreedyMutation
[    54s]    INFO opentuner.search.plugin.DisplayPlugin: tests=43, best {'BS': 48, 'BM': 8, 'BN': 1}, cost accuracy=32.20000000, size=1.0, found by UniformGreedyMutation
[    67s]    INFO opentuner.search.plugin.DisplayPlugin: tests=53, best {'BS': 48, 'BM': 8, 'BN': 1}, cost accuracy=32.20000000, size=1.0, found by UniformGreedyMutation
```

The script finally writes a JSON-file with a filename like `tune_multiply-float-12x12x12-60gflops.json` which is encoding the benchmark (multiply), the precision (float), the kernel (12x12x12), and the achieved performance (60gflops). The script handles SIGINT (like Ctrl-C), and output is still written despite of not terminating normally (can be abused to tune interactively). Tuning starts from an internal default that is supposed to match LIBSMM's internal default parameters. However, tuning can be (re-)started with specific parameters (e.g., `-bs 64`, `-bm 13`, `-bn 1` for `OPENCL_LIBSMM_SMM_BATCHSIZE`, `OPENCL_LIBSMM_SMM_BLOCK_M`, and `OPENCL_LIBSMM_SMM_BLOCK_N` respectively).

**NOTE**: The `acc_bench_smm` executable is potentially started many times during auto-tuning parameters, therefore it is advisable to keep the state of the GPU driver stack persistent (if the setup would otherwise unload the driver configuration), e.g., `nvidia-smi -pm ENABLED`. This can happen in cases where the GPU is not used other than for compute (e.g., in case of a "headless" system).

## Optimized Kernels

JSON-files in the above mentioned smm-directory are automatically summarized into a CSV-file (can be disabled). Further and beyond actual auto-tuning kernels, [tune_multiply.py](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/tune_multiply.py) can be used to perform some basic operations on collected data: explicitly merging all JSON-files into a CSV-file (`tune_multiply.py -m`), and updating the device name in all JSON-files according to current driver version (`tune_multiply.py -u`).

Collected or auto-tuned parameters achieved with single-precision (SP), double-precision (DP), or from different devices can be safely combined. However, care must still be taken to not summarize unrelated results, e.g., after (major) source code changes. The CSV-file is automatically incorporated into LIBSMM by the next clean (re-)build. The format of the CSV-file is assumed to contain column names in the first row (header).

```bash
cd src/acc/opencl
make realclean
make DBG=0
```

This way auto-tuned kernels just work and can be of course exercised using the afore mentioned benchmark:

```bash
cd src/acc
./acc_bench_smm 5 30000 13 5 7
```

Tuned parameters can be also disabled at runtime like:

```bash
cd src/acc
OPENCL_LIBSMM_SMM_PARAMS=0 ./acc_bench_smm 5 30000 13 5 7
```

Further, a CSV-file can be supplied to override embedded parameters or defaults:

```bash
cd src/acc
OPENCL_LIBSMM_SMM_PARAMS=opencl/smm/tune_multiply.csv ./acc_bench_smm 5 30000 13 5 7
```

To tune multiple kernels in a convenient fashion, a triplet specification can be supplied to the [tune_multiply.sh](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/tune_multiply.sh) wrapper script. This script estimates the total runtime for auto-tuning kernels, cleans up intermediate results (`opentuner.db`), allows to specify triplets, and to split work to auto-tune in parallel.

Triplets are used to conveniently describe multiple kernels. A triplet specification consists of comma-separated groups of M,N,K-extents, i.e., matrix shapes according to GEMM. For example:

```text
4 10 15, 6 7 8, 23
```

This triplet specification expands to 55 kernels using the Cartesian product, concatenating the triplets from all expanded groups by combining all values within a comma-separated group. Further, the wrapper script allows to limit the time spent for tuning a single kernel and to partition the amount of kernels to be tuned, e.g., among a cluster of eight systems (below the first partition out of eight would be processed with five minutes per kernel and about 35 minutes in total per partition).

```bash
cd src/acc/opencl/smm
./tune_multiply.sh 300  8 1  4 10 15, 6 7 8, 23
```

The script `tune_multiply.sh` is tuning 1266 kernels by default (`./tune_multiply.sh 300 8 1` taking approximately 13 hours per part). If the process is interrupted earlier (per SIGINT or Ctrl-C), the execution terminates for all requested kernels (triplet specification) unless an environment variable `CONTINUE=1` is set (proceeds to the next kernel).
