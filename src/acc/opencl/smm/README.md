# LIBSMM (OpenCL)

## Overview

The LIBSMM library implements the [ACC LIBSMM interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_libsmm.h), and depends on the [OpenCL backend](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/README.md). At least the compile-time settings below are typically for development, e.g., when attempting to contribute new functionality or features, or meant for debug purpose (and not necessarily settings to be made when using DBCSR or CP2K).

## Customization

### Compile-time Settings

Compile-time settings are (implicitly) documented and can be adjusted by editing [opencl_libsmm.h](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/opencl_libsmm.h) (adjusting the build-line as per `-D` is possible as well but less convenient). For example, `OPENCL_LIBSMM_F32` is enabled by default but can be disabled, or `OPENCL_LIBSMM_DEBUG` (which is disabled by default) can be enabled for debug purpose.

The `OPENCL_LIBSMM_DEBUG` compile-time setting enables side-by-side validation of matrix transpose and multiply operations on GPU against a built-in CPU implementation. For example, running DBCSR's unit tests with this setting produces useful console output that allows to pin-point the exact call arguments causing a validation error.

### Runtime Settings

Runtime settings are made by the means of environment variables (implemented in `opencl_libsmm.c`). There are two categories (for the two major functions) like matrix transpose (`OPENCL_LIBSMM_TRANS_*`) and matrix multiplication (`OPENCL_LIBSMM_SMM_*`). Common settings are (see OpenCL backend documentation for more details):

* `ACC_OPENCL_DEVSPLIT`: integer enabling devices to be split into subdevices (non-zero: enabled, default/zero: disabled).
* `ACC_OPENCL_DEVTYPE`: character string matching the device-kind like "cpu", "gpu", or another kind if neither CPU or GPU.
* `ACC_OPENCL_DEVICE`: non-negative integer number to select a device from the (internally enumerated) list of devices.
* `ACC_OPENCL_VENDOR`: character string matching the vendor of the OpenCL device in an case-insensitive fashion, e.g., "intel".
* `ACC_OPENCL_VERBOSE`: verbosity level (integer) with console output on `stderr`.
    * `ACC_OPENCL_VERBOSE=1`: outputs the number of devices found and the name of the selected device.
    * `ACC_OPENCL_VERBOSE=2`: outputs the duration needed to generate a requested kernel.
    * `ACC_OPENCL_VERBOSE=3`: outputs device-side measured performance of kernels (geometric mean).
    * `ACC_OPENCL_VERBOSE=4`: outputs device-side performance of kernels (every launch profiled).

For tranposing matrices:

* `OPENCL_LIBSMM_TRANS_BUILDOPTS`: character string with build options (compile and link) supplied to the OpenCL runtime compiler.
* `OPENCL_LIBSMM_TRANS_INPLACE`: Boolean value (zero or non-zero integer) for inplace matrix transpose (no local memory needed).
* `OPENCL_LIBSMM_TRANS_BLOCK_M`: non-negative integer number (less/equal than the M-extent) denoting the blocksize in M-direction.

For multiplying matrices:

* `OPENCL_LIBSMM_SMM_BUILDOPTS`: character string with build options (compile and link) supplied to the OpenCL runtime compiler.
* `OPENCL_LIBSMM_SMM_ATOMICS`: selects the kind of atomic operation used for global memory updates ("xchg", "cmpxchg", "cmpxchg2"), or disables atomic updates ("0"). The latter is to quantify the impact of atomic operations rather than for achieving correct results.
* `OPENCL_LIBSMM_SMM_BATCHSIZE`: non-negative integer number denoting the intr-kernel (mini-)batchsize mainly used to amortize atomic updates of data in global/main memory. The remainder with respect to the "stacksize" is handled by the kernel.
* `OPENCL_LIBSMM_SMM_BLOCK_M`: non-negative integer number (less/equal than the M-extent) denoting the blocksize in M-direction.
* `OPENCL_LIBSMM_SMM_BLOCK_N`: non-negative integer number (less/equal than the N-extent) denoting the blocksize in N-direction.

**NOTE**: LIBSMM's tunable runtime settings may be non-smooth in the sense of enabling a distinct code-path depending on a specific value, e.g., `OPENCL_LIBSMM_SMM_BATCHSIZE=1` vs. `OPENCL_LIBSMM_SMM_BATCHSIZE=2`.

## Auto Tuning

Auto tuning code for performance is a practical way to find the "best" setting for parameterized code (e.g., GPU kernels). Introducing effective parameters is a prerequisite, and exploring the (potentially) high-dimensional parameter space in an efficient way is an art. It is desirable to have reasonable defaults even without auto-tuning the parameters. It would be even better to avoid auto-tuning if best performance was possible right away, i.e., if auto-tuning is not able to find better settings.

For the OpenCL based LIBSMM, `OPENCL_LIBSMM_SMM_BATCHSIZE`, `OPENCL_LIBSMM_SMM_BLOCK_M`, and `OPENCL_LIBSMM_SMM_BLOCK_N` are explored using [OpenTuner](http://opentuner.org/). The script [tune_multiply.py](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/tune_multiply.py) leverages the `acc_bench_smm` benchmark by parsing console output (timing, data type, etc.). This way, the tuning is implemented without being intermingled with the subject being tuned.

**NOTE**: To toggle between tuning single-precision (SP) and double-precision (DP), the `ELEM_TYPE` in [acc_bench_smm.c](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_bench_smm.c#L22) can be edited. Auto-tuned parameters for SP and DP can both be embedded into the final application and are picked up correctly at runtime.

To build the benchmarks:

```bash
cd src/acc/opencl
make DBG=0
```

To auto-tune, please install the Python `wheel` and `opentuner` packages:

```bash
cd src/acc/opencl/smm
pip install -r requirements.txt
```

The OpenTuner script supports several command line arguments (`tune_multiply.py --help`). For example, `--stop-after=300` can be interest to finish in five minutes (without a limit, OpenTuner decides when the auto-tuning process is finished). A single kernel can be selected by M, N, and K parameters (GEMM), e.g., `M=15`, `N=5`, and `K=7`:

```bash
./tune_multiply.py 13 5 7 --no-dups
```

**NOTE**: If multiple different kernels are tuned using `tune_multiply.py`, it is advisible to delete the `opentuner.db` directory prior to a new kernel otherwise auto-tuning is potentially (mis-)guided by information which was collected for a different kernel (`tune_multiply.sh` does this automatically).

The OpenTuner script implements multiple objectives ("cost"), primarily "accuracy" (maximized) and a secondary objective "size" (minimized). The former represents the achieved performance (GFLOPS/s) while the latter represents an artificial kernel requirement (just to prefer one parameter set over another in case of similar performance). The console output looks like:

```text
[    15s]    INFO opentuner.search.plugin.DisplayPlugin: tests=8, best {'BS': 32, 'BM': 6, 'BN': 1}, cost accuracy=28.80000000, size=1.0, found by UniformGreedyMutation
[    27s]    INFO opentuner.search.plugin.DisplayPlugin: tests=19, best {'BS': 48, 'BM': 8, 'BN': 1}, cost accuracy=32.20000000, size=1.0, found by UniformGreedyMutation
[    40s]    INFO opentuner.search.plugin.DisplayPlugin: tests=31, best {'BS': 48, 'BM': 8, 'BN': 1}, cost accuracy=32.20000000, size=1.0, found by UniformGreedyMutation
[    54s]    INFO opentuner.search.plugin.DisplayPlugin: tests=43, best {'BS': 48, 'BM': 8, 'BN': 1}, cost accuracy=32.20000000, size=1.0, found by UniformGreedyMutation
[    67s]    INFO opentuner.search.plugin.DisplayPlugin: tests=53, best {'BS': 48, 'BM': 8, 'BN': 1}, cost accuracy=32.20000000, size=1.0, found by UniformGreedyMutation
```

The script finally writes a JSON-file with a filename like `tune_multiply-float-12x12x12-60gflops.json` which is encoding the benchmark (multiply), the precision (float), the kernel (12x12x12), and the achieved performance (60gflops). The script handles SIGINT (like Ctrl-C), and output is still written despite of not terminating normally (can abused to tune interactively). Tuninig starts from an internal default that is supposed to match LIBSMM's internal default parameters. However, tuning can be (re-)started with specific parameters (e.g., `-bs 64`, `-bm 13`, `-bn 1` for `OPENCL_LIBSMM_SMM_BATCHSIZE`, `OPENCL_LIBSMM_SMM_BLOCK_M`, and `OPENCL_LIBSMM_SMM_BLOCK_N` respectively).

## Optimized Kernels

JSON-files in the above mentioned smm-directory are automatically summarized into a CSV-file (can be disabled). Parameters achieved with single-precision (SP) and double-precision (DP) can be safely combined. However, care must be taken to not summarize unrelated results like for different devices or after (major) kernel changes. The CSV-file contains a header-row with column names, and the content is automatically incorporated into LIBSMM by the next clean (re-)build.

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

To tune multiple kernels in a convenient fashion, a triplet specification can be supplied to the [tune_multiply.sh](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/tune_multiply.sh) wrapper script. This script estimates the total runtime for auto-tuning kernels, cleans up intermediate results (`opentuner.db`), allows to specify triplets, and to split work in order to auto-tune in parallel.

Triplets are used to conveniently describe multiple kernels. A triplet specification consists of comma-separated groups of M,N,K-extents, i.e., matrix shapes according to GEMM. For example:

```text
4 10 15, 6 7 8, 23
```

This triplet specification expands to 55 kernels using the Cartesian product, concatenating the triplets from all expanded groups by combining all values within a comma-separated group. Further, the wrapper script allows to limit the time spent for tuning a single kernel and to partition the amount of kernels to be tuned, e.g., among a cluster of eight systems (below the first partition out of eight would be procesed with five minutes per kernel and about 35 minutes in total per partition).

```bash
cd src/acc/opencl/smm
./tune_multiply.sh 300  8 1  4 10 15, 6 7 8, 23
```

The script `tune_multiply.sh` is tuning 1266 kernels by default (`./tune_multiply.sh 300 8 1` taking approximately 13 hours per part). If the process is interrupted earlier (per SIGINT or Ctrl-C), the execution terminates for all requested kernels (triplet specification) unless an environment variable `CONTINUE=1` is set (proceeds to the next kernel).
