# LIBSMM (OpenCL)

## Overview

The LIBSMM library implements the [ACC LIBSMM interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_libsmm.h), and depends on the [OpenCL backend](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/README.md).

## Customization

Compile-time settings are (implicitly) documented and can be adjusted by editing [opencl_libsmm.h](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/opencl_libsmm.h), e.g., `OPENCL_LIBSMM_VALIDATE` is disabled by default but can be enabled for debug purpose. The `OPENCL_LIBSMM_VALIDATE` compile-time setting enables side-by-side validation of matrix transpose and multiply operations between device and host. For example, running DBCSR's unit tests with `OPENCL_LIBSMM_VALIDATE` enabled produces console output that allows to pin-point a kernel which misses validation. Runtime settings are made by the means of environment variables. The OpenCL backend provides `acc_getenv.sh` to list all occurrences of `getenv` categorized into "OpenCL Backend environment variables" and "OpenCL LIBSMM environment variables". Common backend related settings are:

* `ACC_OPENCL_DEVSPLIT`: integer enabling devices to be split into subdevices (non-zero/default: subdevices, zero: aggregated).
* `ACC_OPENCL_DEVTYPE`: character string selecting "cpu", "gpu", "all" (unfiltered), or any other string (neither CPU or GPU).
* `ACC_OPENCL_DEVICE`: non-negative integer number to select a device from the (internally enumerated) list of devices.
* `ACC_OPENCL_VENDOR`: character string matching the vendor of the OpenCL device in a case-insensitive fashion, e.g., "intel".
* `ACC_OPENCL_VERBOSE`: verbosity level (integer) with console output on `stderr`.
    * `ACC_OPENCL_VERBOSE=1`: outputs the number of devices found and the name of the selected device.
    * `ACC_OPENCL_VERBOSE=2`: outputs the duration needed to generate a requested kernel.
    * `ACC_OPENCL_VERBOSE=3`: outputs device-side performance of kernels (every launch profiled).
* `ACC_OPENCL_DUMP`: dump preprocessed kernel source code (1) or dump compiled OpenCL kernels (2).
    * `ACC_OPENCL_DUMP=1`: dump preprocessed kernel source code and use it for JIT compilation. Instantiates the original source code using preprocessor definitions (`-D`) and collapses the code accordingly.
    * `ACC_OPENCL_DUMP=2`: dump compiled OpenCL kernels (depends on OpenCL implementation), e.g., PTX code on Nvidia.

There are two categories for the two domains in OpenCL based LIBSMM, i.e., matrix transpose (`OPENCL_LIBSMM_TRANS_*`) and matrix multiplication (`OPENCL_LIBSMM_SMM_*`). For transposing matrices, the settings are:

* `OPENCL_LIBSMM_TRANS_BUILDOPTS`: character string with build options (compile and link) supplied to the OpenCL runtime compiler.
* `OPENCL_LIBSMM_TRANS_INPLACE`: Boolean value (zero or non-zero integer) for in-place matrix transpose (no local memory needed).
* `OPENCL_LIBSMM_TRANS_BM`: non-negative integer number (less/equal than the M-extent) denoting the blocksize in M-direction.

The most common settings for multiplying matrices are:

* `OPENCL_LIBSMM_SMM_BUILDOPTS`: character string with build options (compile and link) supplied to the OpenCL runtime compiler.
* `OPENCL_LIBSMM_SMM_ATOMICS`: selects the kind of atomic operation used for global memory updates (`xchg`, `cmpxchg`, `cmpxchg2`), attempts to force atomic instructions, or disables atomic instructions (`0`). The latter is for instance to quantify the impact of atomic operations.
* `OPENCL_LIBSMM_SMM_PARAMS`: Disable embedded/auto-tuned parameters (`0`), or load CSV-file (e.g., `path/to/tune_multiply.csv`).
* `OPENCL_LIBSMM_SMM_BS`: non-negative integer number denoting the intra-kernel (mini-)batchsize mainly used to amortize atomic updates of data in global/main memory. The remainder with respect to the "stacksize" is handled by the kernel.
* `OPENCL_LIBSMM_SMM_BM`: non-negative integer number (less/equal than the M-extent) denoting the blocksize in M-direction.
* `OPENCL_LIBSMM_SMM_BN`: non-negative integer number (less/equal than the N-extent) denoting the blocksize in N-direction.
* `OPENCL_LIBSMM_SMM_AP`: specifies access to array of parameters (batch or "stack").
* `OPENCL_LIBSMM_SMM_AA`: specifies access to array of A-matrices.
* `OPENCL_LIBSMM_SMM_AB`: specifies access to array of B-matrices.
* `OPENCL_LIBSMM_SMM_AC`: specifies access to array of C-matrices.

The full list of tunable parameters and some explanation can be received with `smm/tune_multiply.py --help`, i.e., short description, default settings, and accepted values.

**NOTE**: LIBSMM's tunable runtime settings can be non-smooth like producing distinct code-paths, e.g., `OPENCL_LIBSMM_SMM_BS=1` vs. `OPENCL_LIBSMM_SMM_BS=2`.

## Auto Tuning

Auto tuning code for performance is a practical way to find the "best" setting for parameterized code (e.g., GPU kernels). Introducing effective parameters is a prerequisite, and exploring the (potentially) high-dimensional parameter space in an efficient way is an art. It is desirable to have reasonable defaults even without auto-tuning the parameters. It would be even better to avoid auto-tuning if best performance was possible right away.

For the OpenCL based LIBSMM, a variety of parameters are explored using [OpenTuner](http://opentuner.org/). The script [tune_multiply.py](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/tune_multiply.py) (or tune_multiply.sh) leverages the `acc_bench_smm` by parsing console output (timing, data type, etc.). This way, the tuning is implemented without being intermingled with the subject being tuned. The "communication" between the tuner and the executable is solely based on environment variables.

**NOTE**: If `tune_multiply.py` (or `tune_multiply.sh`) is called with an environment variable already set, the respective parameter (e.g., `OPENCL_LIBSMM_SMM_BM` or `OPENCL_LIBSMM_SMM_BN`) is considered fixed (and not tuned automatically). This way, the parameter space is reduced in size and effort can be directed more intensely towards the remaining parameters.

To toggle the benchmarks between tuning single precision (SP) and double precision (DP), `make ELEM_TYPE=float` can be used when building the benchmark drivers (`ELEM_TYPE` can be also directly edited in [acc_bench_smm.c](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_bench_smm.c#L26)). Auto-tuned parameters for SP and DP can be embedded into the same final application and are considered correctly at runtime.

To build the benchmarks in double precision (`ELEM_TYPE=double` is default):

```bash
cd src/acc/opencl
make
```

To build the benchmarks in single precision (SP):

```bash
cd src/acc/opencl
make ELEM_TYPE=float
```

To auto-tune, please install the Python `wheel` and `opentuner` packages:

```bash
cd src/acc/opencl/smm
pip install -r requirements.txt
```

The OpenTuner script supports several command line arguments (`tune_multiply.py --help`). For example, `--stop-after=300` can be of interest to finish in five minutes (without limit, OpenTuner decides when the auto-tuning process is finished). A single kernel can be selected by M, N, and K parameters (GEMM), e.g., `M=15`, `N=5`, and `K=7`:

```bash
./tune_multiply.py 13x5x7
```

**NOTE**: If multiple different kernels are tuned using `tune_multiply.py`, it is advisable to delete the `opentuner.db` directory prior to tuning a different kernel since otherwise auto-tuning is potentially (mis-)guided by information which was collected for a different kernel (`tune_multiply.sh` does this automatically).

The OpenTuner script implements multiple objectives ("cost"), primarily "accuracy" (maximized) and a secondary objective "size" (minimized). The former represents the achieved performance (GFLOPS/s) while the latter represents an artificial kernel requirement (just to prefer one parameter set over another in case of similar performance). The console output looks like ("accuracy" denotes performance in GFLOPS/s):

```text
[    15s]    INFO opentuner.search.plugin.DisplayPlugin: tests=8, best {'BS': 32, 'BM': 6, 'BN': 1}, cost accuracy=28.80000000, size=1.0, found by UniformGreedyMutation
[    27s]    INFO opentuner.search.plugin.DisplayPlugin: tests=19, best {'BS': 48, 'BM': 8, 'BN': 1}, cost accuracy=32.20000000, size=1.0, found by UniformGreedyMutation
[    40s]    INFO opentuner.search.plugin.DisplayPlugin: tests=31, best {'BS': 48, 'BM': 8, 'BN': 1}, cost accuracy=32.20000000, size=1.0, found by UniformGreedyMutation
[    54s]    INFO opentuner.search.plugin.DisplayPlugin: tests=43, best {'BS': 48, 'BM': 8, 'BN': 1}, cost accuracy=32.20000000, size=1.0, found by UniformGreedyMutation
[    67s]    INFO opentuner.search.plugin.DisplayPlugin: tests=53, best {'BS': 48, 'BM': 8, 'BN': 1}, cost accuracy=32.20000000, size=1.0, found by UniformGreedyMutation
```

The script finally writes a JSON-file with a filename like `tune_multiply-float-12x12x12-s15-60gflops.json` which is encoding the benchmark ("multiply"), the precision ("float"), the kernel ("12x12x12"), the number of bits necessary to represent the size of the problem, i.e., log2 of the problem-size ("s15"), and the achieved performance ("60gflops"). The script handles SIGINT (like Ctrl-C), and output is still written despite of abnormally terminating (can be abused to tune interactively). Tuning starts from an internal default that is supposed to match LIBSMM's internal default parameters. However, tuning can be (re-)started with specific parameters (e.g., `-bs 64`, `-bm 13`, `-bn 1` for `OPENCL_LIBSMM_SMM_BS`, `OPENCL_LIBSMM_SMM_BM`, and `OPENCL_LIBSMM_SMM_BN` respectively), or partially fixed for a subset of parameters.

**NOTE**: The `acc_bench_smm` executable is potentially started many times when auto-tuning parameters, therefore it is advisable to keep the state of the GPU driver stack persistent (if the setup would otherwise unload the driver configuration), e.g., `nvidia-smi -pm ENABLED`. This can happen in cases where the GPU is only for compute and not used for graphics (no X-Window system, e.g., in case of a "headless" system). Time needed for tuning parameters is not only impacted by accessing and readying the device, but also by the time needed to compile a kernel at runtime aka Just-In-Time (JIT).

## Optimized Kernels

JSON-files in the above mentioned smm-directory are automatically summarized into a CSV-file. Further and beyond auto-tuning kernels, [tune_multiply.py](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/tune_multiply.py) can be used to perform basic operations on collected data: explicitly merging all JSON-files into a CSV-file (`tune_multiply.py -m`), and updating the device name in all JSON-files according to current driver version (`tune_multiply.py -u`).

Collected or auto-tuned parameters achieved with single precision (SP), double precision (DP), or from different devices can be safely combined. Practically, `acc_opencl.sh` transforms the CSV-file into source code compiled into the final binary, which is independent of `OPENCL_LIBSMM_SMM_PARAMS` accepting a CSV-file (path/filename). However, `acc_opencl.sh` currently limits the origin of parameters to a single device. Care must still be taken to not summarize unrelated results, e.g., after (major) source code changes. The CSV-file is automatically incorporated into LIBSMM by the next clean (re-)build. The format of the CSV-file is assumed to contain column names in the first row (header).

Different problem sizes (like "s15"; see above) are not represented individually, but are instead collected into a maximum value. In turn, this means tuning for a non-default problem-size must be manually kept pure since the result achieved with a larger problem may dominate (maximum value).

```bash
cd src/acc/opencl
make realclean
make
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

By supplying a CSV-file at runtime, embedded parameters and defaults are overriden, and given parameters are applied even if the current device is different from what would match the given parameters:

```bash
cd src/acc
OPENCL_LIBSMM_SMM_PARAMS=opencl/smm/tune_multiply.csv ./acc_bench_smm 5 30000 13 5 7
```

To tune multiple kernels in a convenient fashion, a triplet specification can be supplied to the [tune_multiply.sh](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/tune_multiply.sh) wrapper script. This script estimates the total runtime for auto-tuning kernels, cleans up intermediate results (`opentuner.db`), allows to specify triplets, and splits work to auto-tune in parallel.

Triplets are used to conveniently describe multiple kernels. A triplet specification consists of comma-separated groups of (M,N,K)-extents, i.e., matrix shapes according to GEMM. For example:

```text
4 10 15, 6 7 8, 23
```

This triplet specification expands to 55 kernels using the Cartesian product within each group and concatenating the result of such expanded groups followed by removing duplicate triplets. Further, the wrapper script allows to limit the time spent for tuning a single kernel and to partition the number of kernels to be tuned, e.g., among a cluster of eight systems (below the first partition out of eight would be processed with five minutes per kernel and about 35 minutes in total per partition).

```bash
cd src/acc/opencl/smm
./tune_multiply.sh -t 300  -j 8 -i 1  4 10 15, 6 7 8, 23
```

The script `tune_multiply.sh` is tuning 1266 kernels by default (`./tune_multiply.sh -t 300 -j 8 -i 1` takes approximately 13 hours per part). If the process is interrupted earlier (per SIGINT or Ctrl-C), the execution terminates for all requested kernels (triplet specification) unless `--continue` is given (or `-c`, or an environment variable `CONTINUE=1`).

For convenience, it is possible to "update" an existing set of JSON-files (path can be given with `-p`), i.e., to parse the (M,N,K)-triplet denoted by the JSON filename and to re-tune with an almost unconstrained tuning-level (`-a 1` by default) as well as a limited duration (160 seconds per kernel by default).

```bash
cd src/acc/opencl
make realclean
echo "Rebuild and embed smm/params/tune_multiply_P100.csv"
make WITH_GPU=P100

echo "Retune original parameters"
smm/tune_multiply.sh -p smm/params/p100 -u

echo "Override original parameters"
cp tune_multiply.csv smm/params/tune_multiply_P100.csv
```

Tuning kernels further is only sensible if the previously tuned parameters are embedded into the binary (such that the process does not start from scratch). Retuned parameters are captured with JSON-files as usual.
