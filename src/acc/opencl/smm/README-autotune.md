# Auto Tuning

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
