# Optimized Kernels

Optimized kernel parameters are stored in JSON-files and are automatically summarized into a CSV-file. Further and beyond auto-tuning kernels, [tune_multiply.py](https://github.com/cp2k/dbcsr/blob/develop/src/acc/opencl/smm/tune_multiply.py) can be used to perform basic operations on collected data: explicitly merging all JSON-files into a CSV-file (`tune_multiply.py -m`), and updating the device name in all JSON-files according to current driver version (`tune_multiply.py -u`).

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

# Advanced Tuning

To utilize multiple devices per system and to accelerate tuning kernels, `tune_multiply.py` comes with built-in support for running under MPI (SPMD execution model). The basic assumption is to spawn one process per device usually with different kernels tuned per device (SPMD). Of course, tuning the same kernels in parallel on multiple devices is possible but it is a waste of resources. Tuning on multiple devices per system can be also more realistic given the common power budget of all devices and less room for an increased operating frequency per device (Turbo clock speed).

For example, a single dual-socket system with two PVC cards (modules) per socket exposes eight GPU devices (two GPU stacks or tiles per card). Then 350 kernels can be tuned in less than 2 1/2 hours with a duration of 200 seconds for tuning each kernel.

```bash
MAXTIME=200 NPARTS=8 UPDATE=1 JSONDIR=params/pvc mpirun \
  ./tune_multiply.sh -i 1 : \
  ./tune_multiply.sh -i 2 : \
  ./tune_multiply.sh -i 3 : \
  ./tune_multiply.sh -i 4 : \
  ./tune_multiply.sh -i 5 : \
  ./tune_multiply.sh -i 6 : \
  ./tune_multiply.sh -i 7 : \
  ./tune_multiply.sh -i 8 \
>out.log 2>&1
```

**NOTE**: The above shown example prefers environment variables over command-line options that would be common to the eight launches of `tune_multiply.sh`.
