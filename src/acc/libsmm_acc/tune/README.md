# Autotuning Procedure for Finding Optimal CUDA/HIP Kernel Parameters in `libsmm_acc`

The performance of the matrix-matrix multiplication kernels is highly dependent on the choice of algorithm and parameters. This is why autotuning is used to find optimal kernel parameters.

---

### Requirements

Python version required: `python 3.6`

If you are about to autotune parameters for a new GPU (i.e. a GPU for which there are no autotuned parameters yet), please first follow [the instructions for a new GPU](../README.md#adding-support-for-a-new-gpu-card).

---

### Autotuning procedure

#### 1. Go to the `libsmm_acc/tune` directory

```bash
$ cd dbcsr/src/acc/libsmm_acc/libsmm_acc/tune
```

The `parameters.h` file (a C++ header file generated from the JSON record of multiplication kernels and their optimal parameters) is needed for the autotuning procedure. One can copy it over from a build directory for example, as follows:
```bash
$ cp ~/dbcsr/build_dir/src/acc/libsmm_acc/parameters.h ../
```

#### 2. Adapt `tune_setup.py` to your environment

The `tune_setup.py` script generates job files. You have to adapt the script to the environment of your supercomputer and your personal settings.

```
...
  def gen_jobfile(outdir, m, n, k):

    ...

    output = "#!/bin/bash -l\n"
    output += "#SBATCH --nodes=%d\n" % num_nodes
    output += "#SBATCH --ntasks-per-core=1\n"
    output += "#SBATCH --ntasks-per-node=1\n"
    output += "#SBATCH --cpus-per-task=" + "%d\n" % cpus_per_node
    output += "#SBATCH --time=%s\n" % time
    output += "#SBATCH --partition=normal\n"
    output += "#SBATCH --constraint=gpu\n"
    output += "\n"
    output += "source ${MODULESHOME}/init/sh;\n"
    output += "module load daint-gpu\n"
    output += "module unload PrgEnv-cray\n"
    output += "module load PrgEnv-gnu\n"
    if compiler == "nvcc":
        output += "module load cudatoolkit/8.0.61_2.4.9-6.0.7.0_17.1__g899857c\n"
    else: # i.e. compiler = hipcc
        output += "module load hip\n"
    output += "module list\n"
    output += "export CRAY_CUDA_MPS=1\n"
    output += "cd $SLURM_SUBMIT_DIR \n"
    output += "\n"
    output += "date\n"

    ...

...
```

#### 3. Run the script `tune_setup.py`

Specify which GPU you are autotuning for by passing the appropriate `parameters_GPU.json` file as an argument with `-p`.
In addition, the script takes as arguments the block sizes you want to add to `libsmm_acc`. You can specify these as a list of integers or provide the parameter file of a different GPU from which to read the block sizes to autotune.

For example, if the system you want to autotune for contains blocks of size 5 and 8, run:

```bash
$ ./tune_setup.py 5 8 -p ../parameters/parameters_P100.json
Reading parameters from parameters_P100.json
libsmm_acc: Found 74096 existing parameter sets, of which 1641 are autotuned and 72455 are predicted.
Requested to autotune 8 triplets
Found 41824 parameter sets for 5x5x5
Found 83648 parameter sets for 5x5x8
Found 103072 parameter sets for 5x8x5
Found 103072 parameter sets for 5x8x8
Found 103072 parameter sets for 8x5x5
Found 103072 parameter sets for 8x5x8
Found 125344 parameter sets for 8x8x5
Found 125344 parameter sets for 8x8x8
```

Or, if you want to obtain, for the NVIDIA P100, the parameters of the same block sizes as recorded for the NVIDIA K40, run:

```bash
$ ./tune_setup.py -p ../parameters/parameters_P100.json ../parameters/parameters_K40.json
Reading parameters from parameters_P100.json
libsmm_acc: Found 74093 existing parameter sets, of which 1638 are autotuned and 72455 are predicted.
Reading parameters to autotune from parameters_K40.json
Requested to autotune 19 triplets
Found 41824 parameter sets for 5x5x5
Found 95648 parameter sets for 6x6x6
Found 110496 parameter sets for 7x7x7
Found 125344 parameter sets for 8x8x8
Found 173764 parameter sets for 9x9x9
...
```

The script will create a directory for each combination of the block sizes:

```bash
$ ls -d tune_*
tune_5x5x5  tune_5x5x8  tune_5x8x5  tune_5x8x8  tune_8x5x5  tune_8x5x8  tune_8x8x5  tune_8x8x8
```

Each directory contains a number of files:

```bash
$ ls -1 tune_8x8x8/
Makefile
tune_8x8x8_exe0_main.cu/cpp
tune_8x8x8_exe0_part0.cu/cpp
tune_8x8x8_exe0_part1.cu/cpp
tune_8x8x8_exe0_part2.cu/cpp
tune_8x8x8_exe0_part3.cu/cpp
tune_8x8x8_exe0_part4.cu/cpp
tune_8x8x8.job
```

For each possible parameter-set a *launcher* is generated. A launcher is a small snippet of C code, which launches the kernel by using the CUDA specific `<<< >>>`-notation or HIP's `hipLaunchKernelGGL` function. It also instantiates the C++ template which contains the actual kernel code.

In order to parallelize the benchmarking, the launchers are distributed over multiple executables. Currently, up to 10'000 launchers are benchmarked by one *executable*. Each executable is linked together from several `tune_*_part???.o` and a `tune_*_main.o`. Each part-files contains up to 100 launchers. This allows to parallelize the compilation over multiple CPU cores.

#### 4. Adapt `tune_submit.py` to your environment

The script `tune_submit.py` was written for the slurm batch system as used e.g. by CRAY supercomputers. If your computer runs a different batch system, you have to adapt `tune_submit.py` accordingly.

#### 5. Submit Jobs

Each tune-directory contains a job file. Since there might be many tune-directories, the convenience script `tune_submit.py` can be used to submit jobs. It will go through all the `tune_*`-directories and check if its job has already been submitted or run. For this, the script calls `squeue` in the background and it searches for `slurm-*.out`files. In order to limit the number of jobs submitted at a time, a maximum number of jobs to submit can be specified with `-j`.

When `tune_submit.py` is called without arguments, it will just list the jobs that could be submitted:

```bash
$ ./tune_submit.py 
          tune_5x5x5: Would submit, run with "doit!"
          tune_5x5x8: Would submit, run with "doit!"
          tune_5x8x5: Would submit, run with "doit!"
          tune_5x8x8: Would submit, run with "doit!"
          tune_8x5x5: Would submit, run with "doit!"
          tune_8x5x8: Would submit, run with "doit!"
          tune_8x8x5: Would submit, run with "doit!"
          tune_8x8x8: Would submit, run with "doit!"
Number of jobs submitted: 8
```

Only when `tune_submit.py` is called with `doit!` as its first argument, will it actually submit jobs:

```bash
$ ./tune_submit.py doit!
          tune_5x5x5: Submitting
Submitted batch job 277987
          tune_5x5x8: Submitting
Submitted batch job 277988
          tune_5x8x5: Submitting
Submitted batch job 277989
          tune_5x8x8: Submitting
Submitted batch job 277990
          tune_8x5x5: Submitting
Submitted batch job 277991
          tune_8x5x8: Submitting
Submitted batch job 277992
          tune_8x8x5: Submitting
Submitted batch job 277993
          tune_8x8x8: Submitting
Submitted batch job 277994
Number of jobs submitted: 8
```

#### 6. Collect Results

Run `tune_collect.py` to parse all log files and determine the best kernel for each blocksize:

```bash
$ ./tune_collect.py
Reading: tune_5x5x5/tune_5x5x5_exe0.log
Reading: tune_5x5x8/tune_5x5x8_exe0.log
Reading: tune_5x8x5/tune_5x8x5_exe0.log
Reading: tune_5x8x8/tune_5x8x8_exe0.log
Reading: tune_8x5x5/tune_8x5x5_exe0.log
Reading: tune_8x5x8/tune_8x5x8_exe0.log
Reading: tune_8x8x5/tune_8x8x5_exe0.log
Reading: tune_8x8x8/tune_8x8x8_exe0.log
Kernel_dnt_tiny(m=5, n=5, k=5, split_thread=32, threads=64, grouping=16, minblocks=1) , # 27.9623 GFlops
Kernel_dnt_tiny(m=5, n=5, k=8, split_thread=32, threads=96, grouping=16, minblocks=1) , # 37.8978 GFlops
Kernel_dnt_medium(m=5, n=8, k=5, tile_m=1, tile_n=1, threads=96, grouping=16, minblocks=8) , # 32.9231 GFlops
Kernel_dnt_tiny(m=5, n=8, k=8, split_thread=32, threads=96, grouping=16, minblocks=1) , # 47.0366 GFlops
Kernel_dnt_medium(m=8, n=5, k=5, tile_m=1, tile_n=1, threads=96, grouping=16, minblocks=12) , # 33.1999 GFlops
Kernel_dnt_medium(m=8, n=5, k=8, tile_m=1, tile_n=1, threads=96, grouping=16, minblocks=12) , # 49.3499 GFlops
Kernel_dnt_tiny(m=8, n=8, k=5, split_thread=32, threads=96, grouping=16, minblocks=1) , # 62.8469 GFlops
Kernel_dnt_tiny(m=8, n=8, k=8, split_thread=32, threads=128, grouping=16, minblocks=1) , # 90.7763 GFlops

Wrote parameters.json
```

The file `parameters.json` in `dbcsr/src/acc/libsmm_acc/parameters` now contains the newly autotuned parameters.

#### 7. Merge new parameters with original parameter-file

Run `tune_merge.py` to merge the new parameters with the original ones:

```bash
$ ./tune_merge.py
Merging parameters.json with parameters_P100.json
Wrote parameters.new.json
```

The file `parameters.new.json` can now be used as a parameter file. Rename it to `parameters_GPU.json`, with the appropriate `GPU`.

#### 8. (optional) Explore the data

Explore the data interactively using the [provided Jupyter Notebook](notebooks/inspect_training_data.ipynb).

#### 9. Contribute parameters to the community

**Contribute new optimal parameters**

Submit a pull request updating the appropriate `parameters_GPU.json` file to the [DBCSR repository](https://github.com/cp2k/dbcsr).

**Contribute autotuning data**

See [instructions](https://github.com/cp2k/dbcsr-data#contributing) in DBCSR's [data repository](https://github.com/cp2k/dbcsr-data).
