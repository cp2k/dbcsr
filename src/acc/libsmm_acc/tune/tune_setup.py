#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####################################################################################################
# Copyright (C) by the DBCSR developers group - All rights reserved                                #
# This file is part of the DBCSR library.                                                          #
#                                                                                                  #
# For information on the license, see the LICENSE file.                                            #
# For further information please visit https://dbcsr.cp2k.org                                      #
# SPDX-License-Identifier: GPL-2.0+                                                                #
####################################################################################################

import sys
import os
import stat
import json
import math
from itertools import product
import argparse
from pathlib import Path

sys.path.append("../")
from kernels.smm_acc_predict import (  # noqa: E402
    gpu_architectures,
    kernel_algorithm,
    params_dict_to_kernel,
    compatible_mnk,
)


# ===============================================================================
def main(
    param_fn: Path,
    compiler,
    cpus_per_task,
    max_num_nodes,
    blocksizes,
    blocks_from_param_file,
    tune_dir: Path,
):
    # Read existing parameters
    assert (
        param_fn.name in gpu_architectures.keys()
    ), f"Cannot find GPU architecture for file {param_fn.name}"
    arch_code = gpu_architectures[param_fn.name]
    with open("../kernels/gpu_properties.json") as fhandle:
        gpu_properties = json.load(fhandle)[arch_code]
    with open("../kernels/autotuning_properties.json") as fhandle:
        autotuning_properties = json.load(fhandle)
    with param_fn.open("r") as fhandle:
        all_kernels = [params_dict_to_kernel(**params) for params in json.load(fhandle)]
    print(f"Reading parameters from {param_fn}")
    autotuned_kernels = [k for k in all_kernels if k.autotuned]
    predicted_kernels = [k for k in all_kernels if not k.autotuned]
    print(
        "libsmm_acc: found %d existing parameter sets, of which %d are autotuned and %d are predicted."
        % (len(all_kernels), len(autotuned_kernels), len(predicted_kernels))
    )

    # Get blocksizes to be autotuned
    if blocks_from_param_file:  # open and read file
        with open(blocksizes) as f:
            all_kernels_ref = [
                params_dict_to_kernel(**params) for params in json.load(f)
            ]
        print(f"Reading parameters to autotune from {blocksizes}")
        triples = [(k.m, k.n, k.k) for k in all_kernels_ref if k.autotuned]
    else:
        assert len(set(blocksizes)) == len(blocksizes)
        blocksizes.sort()
        # Get (m, n, k) triplets to be autotuned
        triples = combinations(*blocksizes)
    print(f"Requested to autotune {len(triples)} triplets")

    for m, n, k in triples:
        existing = [kern for kern in autotuned_kernels if kern.can_handle(m, n, k)]
        if existing:
            print(
                "Found existing autotuned parameter set for %dx%dx%d, skipping."
                % (m, n, k)
            )
            continue

        outdir = tune_dir / f"tune_{int(m)}x{int(n)}x{int(k)}"
        if outdir.exists():
            print(f"Directory {outdir} exists already, skipping.")
            continue
        outdir.mkdir()
        gen_benchmark(outdir, gpu_properties, autotuning_properties, compiler, m, n, k)
        gen_jobfile(outdir, compiler, m, n, k, cpus_per_task, max_num_nodes)
        gen_makefile(outdir, compiler, arch_code)


# ===============================================================================
def format_params(params):
    output = []
    order = [
        "m",
        "n",
        "k",
        "tile_m",
        "tile_n",
        "w",
        "v",
        "split_thread",
        "threads",
        "blockdim",
        "grouping",
    ]
    for k in order:
        if k in params.keys():
            output.append(f"{k}={int(params[k])}")

    for k in params.keys():
        if k not in order:
            output.append(f"{k}={int(params[k])}")

    return f"({', '.join(output)})"


def get_file_extension_from_compiler(compiler):
    return ".cu" if compiler == "nvcc" else ".cpp"


# ===============================================================================
def gen_benchmark(outdir, gpu_properties, autotuning_properties, compiler, m, n, k):
    includes = []
    launcher_codes = []
    launchers = []
    kernel_descr = []
    indent = "  "
    file_extension = get_file_extension_from_compiler(compiler)

    # Get the kernel algorithms compatible with the given size:
    compatible_kernels = [
        kernclass
        for classname, kernclass in kernel_algorithm.items()
        if compatible_mnk(classname, m, n, k)
    ]

    # Get the parameter sets to measure for this (m,n,k)
    for kernclass in compatible_kernels:
        params = kernclass.promising_parameters(
            m, n, k, gpu_properties, autotuning_properties
        )
        if params == 0:
            continue

        for p in params:
            kern = kernclass(**p, source="autotuning_candidate", perf=0)
            includes.append(f"../../kernels/{kern.include}")
            launcher_codes.append(kern.launcher_code(compiler))
            launchers.append(f"launch_{kern.name}")
            kernel_descr.append(kernclass.__name__ + format_params(p))

    print(f"Found {len(launchers)} parameter sets for {int(m)}x{int(n)}x{int(k)}")
    if len(launchers) == 0:
        return

    # Compose the "include" line of the benchmark code
    incl_output = '#include "../../kernels/smm_acc_common.h"\n'
    for i in set(includes):
        incl_output += f'#include "{i}"\n'
    incl_output += "\n\n"

    # Compose the benchmark code
    # The benchmark is broken down in
    # - n_exe_files executables
    # - each executable is made of n_obj_files object files
    # - each object file is made up of launchers_per_obj launchers
    # - each launcher launches 1 GPU kernel with a certain set of kernel parameters
    max_launchers_per_exe = 10000
    launchers_per_obj = 100
    n_exe_files = int(len(launcher_codes) / max_launchers_per_exe) + 1
    launchers_per_exe = int(len(launcher_codes) / n_exe_files) + 1

    # Compose source code for each executable file
    for i in range(n_exe_files):
        chunk_a = i * launchers_per_exe
        chunk_b = min((i + 1) * launchers_per_exe, len(launcher_codes))
        n_obj_files = math.ceil((chunk_b - chunk_a) / launchers_per_obj)

        if n_obj_files == 0:
            continue
        jdigits = int(math.log10(n_obj_files)) + 1

        # Compose source code for each object file
        for j in range(n_obj_files):
            a = chunk_a + j * launchers_per_obj
            b = min(chunk_a + (j + 1) * launchers_per_obj, chunk_b)
            output = incl_output
            output += "\n\n".join(launcher_codes[a:b])
            fn = outdir / f"tune_{m}x{n}x{k}_exe{i}_part{j:0{jdigits}}{file_extension}"
            writefile(fn, output)

        # Compose source code for "main" of executable file
        output = '#include "../../libsmm_acc_benchmark.h"\n\n'
        for j in range(chunk_b - chunk_a):
            output += (
                f"int {launchers[chunk_a + j]}(const int *param_stack, int stack_size, "
            )
            if compiler == "nvcc":
                output += "cudaStream_t stream, "
            else:
                output += "hipStream_t stream, "
            output += (
                "int m_max, int n_max, int k_max,"
                + " const double *a_data, const double *b_data, double *c_data);\n"
            )

        output += "\n"
        output += "int main(int argc, char** argv) {\n"
        if compiler == "nvcc":
            output += (
                indent
                + "cudaError_t err = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);\n"
            )
            output += f"{indent}if(err != cudaSuccess) return(-1);\n"
        else:  # i.e. compiler = hipcc
            output += (
                indent
                + "hipError_t err = hipDeviceSetSharedMemConfig(hipSharedMemBankSizeEightByte);\n"
            )
            output += f"{indent}if(err != hipSuccess) return(-1);\n"
        output += f"{indent}libsmm_acc_benchmark_t* handle;\n"
        output += f"{indent}KernelLauncher launchers[{int(chunk_b - chunk_a)}];\n"
        output += f"{indent}char *kernel_descr[{int(chunk_b - chunk_a)}];\n"

        for j in range(chunk_b - chunk_a):
            output += f"{indent}launchers[{int(j)}]    = {launchers[chunk_a + j]};\n"
            output += indent + 'kernel_descr[%d] = (char *) "%s";\n' % (
                j,
                kernel_descr[chunk_a + j],
            )
        output += indent + "libsmm_acc_benchmark_init(&handle, tune, %d, %d, %d);\n" % (
            m,
            n,
            k,
        )
        output += (
            indent
            + "int result = libsmm_acc_benchmark(handle, %d, %d, %d, %d, launchers, kernel_descr);\n"
            % (m, n, k, chunk_b - chunk_a)
        )
        output += f"{indent}libsmm_acc_benchmark_finalize(handle);\n"
        output += f"{indent}return result;"
        output += "}\n"

        fn = (
            outdir / f"tune_{int(m)}x{int(n)}x{int(k)}_exe{int(i)}_main{file_extension}"
        )
        writefile(fn, output)


# ===============================================================================
def gen_jobfile(outdir, compiler, m, n, k, cpus_per_task, max_num_nodes=0):
    file_extension = get_file_extension_from_compiler(compiler)

    tprefix = f"tune_{int(m)}x{int(n)}x{int(k)}"
    all_exe_src = [fn.name for fn in outdir.glob(f"{tprefix}_*_main{file_extension}")]
    all_exe = sorted([fn.replace(f"_main{file_extension}", "") for fn in all_exe_src])
    if max_num_nodes > 0:
        num_nodes = min(len(all_exe), max_num_nodes)
    else:
        num_nodes = len(all_exe)
    time = "00:40:00"

    output = f"""\
#!/bin/bash -l
#SBATCH --nodes={int(num_nodes)}
#SBATCH --exclusive
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={int(cpus_per_task)}
#SBATCH --time={time}
#SBATCH --account=jiek61
#SBATCH --partition=dc-gpu
#SBATCH --cuda-mps
#SBATCH --gres=gpu:4
module purge
module add GCC/11.3.0
module add ParaStationMPI/5.8.0-1-mt
module add CUDA/11.7
module list
nvidia-smi
t1=$(date +%s)
"""
    # Compilation
    num_nodes_busy = 0
    for exe in all_exe:
        output += (
            f"srun --nodes=1 --ntasks=1 --ntasks-per-node=1"
            f" --cpus-per-task={cpus_per_task} --exact make -j {cpus_per_task} {exe} &\n"
        )
        num_nodes_busy += 1
        if num_nodes_busy == num_nodes:
            output += "wait\n"
            num_nodes_busy = 0

    output += "wait\n"
    output += "t2=$(date +%s)\n"
    output += "echo $((t2-t1)) seconds for compilation step\n\n"

    # Execution
    output += "t1=$(date +%s)\n"
    for exe in all_exe:
        output += (
            f"srun --nodes=1 --ntasks=1 --ntasks-per-node=1"
            f" --cpus-per-task=1 --exact ./{exe} > {exe}.log 2>&1 & \n"
        )
        num_nodes_busy += 1
        if num_nodes_busy == num_nodes:
            output += "wait\n"
            num_nodes_busy = 0

    output += "wait\n"
    output += "t2=$(date +%s)\n"
    output += "echo $((t2-t1)) seconds for execution step\n\n"

    # Winner
    output += "echo Over all winner:\n"
    output += f"grep WINNER {tprefix}_exe*.log | sort -n --field-separator='#' -k 2 | tail -n 1\n\n"

    # Cleaning
    output += "make realclean\n"

    fn = outdir / f"{tprefix}.job"
    writefile(fn, output)


# ===============================================================================
def gen_makefile(outdir, compiler, arch):
    file_extension = get_file_extension_from_compiler(compiler)

    # header
    output = ".SECONDARY:\n"
    output += f"vpath %{file_extension}../\n\n"
    output += ".PHONY: do_nothing build_all clean realclean\n\n"
    output += "do_nothing:\n\n"
    output += "clean:\n"
    output += "	rm -f *.o\n\n"
    output += "realclean: clean\n"
    output += "	rm -f *.cu\n\n"

    # target "build_all"
    all_exe_src = sorted(
        [fn.name for fn in outdir.glob(f"tune_*_main{file_extension}")]
    )
    build_targets = [fn.replace(f"_main{file_extension}", "") for fn in all_exe_src]
    output += f"build_all: {' '.join(build_targets)}\n\n"

    # compilation rule for helper-files: libsmm_acc_benchmark, acc_cuda/hip
    output += "libsmm_acc_benchmark.o : ../../libsmm_acc_benchmark.cpp\n"
    output += "acc.o :"
    if compiler == "nvcc":
        output += " ../../../cuda/acc_cuda.cpp\n\n"
    else:
        output += " ../../../hip/acc_hip.cpp\n\n"

    output += "libsmm_acc_benchmark.o acc.o :\n"
    if compiler == "nvcc":
        output += (
            "\tnvcc -O3 -D__TUNING -D__CUDA -arch="
            + str(arch)
            + " -w -c -o $@ -std=c++11 $<\n\n"
        )
    else:
        output += (
            "\thipcc -O3 -D__TUNING -D__HIP -w -munsafe-fp-atomics -c -o $@ $<\n\n"
        )

    # compilation rule for kernel files
    headers = " ".join([f"../{fn}" for fn in Path("../kernels").glob("*.h")])
    output += f"%.o : %{file_extension} {headers}\n"
    if compiler == "nvcc":
        output += f"	nvcc -O3 -D__TUNING -D__CUDA -arch={str(arch)} -w -c $<\n\n"
    else:
        output += "\thipcc -O3 -D__TUNING -D__HIP -w -munsafe-fp-atomics -c $<\n\n"

    # compilation rule for autotuning executables
    for exe_src in all_exe_src:
        parts = sorted(
            [
                fn.name
                for fn in outdir.glob(
                    exe_src.replace(f"_main{file_extension}", "_part*")
                )
            ]
        )
        deps = [exe_src, "libsmm_acc_benchmark.cpp", "acc.cpp"] + parts
        deps_obj = " ".join(
            [fn.replace(".cu", ".o").replace(".cpp", ".o") for fn in deps]
        )
        exe = exe_src.replace(f"_main{file_extension}", "")
        output += f"{exe} : {deps_obj}\n"
        if compiler == "nvcc":
            output += (
                f"	nvcc -O3 -D__CUDA -arch={str(arch)} -w -o $@ $^ -lcuda -lnvrtc\n\n"
            )
        else:
            rocm_path = os.getenv("ROCM_PATH", "/opt/rocm")
            output += f"\thipcc -O3 -D__HIP -w -munsafe-fp-atomics -o $@ $^ {rocm_path}/hip/lib/libamdhip64.so\n\n"

    # write Makefile
    writefile(outdir / "Makefile", output)


# ===============================================================================
def gen_collect(outdir: Path, triples):
    output = "#!/bin/bash\n"
    for m, n, k in triples:
        output += f"grep WINNER tune_{int(m)}x{int(n)}x{int(k)}_exe*.log  |  sort -n --field-separator='#' -k 2 | tail -n 1\n"
    output += "#EOF\n"
    fn = outdir / "collect_winners.sh"
    writefile(fn, output)
    fn.chmod(fn.stat().st_mode | stat.S_IEXEC)


# ===============================================================================
def writefile(fn: Path, content):
    if fn.exists() and fn.read_text() == content:
        return

    fn.write_text(content)


# ===============================================================================
def combinations(*sizes):
    return list(product(sizes, sizes, sizes))


# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Set up the autotuning of specified blocksizes. This script produces folders (tune_*x*x*)
        containing the code, Makefile and jobfiles for the autotuning of a given (m, n, k)-triplet.

        This script is part of the workflow for autotuning optimal libsmm_acc parameters.
        For more details, see README.md.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--params",
        metavar="parameters_GPU.json",
        default="../parameters/parameters_P100.json",
        type=Path,
        help="Parameter file that this autotuning should extend (pick the right GPU)",
    )
    parser.add_argument(
        "-b",
        "--compiler",
        metavar="compiler",
        default="nvcc",
        help="Compiler to use for compiling kernel code (Options: nvcc, hipcc)",
    )
    parser.add_argument(
        "-c",
        "--cpus_per_task",
        metavar="INT",
        default=128,
        type=int,
        help="Number of CPUs required per task",
    )
    parser.add_argument(
        "-n",
        "--nodes",
        metavar="INT",
        default=1,
        type=int,
        help="Maximum number of nodes an slurm allocation can get. 0: not a limiting factor"
        + "(choose this option if you can allocate jobs of 20-30 nodes without a problem.",
    )
    parser.add_argument(
        "blocksizes",
        metavar="BLOCKSIZE",
        nargs="+",
        type=str,
        help='Blocksize(s) to autotune. They can be provided as a list of integers (eg. "23",'
        + ' "4 5 13", "32 45") or provide a parameter file from which to read the blocksizes '
        + "to autotune, of the format parameters_GPU.json.",
    )

    args = parser.parse_args()

    # ==========
    # Verify option choice validity
    valid_compilers = ["nvcc", "hipcc"]
    assert (
        args.compiler in valid_compilers
    ), f"Compiler chosen ({args.compiler}) is not valid, please choose among: {valid_compilers}"

    # ==========
    # Blocksizes from parameter file or as list of integers
    blocksizes_from_param_file = False
    if args.blocksizes[0].isdigit():  # blocksizes is a sequence of strings
        args.blocksizes = [int(b) for b in args.blocksizes]
    else:  # blocksizes is a file name
        blocksizes_from_param_file = True
        args.blocksizes = args.blocksizes[0]

    # ==========
    main(
        args.params,
        args.compiler,
        args.cpus_per_task,
        args.nodes,
        args.blocksizes,
        blocksizes_from_param_file,
        Path("."),
    )
