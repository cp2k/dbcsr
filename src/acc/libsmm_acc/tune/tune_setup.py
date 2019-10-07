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
import json
import math
from glob import glob
from itertools import product
import argparse
from kernels.smm_acc_predict import (
    gpu_architectures,
    kernel_algorithm,
    params_dict_to_kernel,
    compatible_mnk,
)

sys.path.append("../")


# ===============================================================================
def main(
    param_fn,
    compiler,
    cpus_per_node,
    max_num_nodes,
    blocksizes,
    blocks_from_param_file,
    tune_dir,
):

    # Read existing parameters
    assert (
        os.path.basename(param_fn) in gpu_architectures.keys()
    ), "Cannot find GPU architecture for file " + os.path.basename(param_fn)
    arch_code = gpu_architectures[os.path.basename(param_fn)]
    with open("../kernels/gpu_properties.json") as f:
        gpu_properties = json.load(f)[arch_code]
    with open("../kernels/autotuning_properties.json") as f:
        autotuning_properties = json.load(f)
    with open(param_fn) as f:
        all_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]
    print("Reading parameters from %s" % param_fn)
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
        print("Reading parameters to autotune from %s" % blocksizes)
        triples = [(k.m, k.n, k.k) for k in all_kernels_ref if k.autotuned]
    else:
        assert len(set(blocksizes)) == len(blocksizes)
        blocksizes.sort()
        # Get (m, n, k) triplets to be autotuned
        triples = combinations(*blocksizes)
    print("Requested to autotune %d triplets" % len(triples))

    for (m, n, k) in triples:
        existing = [kern for kern in autotuned_kernels if kern.can_handle(m, n, k)]
        if existing:
            print(
                "Found existing autotuned parameter set for %dx%dx%d, skipping."
                % (m, n, k)
            )
            continue

        outdir = os.path.join(tune_dir, "tune_%dx%dx%d/" % (m, n, k))
        if os.path.exists(outdir):
            print("Directory %s exists already, skipping." % outdir)
            continue
        os.mkdir(outdir)
        gen_benchmark(outdir, gpu_properties, autotuning_properties, compiler, m, n, k)
        gen_jobfile(outdir, compiler, m, n, k, cpus_per_node, max_num_nodes)
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
            output.append("%s=%d" % (k, params[k]))

    for k in params.keys():
        if k not in order:
            output.append("%s=%d" % (k, params[k]))

    return "(" + ", ".join(output) + ")"


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
        kernel_algorithm[kernclass]
        for kernclass in kernel_algorithm.keys()
        if compatible_mnk(kernclass, m, n, k)
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
            includes.append("../../kernels/" + kern.include)
            launcher_codes.append(kern.launcher_code(compiler))
            launchers.append("launch_" + kern.name)
            kernel_descr.append(kernclass.__name__ + format_params(p))

    print("Found %d parameter sets for %dx%dx%d" % (len(launchers), m, n, k))
    if len(launchers) == 0:
        return

    # Compose the "include" line of the benchmark code
    incl_output = '#include "../../kernels/smm_acc_common.h"\n'
    for i in set(includes):
        incl_output += '#include "%s"\n' % i
    incl_output += "\n\n"

    # Compose the benchmark code
    # The benchmark is broken down in
    # - n_exe_files executables
    # - each executable is made of n_obj_files object files
    # - each object file is made up of launchers_per_obj launchers
    # - each launcher launches 1 GPU kernel with a certain set of kernel parameters
    # the hipcc compiler is very slow -> make a larger number of smaller executables
    max_launchers_per_exe = 10000 if compiler == "nvcc" else 100
    launchers_per_obj = 100 if compiler == "nvcc" else 10
    n_exe_files = int(len(launcher_codes) / max_launchers_per_exe) + 1
    launchers_per_exe = int(len(launcher_codes) / n_exe_files) + 1

    # Compose source code for each executable file
    for i in range(n_exe_files):
        chunk_a = i * launchers_per_exe
        chunk_b = min((i + 1) * launchers_per_exe, len(launcher_codes))
        n_obj_files = math.ceil((chunk_b - chunk_a) / launchers_per_obj)

        # Compose source code for each object file
        for j in range(n_obj_files):
            a = chunk_a + j * launchers_per_obj
            b = min(chunk_a + (j + 1) * launchers_per_obj, chunk_b)
            output = incl_output
            output += "\n\n".join(launcher_codes[a:b])
            fn = outdir + "/tune_%dx%dx%d_exe%d_part%d%s" % (
                m,
                n,
                k,
                i,
                j,
                file_extension,
            )
            writefile(fn, output)

        # Compose source code for "main" of executable file
        output = '#include "../../libsmm_acc_benchmark.h"\n\n'
        for j in range(chunk_b - chunk_a):
            output += (
                "int " + launchers[chunk_a + j] + "(int *param_stack, int stack_size, "
            )
            if compiler == "nvcc":
                output += "cudaStream_t stream, "
            else:
                output += "hipStream_t stream, "
            output += (
                "int m_max, int n_max, int k_max,"
                + " double *a_data, double *b_data, double *c_data);\n"
            )

        output += "\n"
        output += "int main(int argc, char** argv){\n"
        if compiler == "nvcc":
            output += (
                indent
                + "cudaError_t err = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);\n"
            )
            output += indent + "if(err != cudaSuccess) return(-1);\n"
        else:  # i.e. compiler = hipcc
            output += (
                indent
                + "hipError_t err = hipDeviceSetSharedMemConfig(hipSharedMemBankSizeEightByte);\n"
            )
            output += indent + "if(err != hipSuccess) return(-1);\n"
        output += indent + "libsmm_acc_benchmark_t* handle;\n"
        output += indent + "KernelLauncher launchers[%d];\n" % (chunk_b - chunk_a)
        output += indent + "char *kernel_descr[%d];\n" % (chunk_b - chunk_a)

        for j in range(chunk_b - chunk_a):
            output += indent + "launchers[%d]    = %s;\n" % (j, launchers[chunk_a + j])
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
        output += indent + "libsmm_acc_benchmark_finalize(handle);\n"
        output += indent + "return result;"
        output += "}\n"

        fn = outdir + "/tune_%dx%dx%d_exe%d_main%s" % (m, n, k, i, file_extension)
        writefile(fn, output)


# ===============================================================================
def gen_jobfile(outdir, compiler, m, n, k, cpus_per_node=12, max_num_nodes=0):

    file_extension = get_file_extension_from_compiler(compiler)

    t = "/tune_%dx%dx%d" % (m, n, k)
    all_exe_src = [
        os.path.basename(fn) for fn in glob(outdir + t + "_*_main" + file_extension)
    ]
    all_exe = sorted([fn.replace("_main" + file_extension, "") for fn in all_exe_src])
    if max_num_nodes > 0:
        num_nodes = min(len(all_exe), max_num_nodes)
    else:
        num_nodes = len(all_exe)
    if num_nodes < 3:
        time = "1:30:00"
    else:
        time = "0:30:00"

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
    else:  # i.e. compiler = hipcc
        output += "module load hip\n"
    output += "module list\n"
    output += "export CRAY_CUDA_MPS=1\n"
    output += "cd $SLURM_SUBMIT_DIR \n"
    output += "\n"
    output += "date\n"

    # Compilation
    num_nodes_busy = 0
    for exe in all_exe:
        output += (
            "srun --nodes=1 --bcast=/tmp/${USER} --ntasks=1 --ntasks-per-node=1 --cpus-per-task=%d make -j %d %s &\n"
            % (cpus_per_node, 2 * cpus_per_node, exe)
        )
        num_nodes_busy += 1
        if num_nodes_busy == num_nodes:
            output += "wait\n"
            num_nodes_busy = 0

    output += "wait\n"
    output += "date\n"
    output += "\n"

    # Execution
    for exe in all_exe:
        output += (
            "srun --nodes=1 --bcast=/tmp/${USER} --ntasks=1 --ntasks-per-node=1 --cpus-per-task=1 ./"
            + exe
            + " >"
            + exe
            + ".log 2>&1 & \n"
        )
        num_nodes_busy += 1
        if num_nodes_busy == num_nodes:
            output += "wait\n"
            num_nodes_busy = 0

    output += "wait\n"
    output += "date\n"
    output += "\n"

    # Winner
    output += "echo Over all winner:\n"
    output += (
        "grep WINNER ."
        + t
        + '_exe*.log  |  sort -n --field-separator="#" -k 2 | tail -n 1\n'
    )
    output += "\n"
    output += "#EOF\n"

    fn = outdir + t + ".job"
    writefile(fn, output)


# ===============================================================================
def gen_makefile(outdir, compiler, arch):

    file_extension = get_file_extension_from_compiler(compiler)

    # header
    output = ".SECONDARY:\n"
    output += "vpath %" + file_extension + "../\n\n"
    output += ".PHONY: do_nothing build_all \n\n"
    output += "do_nothing:\n\n"

    # target "build_all"
    all_exe_src = sorted(
        [os.path.basename(fn) for fn in glob(outdir + "/tune_*_main" + file_extension)]
    )
    build_targets = [fn.replace("_main" + file_extension, "") for fn in all_exe_src]
    output += "build_all: " + " ".join(build_targets) + "\n\n"

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
            "\tnvcc -O3 -D__CUDA -arch=" + str(arch) + " -w -c -o $@ -std=c++11 $<\n\n"
        )
    else:
        output += "\thipcc -O3 -D__HIP -w -c -o $@ $<\n\n"

    # compilation rule for kernel files
    headers = " ".join(["../" + fn for fn in glob("../kernels/*.h")])
    output += "%.o : %" + file_extension + " " + headers + "\n"
    if compiler == "nvcc":
        output += "\tnvcc -O3 -D__CUDA -arch=" + str(arch) + " -w -c $<\n\n"
    else:
        output += "\thipcc -O3 -D__HIP -w -c $<\n\n"

    # compilation rule for autotuning executables
    for exe_src in all_exe_src:
        absparts = sorted(
            glob(outdir + "/" + exe_src.replace("_main" + file_extension, "_part*"))
        )
        parts = [os.path.basename(fn) for fn in absparts]
        deps = [exe_src, "libsmm_acc_benchmark.cpp", "acc.cpp"] + parts
        deps_obj = " ".join(
            [fn.replace(".cu", ".o").replace(".cpp", ".o") for fn in deps]
        )
        exe = exe_src.replace("_main" + file_extension, "")
        output += exe + " : " + deps_obj + "\n"
        if compiler == "nvcc":
            output += (
                "\tnvcc -O3 -D__CUDA -arch="
                + str(arch)
                + " -w -o $@ $^ -lcuda -lnvrtc\n\n"
            )
        else:
            output += (
                "\thipcc -O3 -D__HIP -w -o $@ $^ /opt/rocm/hip/lib/libhiprtc.so\n\n"
            )

    # write Makefile
    writefile(outdir + "/Makefile", output)


# ===============================================================================
def gen_collect(outdir, triples):
    output = "#!/bin/bash\n"
    for (m, n, k) in triples:
        t = "/tune_%dx%dx%d" % (m, n, k)
        output += (
            "grep WINNER ."
            + t
            + '_exe*.log  |  sort -n --field-separator="#" -k 2 | tail -n 1\n'
        )
    output += "#EOF\n"
    fn = outdir + "/collect_winners.sh"
    writefile(fn, output)
    os.system("chmod +x " + fn)


# ===============================================================================
def writefile(fn, content):
    if os.path.exists(fn):
        with open(fn, "r") as f:
            old_content = f.read()
        if old_content == content:
            return

    with open(fn, "w") as f:
        f.write(content)


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
        For more details, see README.md#autotuning-procedure.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dir",
        metavar="tune_directory",
        default=".",
        type=str,
        help="Path from which to read already-existing tune-folders and write new tune-folders",
    )
    parser.add_argument(
        "-p",
        "--params",
        metavar="parameters_GPU.json",
        default="../parameters/parameters_P100.json",
        help="Parameter file that this autotuning should extend (pick the right GPU)",
    )
    parser.add_argument(
        "-b",
        "--compiler",
        metavar="compiler",
        default="nvcc",
        help="Compiler to use for compiling kernel code (Opions: nvcc, hipcc)",
    )
    parser.add_argument(
        "-c",
        "--cpus_per_node",
        metavar="INT",
        default=12,
        type=int,
        help="Number of CPUs per node",
    )
    parser.add_argument(
        "-n",
        "--nodes",
        metavar="INT",
        default=0,
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
    ), "Compiler chosen ({}) is not valid, please choose among: {}".format(
        args.compiler, valid_compilers
    )

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
        args.cpus_per_node,
        args.nodes,
        args.blocksizes,
        blocksizes_from_param_file,
        args.dir,
    )
