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

import os
import json
from glob import glob
from itertools import product
import argparse
from kernels.cusmm_predict import arch_number, kernel_algorithm, params_dict_to_kernel, compatible_mnk


# ===============================================================================
def main(param_fn, cpus_per_node, max_num_nodes, blocksizes):

    # Read existing parameters
    assert param_fn in arch_number.keys(), "Cannot find compute version for file " + param_fn
    arch = arch_number[param_fn]
    with open("kernels/gpu_properties.json") as f:
        gpu_properties = json.load(f)["sm_" + str(arch)]
    with open("kernels/autotuning_properties.json") as f:
        autotuning_properties = json.load(f)
    with open(param_fn) as f:
        all_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]
    print("Reading parameters from %s" % param_fn)
    autotuned_kernels = [k for k in all_kernels if k.autotuned]
    predicted_kernels = [k for k in all_kernels if not k.autotuned]
    print("Libcusmm: Found %d existing parameter sets, of which %d are autotuned and %d are predicted." %
          (len(all_kernels), len(autotuned_kernels), len(predicted_kernels)))

    # Get blocksizes to be autotuned
    assert len(set(blocksizes)) == len(blocksizes)
    blocksizes.sort()

    # Get (m, n, k) triplets to be autotuned
    triples = combinations(*blocksizes)

    for (m, n, k) in triples:
        existing = [kern for kern in autotuned_kernels if kern.can_handle(m, n, k)]
        if existing:
            print("Found existing autotuned parameter set for %dx%dx%d, skipping." % (m, n, k))
            continue

        outdir = "tune_%dx%dx%d/" % (m, n, k)
        if os.path.exists(outdir):
            print("Directory %s exists already, skipping." % outdir)
            continue
        os.mkdir(outdir)
        gen_benchmark(outdir, gpu_properties, autotuning_properties, m, n, k)
        gen_jobfile(outdir, m, n, k, cpus_per_node, max_num_nodes)
        gen_makefile(outdir, arch)


# ===============================================================================
def format_params(params):
    output = []
    order = ["m", "n", "k", "tile_m", "tile_n", "w", "v", "split_thread", "threads", "blockdim", "grouping"]
    for k in order:
        if k in params.keys():
            output.append("%s=%d" % (k, params[k]))

    for k in params.keys():
        if k not in order:
            output.append("%s=%d" % (k, params[k]))

    return "(" + ", ".join(output) + ")"


# ===============================================================================
def gen_benchmark(outdir, gpu_properties, autotuning_properties, m, n, k):
    includes = []
    launcher_codes = []
    launchers = []
    kernel_descr = []

    # Get the kernel algorithms compatible with the given size:
    compatible_kernels = [
        kernel_algorithm[kernclass] for kernclass in kernel_algorithm.keys() if compatible_mnk(kernclass, m, n, k)
    ]

    # Get the parameter sets to measure for this (m,n,k)
    for kernclass in compatible_kernels:
        params = kernclass.promising_parameters(m, n, k, gpu_properties, autotuning_properties)
        if params == 0:
            continue

        for p in params:
            kern = kernclass(**p, source="autotuning_candidate", perf=0)
            includes.append("../kernels/" + kern.include)
            launcher_codes.append(kern.launcher_code)
            launchers.append("launch_" + kern.name)
            kernel_descr.append(kernclass.__name__ + format_params(p))

    print("Found %d parameter sets for %dx%dx%d" % (len(launchers), m, n, k))
    if len(launchers) == 0:
        return

    # Compose the "include" line of the benchmark code
    incl_output = '#include "../kernels/cusmm_common.h"\n'
    for i in set(includes):
        incl_output += '#include "%s"\n' % i
    incl_output += "\n\n"
    max_launchers_per_exe = 10000
    # Compose the benchmark code
    launchers_per_obj = 100
    n_exe_files = int(len(launcher_codes) / max_launchers_per_exe) + 1
    launchers_per_exe = int(len(launcher_codes) / n_exe_files) + 1

    # Compose source code for each executable file
    for i in range(n_exe_files):
        chunk_a = i * launchers_per_exe
        chunk_b = min((i + 1) * launchers_per_exe, len(launcher_codes))
        n_obj_files = int((chunk_b - chunk_a) / launchers_per_obj) + 1

		# Compose source code for each object file
        for j in range(n_obj_files):
            a = chunk_a + j * launchers_per_obj
            b = min(chunk_a + (j + 1) * launchers_per_obj, chunk_b)
            output = incl_output
            output += "\n\n".join(launcher_codes[a:b])
            fn = outdir + "/tune_%dx%dx%d_exe%d_part%d.cu" % (m, n, k, i, j)
            writefile(fn, output)

		# Compose source code for "main" of executable file
        output = '#include "../libcusmm_benchmark.h"\n\n'
        for l in launchers:
            output += (
                "int " + l + "(int *param_stack, int stack_size, cudaStream_t stream, int m_max, int n_max, int k_max,"
                + " double *a_data, double *b_data, double *c_data);\n")

        output += "\n"
        output += "int main(int argc, char** argv){\n"
        output += "libcusmm_benchmark_t* handle;\n"
        output += "KernelLauncher launchers[%d];\n" % (chunk_b - chunk_a)
        output += "char *kernel_descr[%d];\n" % (chunk_b - chunk_a)

        for j in range(chunk_b - chunk_a):
            output += "launchers[%d]    = %s;\n" % (j, launchers[chunk_a + j])
            output += 'kernel_descr[%d] = (char *) "%s";\n' % (j, kernel_descr[chunk_a + j])
        output += "libcusmm_benchmark_init(&handle, tune, %d, %d, %d);\n" % (m, n, k)
        output += "int result = libcusmm_benchmark(handle, %d, %d, %d, %d, launchers, kernel_descr);\n" % (
            m,
            n,
            k,
            chunk_b - chunk_a,
        )
        output += "libcusmm_benchmark_finalize(handle);\n"
        output += "return result;"
        output += "}\n"

        fn = outdir + "/tune_%dx%dx%d_exe%d_main.cu" % (m, n, k, i)
        writefile(fn, output)


# ===============================================================================
def gen_jobfile(outdir, m, n, k, cpus_per_node=12, max_num_nodes=0):
    t = "/tune_%dx%dx%d" % (m, n, k)
    all_exe_src = [os.path.basename(fn) for fn in glob(outdir + t + "_*_main.cu")]
    all_exe = sorted([fn.replace("_main.cu", "") for fn in all_exe_src])
    if max_num_nodes > 0:
        num_nodes = min(len(all_exe), max_num_nodes)
    else:
        num_nodes = len(all_exe)

    output = "#!/bin/bash -l\n"
    output += "#SBATCH --nodes=%d\n" % num_nodes
    output += "#SBATCH --ntasks-per-core=1\n"
    output += "#SBATCH --ntasks-per-node=1\n"
    output += "#SBATCH --cpus-per-task=" + "%d\n" % cpus_per_node
    if num_nodes < 3:
        output += "#SBATCH --time=1:30:00\n"
    else:
        output += "#SBATCH --time=0:30:00\n"
    output += "#SBATCH --account=s238\n"
    output += "#SBATCH --partition=normal\n"
    output += "#SBATCH --constraint=gpu\n"
    output += "\n"
    output += "source ${MODULESHOME}/init/sh;\n"
    output += "module load daint-gpu\n"
    output += "module unload PrgEnv-cray\n"
    output += "module load PrgEnv-gnu/6.0.3\n"
    output += "module load cudatoolkit/8.0.54_2.2.8_ga620558-2.1\n"
    output += "module list\n"
    output += "export CRAY_CUDA_MPS=1\n"
    output += "cd $SLURM_SUBMIT_DIR \n"
    output += "\n"
    output += "date\n"

    # Compilation
    num_nodes_busy = 0
    for exe in all_exe:
        output += (
            "srun --nodes=1 --bcast=/tmp/${USER} --ntasks=1 --ntasks-per-node=1 --cpus-per-task=%d make -j %d %s &\n" %
            (cpus_per_node, 2*cpus_per_node, exe))
        num_nodes_busy += 1
        if num_nodes_busy == num_nodes:
            output += "wait\n"
            num_nodes_busy = 0

    output += "wait\n"
    output += "date\n"
    output += "\n"

    # Execution
    for exe in all_exe:
        output += ("srun --nodes=1 --bcast=/tmp/${USER} --ntasks=1 --ntasks-per-node=1 --cpus-per-task=1 ./" + exe +
                   " >" + exe + ".log 2>&1 & \n")
        num_nodes_busy += 1
        if num_nodes_busy == num_nodes:
            output += "wait\n"
            num_nodes_busy = 0

    output += "wait\n"
    output += "date\n"
    output += "\n"

    # Winner
    output += "echo Over all winner:\n"
    output += "grep WINNER ." + t + '_exe*.log  |  sort -n --field-separator="#" -k 2 | tail -n 1\n'
    output += "\n"
    output += "#EOF\n"

    fn = outdir + t + ".job"
    writefile(fn, output)


# ===============================================================================
def gen_makefile(outdir, arch):

    output = ".SECONDARY:\n"
    output += "vpath %.cu ../\n\n"
    all_exe_src = sorted([os.path.basename(fn) for fn in glob(outdir + "/tune_*_main.cu")])
    build_targets = [fn.replace("_main.cu", "") for fn in all_exe_src]

    output += ".PHONY: do_nothing build_all \n\n"
    output += "do_nothing:\n\n"
    output += "build_all: " + " ".join(build_targets) + "\n\n"

    output += "libcusmm_benchmark.o : libcusmm_benchmark.cu\n"
    output += "\tnvcc -O3 -arch=sm_" + str(arch) + " -w -c -std=c++11 $<\n\n"

    headers = " ".join(["." + fn for fn in glob("./kernels/*.h")])
    output += "%.o : %.cu " + headers + "\n"
    output += "\tnvcc -O3 -arch=sm_" + str(arch) + " -w -c $<\n\n"

    for exe_src in all_exe_src:
        absparts = sorted(glob(outdir + "/" + exe_src.replace("_main.cu", "_part*")))
        parts = [os.path.basename(fn) for fn in absparts]
        deps = [exe_src, "libcusmm_benchmark.cu"] + parts
        deps_obj = " ".join([fn.replace(".cu", ".o") for fn in deps])
        exe = exe_src.replace("_main.cu", "")
        output += exe + " : " + deps_obj + "\n"
        output += "\tnvcc -O3 -arch=sm_" + str(arch) + " -w -o $@ $^ -L $(CUDA_PATH) -lcuda\n\n"

    writefile(outdir + "/Makefile", output)


# ===============================================================================
def gen_collect(outdir, triples):
    output = "#!/bin/bash\n"
    for (m, n, k) in triples:
        t = "/tune_%dx%dx%d" % (m, n, k)
        output += "grep WINNER ." + t + '_exe*.log  |  sort -n --field-separator="#" -k 2 | tail -n 1\n'
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
        Usage: tune.py <blocksize 1> ... <blocksize N> --params parameters_GPU.json

        This script is part of the workflow for autotuning optimal libcusmm parameters.
        For more details, see README.md#autotuning-procedure.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-p", "--params", metavar="parameters_GPU.json", default="parameters_P100.json", help="Parameter file to extend by this autotuning (pick the right GPU)")
    parser.add_argument(
        "-c", "--cpus_per_node", metavar="INT", default=12, type=int, help="Maximum number of nodes an slurm allocation can get")
    parser.add_argument(
        "-n", "--nodes", metavar="INT", default=0, type=int, help="Maximum number of nodes an slurm allocation can get. 0: not a limiting factor (choose this option if you can allocate jobs of 20-30 nodes without a problem.")
    parser.add_argument('blocksizes', metavar="BLOCKSIZE", nargs='+', type=int, help='Blocksize to autotune')

    args = parser.parse_args()
    main(args.params, args.cpus_per_node, args.nodes, args.blocksizes)
