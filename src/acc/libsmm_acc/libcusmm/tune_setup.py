#!/usr/bin/env python
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
from glob import glob
from itertools import product
from optparse import OptionParser
from kernels.cusmm_predict import arch_number, kernel_algorithm, params_dict_to_kernel, compatible_mnk


# ===============================================================================
def main():
    usage = "Usage: tune.py <blocksize 1> ... <blocksize N>"
    parser = OptionParser(usage)
    parser.add_option(
        "-p", "--params", metavar="filename.json", default="parameters_P100.json", help="Default: %default"
    )

    (options, args) = parser.parse_args(sys.argv)
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    # Read existing parameters
    param_fn = options.params
    assert param_fn in arch_number.keys(), "Cannot find compute version for file " + param_fn
    arch = arch_number[param_fn]
    with open("kernels/gpu_properties.json") as f:
        gpu_properties = json.load(f)["sm_" + str(arch)]
    with open("kernels/autotuning_properties.json") as f:
        autotuning_properties = json.load(f)
    with open(param_fn) as f:
        all_kernels = [params_dict_to_kernel(**params) for params in json.load(f)]
    autotuned_kernels = [k for k in all_kernels if k.autotuned]
    predicted_kernels = [k for k in all_kernels if not k.autotuned]
    print(
        "Libcusmm: Found %d existing parameter sets, of which %d are autotuned and %d are predicted."
        % (len(all_kernels), len(autotuned_kernels), len(predicted_kernels))
    )

    # Get blocksizes to be autotuned
    blocksizes = [int(i) for i in args[1:]]
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
        gen_jobfile(outdir, m, n, k)
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

    # Get the kernels compatible with the given size:
    compatible_kernels = [
        kernel_algorithm[kernclass] for kernclass in kernel_algorithm.keys() if compatible_mnk(kernclass, m, n, k)
    ]

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

    incl_output = '#include "../kernels/cusmm_common.h"\n'
    for i in set(includes):
        incl_output += '#include "%s"\n' % i
    incl_output += "\n\n"

    max_launchers_per_exe = 10000
    launchers_per_obj = 100

    n_exe_files = int(len(launcher_codes) / max_launchers_per_exe) + 1
    launchers_per_exe = int(len(launcher_codes) / n_exe_files) + 1

    for i in range(n_exe_files):
        chunk_a = i * launchers_per_exe
        chunk_b = min((i + 1) * launchers_per_exe, len(launcher_codes))
        for j in range(int((chunk_b - chunk_a) / launchers_per_obj) + 1):
            output = incl_output
            a = chunk_a + j * launchers_per_obj
            b = min(chunk_a + (j + 1) * launchers_per_obj, chunk_b)
            output += "\n\n".join(launcher_codes[a:b])
            fn = outdir + "/tune_%dx%dx%d_exe%d_part%d.cu" % (m, n, k, i, j)
            writefile(fn, output)

        output = '#include "../libcusmm_benchmark.h"\n\n'
        for l in launchers:
            output += (
                "int "
                + l
                + "(int *param_stack, int stack_size, cudaStream_t stream, int m_max, int n_max, int k_max,"
                + " double *a_data, double *b_data, double *c_data);\n"
            )

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
def gen_jobfile(outdir, m, n, k):
    t = "/tune_%dx%dx%d" % (m, n, k)
    all_exe_src = [os.path.basename(fn) for fn in glob(outdir + t + "_*_main.cu")]
    all_exe = sorted([fn.replace("_main.cu", "") for fn in all_exe_src])

    output = "#!/bin/bash -l\n"
    output += "#SBATCH --nodes=%d\n" % len(all_exe)
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
    for exe in all_exe:
        output += (
            "srun --nodes=1 --bcast=/tmp/${USER} --ntasks=1 --ntasks-per-node=1 --cpus-per-task=12 make -j 24 %s &\n"
            % exe
        )
    output += "wait\n"
    output += "date\n"
    output += "\n"
    for exe in all_exe:
        output += (
            "srun --nodes=1 --bcast=/tmp/${USER} --ntasks=1 --ntasks-per-node=1 --cpus-per-task=1 ./"
            + exe
            + " >"
            + exe
            + ".log 2>&1 & \n"
        )
    output += "wait\n"
    output += "date\n"
    output += "\n"
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
main()
