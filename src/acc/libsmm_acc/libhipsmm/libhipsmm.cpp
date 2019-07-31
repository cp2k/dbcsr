/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include "../include/libsmm_acc.h"
#include "parameters.h"
#include "parameters_utils.h"
#include "libhipsmm.h"
#include "libhipsmm_benchmark.h"
#include "cusmm_kernels.h"

#include <hip/hiprtc.h>

#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <algorithm>
#include <array>
#include <iostream>

#if defined _OPENMP
#include <omp.h>
#endif

#define dbcsr_type_real_4     1
#define dbcsr_type_real_8     3
#define dbcsr_type_complex_4  5
#define dbcsr_type_complex_8  7


//===========================================================================
inline int launch_kernel_from_handle(hipFunction_t const& kern_func, int nblks, int threads, hipStream_t stream, void** args){
    // Get some hints here:
    // https://github.com/libocca/occa/blob/master/src/modes/hip/kernel.cpp

    HIP_SAFE_CALL(
        "hipModuleLaunchKernel",
        hipModuleLaunchKernel(kern_func,   // hipFunction_t
                       nblks, 1, 1,     // grid dimension x, y, z
                       threads, 1, 1,	// block dimension x, y, z
                       0, stream,       // shared memory size and stream
                       args, NULL));    // arguments
    return(0);

}


//===========================================================================
inline void validate_kernel(hipFunction_t& kern_func, hipStream_t stream, int threads, int grouping, int m, int n, int k){

    libcusmm_benchmark_t* h;
    libcusmm_benchmark_init(&h, test, m, n, k);

    // Run the matrix-matrix multiplication on the CPU
    memset(h->mat_c, 0, h->n_c * m * n * sizeof(double));
    matInit(h->mat_a, h->n_a, m, k, 42);
    matInit(h->mat_b, h->n_b, k, n, 24);
    stackInit(h->stack, h->n_stack, h->n_c, h->mat_c, h->n_a, h->mat_a, h->n_b, h->mat_b, m, n, k);

    stackCalc(h->stack, h->n_stack, h->mat_c, h->mat_a, h->mat_b, m, n, k);
    double sumCPU = checkSum(h->mat_c, h->n_c, m, n);

    // Run the matrix-matrix multiplication kernel on the GPU
    HIP_SAFE_CALL("hipMemcpy", hipMemcpy(h->d_mat_a, h->mat_a, h->n_a * m * k * sizeof(double), hipMemcpyHostToDevice));
    HIP_SAFE_CALL("hipMemcpy", hipMemcpy(h->d_mat_b, h->mat_b, h->n_b * k * n * sizeof(double), hipMemcpyHostToDevice));
    HIP_SAFE_CALL("hipMemcpy", hipMemcpy(h->d_stack, h->stack, h->n_stack * 3 * sizeof(int), hipMemcpyHostToDevice));
    HIP_SAFE_CALL("hipMemset", hipMemset(h->d_mat_c, 0, h->n_c * m * n * sizeof(double)));

    void *args[] = { &h->d_stack, &h->n_stack, &h->d_mat_a, &h->d_mat_b, &h->d_mat_c };
    int res = launch_kernel_from_handle(kern_func, ((h->n_stack + grouping - 1) / grouping), threads, stream, args);
    HIP_SAFE_CALL("hipMemcpy", hipMemcpy(h->mat_c, h->d_mat_c, h->n_c * m * n * sizeof(double), hipMemcpyDeviceToHost));

    // Validate the kernel based on results
    double sumGPU =  checkSum(h->mat_c, h->n_c, m, n);
    if(sumGPU != sumCPU){
        printf("Kernel validation failed for kernel %ix%ix%i\nchecksum_diff: %g\nthreads: %i, grouping: %i\n", m, n, k, sumGPU-sumCPU, threads, grouping);
        exit(1);
    }
    libcusmm_benchmark_finalize(h);
}


//===========================================================================
inline void jit_kernel(hipFunction_t& kern_func, libcusmm_algo algo, int tile_m, int tile_n, int w, int v, int threads, int grouping, int minblocks, int m, int n, int k){

    // Get the code and the lowered name corresponding the kernel to launch
    std::string kernel_code = cusmm_common; // prepend include file content to code
    std::string kernel_name;
    switch(algo) {
        case 1:
            kernel_code += cusmm_dnt_largeDB1;
            kernel_name = "cusmm_dnt_largeDB1<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(w) + ", " + std::to_string(v) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 2:
            kernel_code += cusmm_dnt_largeDB2; 
            kernel_name = "cusmm_dnt_largeDB2<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(w) + ", " + std::to_string(v) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 3:
            kernel_code += cusmm_dnt_medium; 
            kernel_name = "cusmm_dnt_medium<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 4:
            kernel_code += cusmm_dnt_small; 
            kernel_name = "cusmm_dnt_small<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 5:
            kernel_code += cusmm_dnt_tiny; 
            kernel_name = "cusmm_dnt_tiny<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        default:
            printf("\nerror: algorithm number %i is not encoded.", algo);
            exit(1);
    }

    // Create hiprtcProgram
    hiprtcProgram kernel_program;
    HIPRTC_SAFE_CALL("hiprtcCreateProgram", hiprtcCreateProgram(&kernel_program, kernel_code.c_str(), "smm_kernel.cpp", 0, NULL, NULL));

    // Add lowered name
    HIPRTC_SAFE_CALL("hiprtcAddNameExpression", hiprtcAddNameExpression(kernel_program, kernel_name.c_str()));

    // (JIT-)compile kernel program
#ifdef __HIP_PLATFORM_NVCC__
    const std::string arch_opt = "--gpu-architecture=compute_" + std::to_string(ARCH_NUMBER);
    const char *compileOptions[] = {"-w", arch_opt.c_str()};
    size_t nOptions = 2;
#else
    const char *compileOptions[] = {};
    size_t nOptions = 0;
#endif
    HIPRTC_SAFE_CALL("hiprtcCompileProgram", hiprtcCompileProgram(kernel_program, nOptions, compileOptions));

    // Obtain cdoe from the program.
    size_t codeSize;
    HIPRTC_SAFE_CALL("hiprtcGetCodeSize", hiprtcGetCodeSize(kernel_program, &codeSize));
    char *code = new char[codeSize];
    HIPRTC_SAFE_CALL("hiprtcGetCode", hiprtcGetCode(kernel_program, code));

    // Get lowered name
    const char *lowered_kernel_name;
    HIPRTC_SAFE_CALL("hiprtcGetLoweredName", hiprtcGetLoweredName(kernel_program, kernel_name.c_str(), &lowered_kernel_name));

    // Get pointer to kernel from code
    hipModule_t module;
    HIP_SAFE_CALL("hipModuleLoadData", hipModuleLoadData(&module, code));
    delete[] code;
    HIP_SAFE_CALL("hipModuleGetFunction", hipModuleGetFunction(&kern_func, module, lowered_kernel_name));

    // Set shared memory configuration
    //HIP_SAFE_CALL("cuFuncSetSharedMemConfig", cuFuncSetSharedMemConfig(kern_func, CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));

    // Destroy program
    HIPRTC_SAFE_CALL("hiprtcDestroyProgram", hiprtcDestroyProgram(&kernel_program));
}


void add_kernel_handle_to_jitted_kernels(hipFunction_t kern_func, hipStream_t stream, Triplet h_mnk, int& threads, int& grouping, bool& cpu_fallback){

    // Check whether autotuned parameters are given for this kernel, and if so, retrieve them
    if (ht.find(h_mnk) != ht.end()){

        // Retrieve launching parameters
        const KernelParameters params = ht.at(h_mnk);
        libcusmm_algo algo = libcusmm_algo(params[0]); // enum {largeDB1, largeDB2, medium, small, tiny}
        int tile_m = params[1];
        int tile_n = params[2];
        int w = params[3];
        int v = params[4];
        threads = params[5];
        grouping = params[6];
        int minblocks =  params[7];

        // JIT and validate the kernel
        jit_kernel(kern_func, algo, tile_m, tile_n, w, v, threads, grouping, minblocks, h_mnk[0], h_mnk[1], h_mnk[2]);
        validate_kernel(kern_func, stream, threads, grouping, h_mnk[0], h_mnk[1], h_mnk[2]);

        // Store the handle to the JIT-ed kernel
        kernel_handles.emplace(h_mnk, kernel_launcher(kern_func, threads, grouping));

    } else { // there exist no autotuned parameters for this (m, n, k)-triplet, fall back to CPU

        cpu_fallback = true;

    }

}


//===========================================================================
int libcusmm_process_d(int *param_stack, int stack_size, hipStream_t stream, int m, int n, int k, double *a_data, double *b_data, double *c_data){

    hipFunction_t kern_func = NULL;
    int threads, grouping;
    Triplet h_mnk = { m, n, k };
    static bool cpu_fallback = false;
    std::unordered_map<std::array<int, 3>, kernel_launcher>::iterator kernel_it;

#if defined _OPENMP
#pragma omp critical (jit_multiplication)
{
#endif

    // Look up the kernel in the table of already JITed kernels
    kernel_it = kernel_handles.find(h_mnk);
    if (kernel_it == kernel_handles.end()){  // the kernel has not been JIT-ed yet

        add_kernel_handle_to_jitted_kernels(kern_func, stream, h_mnk, threads, grouping, cpu_fallback);
        kernel_it = kernel_handles.find(h_mnk);

    }  // now the kernel has been jitted

#if defined _OPENMP
}
#endif

    if(cpu_fallback)
        return -2; // fall back to CPU

    // Retrieve kernel launching parameters
    kern_func = kernel_it->second.kernel_function;
    threads = kernel_it->second.threads;
    grouping = kernel_it->second.grouping;

    // Construct argument pointer list and launch kernel
    void *args[] = { &param_stack, &stack_size, &a_data, &b_data, &c_data };
    return launch_kernel_from_handle(kern_func, ((stack_size + grouping - 1) / grouping), threads, stream, args);

}


//===========================================================================
extern "C" int libsmm_acc_process (void *param_stack, int stack_size, int nparams, int datatype, void *a_data, void *b_data, void *c_data, int m, int n, int k, int def_mnk, void *stream){
    if(def_mnk!=1)
        return(-1); // inhomogeneous stacks not supported
    if(datatype==dbcsr_type_real_8) {
      if(m>MAX_BLOCK_DIM || n>MAX_BLOCK_DIM || k>MAX_BLOCK_DIM)
	return(-1); // maximum size over any dimention
      else
        return (libcusmm_process_d ((int *) param_stack, stack_size, *((hipStream_t *) stream), m, n, k, (double *) a_data, (double *) b_data, (double *) c_data));
    }
    return(-1); // datatype not supported
};


//===========================================================================
void jit_transpose_handle(hipFunction_t & kern_func, int m, int n){

    // Create hiprtcProgram
    hiprtcProgram kernel_program;
    std::string transpose_code = cusmm_common + cusmm_transpose; 
    HIPRTC_SAFE_CALL("hiprtcCreateProgram", hiprtcCreateProgram(&kernel_program, transpose_code.c_str(), "transpose_kernel.cpp", 0, NULL, NULL));

    // Add lowered name
    std::string kernel_name = "transpose_d<" + std::to_string(m) + ", " + std::to_string(n) + ">";
    HIPRTC_SAFE_CALL("hiprtcAddNameExpression", hiprtcAddNameExpression(kernel_program, kernel_name.c_str()));

    // (JIT-)compile
#ifdef __HIP_PLATFORM_NVCC__
    const std::string arch_opt = "--gpu-architecture=compute_" + std::to_string(ARCH_NUMBER);
    size_t nOptions = 2;
    const char *compileOptions[] = {"-w", arch_opt.c_str()};
#else
    size_t nOptions = 0;
    const char *compileOptions[] = {};
#endif
    HIPRTC_SAFE_CALL("hiprtcCompileProgram", hiprtcCompileProgram(kernel_program, nOptions, compileOptions));

    // Obtain code from the program.
    size_t codeSize;
    HIPRTC_SAFE_CALL("hiprtcGetCodeSize", hiprtcGetCodeSize(kernel_program, &codeSize));
    char *code = new char[codeSize];
    HIPRTC_SAFE_CALL("hiprtcGetCode", hiprtcGetCode(kernel_program, code));

    // Get lowered name
    const char *lowered_kernel_name;
    HIPRTC_SAFE_CALL("hiprtcGetLoweredName", hiprtcGetLoweredName(kernel_program, kernel_name.c_str(), &lowered_kernel_name));

    // Get pointer to kernel from code
    hipModule_t module;
    HIP_SAFE_CALL("hipModuleLoadData", hipModuleLoadData(&module, code));
    delete[] code;
    HIP_SAFE_CALL("hipModuleGetFunction", hipModuleGetFunction(&kern_func, module, lowered_kernel_name));

    // Set shared memory configuration
    //HIP_SAFE_CALL("cuFuncSetSharedMemConfig", cuFuncSetSharedMemConfig(kern_func, CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));

    // Destroy program
    HIPRTC_SAFE_CALL("hiprtcDestroyProgram", hiprtcDestroyProgram(&kernel_program));
}


//===========================================================================
int libcusmm_transpose_d(int *trs_stack, int offset, int nblks,
                         double *buffer, int m, int n, hipStream_t stream) {

    hipFunction_t kern_func;

    // Look up the kernel in the table of already JITed kernels
    Triplet h_mnk = { m, n, 0 };
    std::unordered_map<std::array<int, 3>, hipFunction_t>::iterator kernel_it;

#if defined _OPENMP
#pragma omp critical (jit_transpose)
{
#endif

    kernel_it = transpose_handles.find(h_mnk);
    if(kernel_it == transpose_handles.end()){  // the kernel has not been JIT-ed yet

        // JIT and store a kernel for this transposition
        jit_transpose_handle(kern_func, m, n);
        transpose_handles.emplace(h_mnk, kern_func);
        kernel_it = transpose_handles.find(h_mnk);

    }

#if defined _OPENMP
}
#endif

    // Construct argument pointer list and launch function
    kern_func = kernel_it->second; // retrieve handle
    int* trs_stack_ = trs_stack + offset; 
    void *args[] = { &trs_stack_, &buffer};

    return launch_kernel_from_handle(kern_func, nblks, 128, stream, args);

}


//===========================================================================
extern "C" int libsmm_acc_transpose (void *trs_stack, int offset, int nblks, void *buffer,int datatype, int m, int n, void* stream) {
    hipStream_t* custream = (hipStream_t*) stream;
    if(datatype != dbcsr_type_real_8)
        return 0; //transpose not needed
    if(m>MAX_BLOCK_DIM || n>MAX_BLOCK_DIM)
      return 0; // maximum size over any dimention
    return libcusmm_transpose_d((int *) trs_stack, offset, nblks, (double *) buffer, m, n, *((hipStream_t *) stream));
}


//===========================================================================
extern "C" int libsmm_acc_libcusmm_is_thread_safe () {
#if defined _OPENMP
    return 1;  // i.e. true, libcusmm is threaded
#else
    return 0;  // i.e. false, libcusmm is not threaded
#endif
}
