/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include "parameters.h"
#include "parameters_utils.h"
#include "libsmm_acc.h"
#include "libsmm_acc_benchmark.h"
#include "smm_acc_kernels.h"

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

// MACRO HELPERS
#define STRINGIFY_NX(x) #x
#define STRINGIFY(x) STRINGIFY_NX(x)
#define CONCAT_NX(A, B) A ## B
#define CONCAT(A, B) CONCAT_NX(A, B)

// The macro ARCH_OPTION, when expanded, is a string literal containing the
// jit compiler option specifying the target architecture
#if defined(__CUDA) || defined(__HIP_PLATFORM_NVCC__)
#define ARCH_OPTION_NAME --gpu-architecture=compute_
#else
#define ARCH_OPTION_NAME --amdgpu-target=
#endif
#define ARCH_OPTION STRINGIFY(CONCAT(ARCH_OPTION_NAME, ARCH_NUMBER))


//===========================================================================
inline int launch_kernel_from_handle(ACC_DRV(function) const& kern_func, int nblks, int threads, ACC_DRV(stream) stream, void** args){

    ACC_DRV_CALL(
        LaunchJITKernel, (kern_func,      // kernel function,
                          nblks, 1, 1,    // grid dimension x, y, z
                          threads, 1, 1,  // block dimension x, y, z
                          0, stream,      // shared memory size and stream
                          args, NULL));   // arguments
    return 0;

}


//===========================================================================
inline void validate_kernel(ACC_DRV(function)& kern_func, ACC_DRV(stream) stream, int threads, int grouping, int m, int n, int k){

    libsmm_acc_benchmark_t* h;
    libsmm_acc_benchmark_init(&h, test, m, n, k);

    // Run the matrix-matrix multiplication on the CPU
    memset(h->mat_c, 0, h->n_c * m * n * sizeof(double));
    matInit(h->mat_a, h->n_a, m, k, 42);
    matInit(h->mat_b, h->n_b, k, n, 24);
    stackInit(h->stack, h->n_stack, h->n_c, h->mat_c, h->n_a, h->mat_a, h->n_b, h->mat_b, m, n, k);

    stackCalc(h->stack, h->n_stack, h->mat_c, h->mat_a, h->mat_b, m, n, k);
    double sumCPU = checkSum(h->mat_c, h->n_c, m, n);

    // Run the matrix-matrix multiplication kernel on the GPU
    ACC_API_CALL(Memcpy, (h->d_mat_a, h->mat_a, h->n_a * m * k * sizeof(double), ACC(MemcpyHostToDevice)));
    ACC_API_CALL(Memcpy, (h->d_mat_b, h->mat_b, h->n_b * k * n * sizeof(double), ACC(MemcpyHostToDevice)));
    ACC_API_CALL(Memcpy, (h->d_stack, h->stack, h->n_stack * 3 * sizeof(int), ACC(MemcpyHostToDevice)));
    ACC_API_CALL(Memset, (h->d_mat_c, 0, h->n_c * m * n * sizeof(double)));

    void *args[] = { &h->d_stack, &h->n_stack, &h->d_mat_a, &h->d_mat_b, &h->d_mat_c };
    int res = launch_kernel_from_handle(kern_func, ((h->n_stack + grouping - 1) / grouping), threads, stream, args);
    ACC_API_CALL(Memcpy, (h->mat_c, h->d_mat_c, h->n_c * m * n * sizeof(double), ACC(MemcpyDeviceToHost)));

    // Validate the kernel based on results
    double sumGPU =  checkSum(h->mat_c, h->n_c, m, n);
    if(sumGPU != sumCPU){
        printf("Kernel validation failed for kernel %ix%ix%i\nchecksum_diff: %g\nthreads: %i, grouping: %i\n", m, n, k, sumGPU-sumCPU, threads, grouping);
        exit(1);
    }
    libsmm_acc_benchmark_finalize(h);
}


//===========================================================================
inline void jit_kernel(ACC_DRV(function)& kern_func, libsmm_acc_algo algo, int tile_m, int tile_n, int w, int v, int threads, int grouping, int minblocks, int m, int n, int k){

    // Get the code and the lowered name corresponding the kernel to launch
    std::string kernel_code = smm_acc_common; // prepend include file content to code
    std::string kernel_name;
    switch(algo) {
        case 1:
            kernel_code += smm_acc_dnt_largeDB1;
            kernel_name = "smm_acc_dnt_largeDB1<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(w) + ", " + std::to_string(v) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 2:
            kernel_code += smm_acc_dnt_largeDB2;
            kernel_name = "smm_acc_dnt_largeDB2<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(w) + ", " + std::to_string(v) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 3:
            kernel_code += smm_acc_dnt_medium;
            kernel_name = "smm_acc_dnt_medium<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 4:
            kernel_code += smm_acc_dnt_small;
            kernel_name = "smm_acc_dnt_small<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 5:
            kernel_code += smm_acc_dnt_tiny;
            kernel_name = "smm_acc_dnt_tiny<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        default:
            printf("\nerror: algorithm number %i is not encoded.", algo);
            exit(1);
    }

    // Create JIT program
    ACC_RTC(Program) kernel_program;
    ACC_RTC_CALL(CreateProgram, (&kernel_program, kernel_code.c_str(), "smm_acc_kernel.cpp", 0, NULL, NULL));

    // Add lowered name
    ACC_RTC_CALL(AddNameExpression, (kernel_program, kernel_name.c_str()));

    // (JIT-)compile kernel program
#if defined(__CUDA) || defined(__HIP_PLATFORM_NVCC__)
    const char *compileOptions[] = {"-D__CUDA", "-w", ARCH_OPTION};
    size_t nOptions = 3;
#else
    const char *compileOptions[] = {"-D__HIP"};
    size_t nOptions = 1;
#endif
    ACC_RTC_CALL(CompileProgram, (kernel_program, nOptions, compileOptions));

    // Obtain PTX from the program.
    size_t codeSize;
    ACC_RTC_CALL(GetLowLevelCodeSize, (kernel_program, &codeSize));
    char *code = new char[codeSize];
    ACC_RTC_CALL(GetLowLevelCode, (kernel_program, code));

    // Get lowered name
    const char *lowered_kernel_name;
    ACC_RTC_CALL(GetLoweredName, (kernel_program, kernel_name.c_str(), &lowered_kernel_name));

    // Get pointer to kernel from PTX
    ACC_DRV(module) module;
    ACC_DRV_CALL(ModuleLoadDataEx, (&module, code, 0, 0, 0));
    delete[] code;
    ACC_DRV_CALL(ModuleGetFunction, (&kern_func, module, lowered_kernel_name));

    // Set shared memory configuration
#if defined(__CUDA) || defined(__HIP_PLATFORM_NVCC__)
    ACC_DRV_CALL(FuncSetSharedMemConfig, (kern_func, ACC_DRV(SharedMemBankSizeEightByte)));
#endif

    // Destroy program
    ACC_RTC_CALL(DestroyProgram, (&kernel_program));
}


void add_kernel_handle_to_jitted_kernels(ACC_DRV(function) kern_func, ACC_DRV(stream) stream, Triplet h_mnk, int& threads, int& grouping, bool& cpu_fallback){

    // Check whether autotuned parameters are given for this kernel, and if so, retrieve them
    if (ht.find(h_mnk) != ht.end()){

        // Retrieve launching parameters
        const KernelParameters params = ht.at(h_mnk);
        libsmm_acc_algo algo = libsmm_acc_algo(params[0]); // enum {largeDB1, largeDB2, medium, small, tiny}
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
int libsmm_acc_process_d(const int *param_stack, int stack_size, ACC_DRV(stream) stream, int m, int n, int k, const double *a_data, const double *b_data, double *c_data){

    ACC_DRV(function) kern_func = NULL;
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

    if(cpu_fallback){

        return -2; // fall back to CPU

    } else {

        // Retrieve kernel launching parameters
        kern_func = kernel_it->second.kernel_function;
        threads = kernel_it->second.threads;
        grouping = kernel_it->second.grouping;

        // Construct argument pointer list and launch kernel
        void *args[] = { &param_stack, &stack_size, &a_data, &b_data, &c_data };
        return launch_kernel_from_handle(kern_func, ((stack_size + grouping - 1) / grouping), threads, stream, args);

    }

}


//===========================================================================
extern "C" int libsmm_acc_process (const libsmm_acc_stack_descriptor_type *param_stack, int stack_size, int nparams, acc_data_t datatype, const void *a_data, const void *b_data, void *c_data, int m, int n, int k, int def_mnk, acc_stream_t *stream){
    if(def_mnk!=1)
        return -1; // inhomogeneous stacks not supported
    if(datatype==dbcsr_type_real_8) {
      if(m>MAX_BLOCK_DIM || n>MAX_BLOCK_DIM || k>MAX_BLOCK_DIM)
        return -1; // maximum size over any dimension
      else
        return (libsmm_acc_process_d ((const int *) param_stack, stack_size, *((ACC_DRV(stream) *) stream), m, n, k, (const double *) a_data, (const double *) b_data, (double *) c_data));
    }
    return -1; // datatype not supported
};


//===========================================================================
void jit_transpose_handle(ACC_DRV(function)& kern_func, int m, int n){

    // Create nvrtcProgram
    ACC_RTC(Program) kernel_program;
    std::string transpose_code = smm_acc_common + smm_acc_transpose;
    ACC_RTC_CALL(CreateProgram, (&kernel_program, transpose_code.c_str(), "transpose_kernel.cpp", 0, NULL, NULL));

    // Add lowered name
    std::string kernel_name = "transpose_d<" + std::to_string(m) + ", " + std::to_string(n) + ">";
    ACC_RTC_CALL(AddNameExpression, (kernel_program, kernel_name.c_str()));

    // (JIT-)compile
#if defined(__CUDA) || defined(__HIP_PLATFORM_NVCC__)
    const char *compileOptions[] = {"-D__CUDA", "-w", ARCH_OPTION};
    size_t nOptions = 3;
#else
    const char *compileOptions[] = {"-D__HIP"};
    size_t nOptions = 1;
#endif
    ACC_RTC_CALL(CompileProgram, (kernel_program, nOptions, compileOptions));

    // Obtain PTX from the program.
    size_t codeSize;
    ACC_RTC_CALL(GetLowLevelCodeSize, (kernel_program, &codeSize));
    char *code = new char[codeSize];
    ACC_RTC_CALL(GetLowLevelCode, (kernel_program, code));

    // Get lowered name
    const char *lowered_kernel_name;
    ACC_RTC_CALL(GetLoweredName, (kernel_program, kernel_name.c_str(), &lowered_kernel_name));

    // Get pointer to kernel from PTX
    ACC_DRV(module) module;
    ACC_DRV_CALL(ModuleLoadDataEx, (&module, code, 0, 0, 0));
    delete[] code;
    ACC_DRV_CALL(ModuleGetFunction, (&kern_func, module, lowered_kernel_name));

    // Set shared memory configuration
#if defined(__CUDA) || defined(__HIP_PLATFORM_NVCC__)
    ACC_DRV_CALL(FuncSetSharedMemConfig, (kern_func, ACC_DRV(SharedMemBankSizeEightByte)));
#endif

    // Destroy program
    ACC_RTC_CALL(DestroyProgram, (&kernel_program));
}


//===========================================================================
int libsmm_acc_transpose_d(const int *trs_stack, int offset, int nblks,
                           double *buffer, int m, int n, ACC_DRV(stream) stream) {

    ACC_DRV(function) kern_func;

    // Look up the kernel in the table of already JITed kernels
    Triplet h_mnk = { m, n, 0 };
    std::unordered_map<std::array<int, 3>, ACC_DRV(function)>::iterator kernel_it;

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
    const int* trs_stack_ = trs_stack + offset;
    void *args[] = { &trs_stack_, &buffer};

    return launch_kernel_from_handle(kern_func, nblks, 128, stream, args);

}


//===========================================================================
extern "C" int libsmm_acc_transpose (const int *trs_stack, int offset, int nblks, void *buffer, acc_data_t datatype, int m, int n, acc_stream_t* stream) {
    if(datatype != dbcsr_type_real_8)
        return 0; // transpose not needed
    if(m>MAX_BLOCK_DIM || n>MAX_BLOCK_DIM)
      return 0; // maximum size over any dimension
    return libsmm_acc_transpose_d(trs_stack, offset, nblks, (double *) buffer, m, n, *((ACC_DRV(stream) *) stream));
}

