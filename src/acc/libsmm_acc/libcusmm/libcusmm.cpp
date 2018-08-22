/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2018  CP2K developers group                         *
 *****************************************************************************/
#include "../include/libsmm_acc.h"
#include "parameters.h"
#include "parameters_utils.h"
#include "libcusmm.h"
#include "libcusmm_benchmark.h"
#include "cusmm_kernels.h"

#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <algorithm>
#include <array>
#include <iostream>

#define dbcsr_type_real_4     1
#define dbcsr_type_real_8     3
#define dbcsr_type_complex_4  5
#define dbcsr_type_complex_8  7
#define MAX_BLOCK_DIM         80


//===========================================================================
inline int launch_kernel_from_handle(CUfunction const& kern_func, int nblks, int threads, CUstream stream, void** args){

    CUDA_SAFE_CALL(
        "cuLaunchKernel",
        cuLaunchKernel(kern_func,       // CUfunction
                       nblks, 1, 1,     // grid dimension x, y, z
                       threads, 1, 1,	// block dimension x, y, z
                       0, stream,       // shared memory size and stream
                       args, NULL));    // arguments
    return(0);

}


//===========================================================================
inline void validate_kernel(CUfunction& kern_func, CUstream stream, int threads, int grouping, int m, int n, int k){

    libcusmm_benchmark_t* h;
    libcusmm_benchmark_init(&h, false, m, n, k);

    // Run the matrix-matrix multiplication on the CPU
    memset(h->mat_c, 0, h->n_c * m * n * sizeof(double));
    matInit(h->mat_a, h->n_a, m, k, 42);
    matInit(h->mat_b, h->n_b, k, n, 24);
    stackInit(h->stack, h->n_stack, h->n_c, h->mat_c, h->n_a, h->mat_a, h->n_b, h->mat_b, m, n, k);

    stackCalc(h->stack, h->n_stack, h->mat_c, h->mat_a, h->mat_b, m, n, k);
    double sumCPU = checkSum(h->mat_c, h->n_c, m, n);

    // Run the matrix-matrix multiplication kernel on the GPU
    cudaMemcpy(h->d_mat_a, h->mat_a, h->n_a * m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_mat_b, h->mat_b, h->n_b * k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_stack, h->stack, h->n_stack * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(h->d_mat_c, 0, h->n_c * m * n * sizeof(double));

    void *args[] = { &h->d_stack, &h->n_stack, &h->d_mat_a, &h->d_mat_b, &h->d_mat_c };
    int res = launch_kernel_from_handle(kern_func, ((h->n_stack + grouping - 1) / grouping), threads, stream, args);
    cudaMemcpy(h->mat_c, h->d_mat_c, h->n_c * m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess){
        printf("validate_kernel: cuda_error: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }

    // Validate the kernel based on results
    double sumGPU =  checkSum(h->mat_c, h->n_c, m, n);
    if(sumGPU != sumCPU){
        printf("Kernel validation failed for kernel %ix%ix%i\nchecksum_diff: %g\nthreads: %i, grouping: %i\n", m, n, k, sumGPU-sumCPU, threads, grouping);
        exit(1);
    }
    libcusmm_benchmark_finalize(h);
}


//===========================================================================
inline void jit_kernel(CUfunction& kern_func, libcusmm_algo algo, int tile_m, int tile_n, int w, int v, int threads, int grouping, int minblocks, int m, int n, int k){

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

    // Create nvrtcProgram
    nvrtcProgram kernel_program;
    NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&kernel_program, kernel_code.c_str(), "smm_kernel.cu", 0, NULL, NULL));

    // Add lowered name
    NVRTC_SAFE_CALL("nvrtcAddNameExpression", nvrtcAddNameExpression(kernel_program, kernel_name.c_str()));

    // (JIT-)compile kernel program
    const std::string arch_opt = "--gpu-architecture=compute_" + std::to_string(ARCH_NUMBER);
    const char *compileOptions[] = {"-w", arch_opt.c_str()};
    size_t nOptions = 2;
    NVRTC_SAFE_CALL("nvrtcCompileProgram", nvrtcCompileProgram(kernel_program, nOptions, compileOptions));

    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL("nvrtcGetPTXsize", nvrtcGetPTXSize(kernel_program, &ptxSize));
    char *ptx = new char[ptxSize];
    NVRTC_SAFE_CALL("nvrtcGetPTX", nvrtcGetPTX(kernel_program, ptx));

    // Get lowered name
    const char *lowered_kernel_name;
    NVRTC_SAFE_CALL("nvrtcGetLoweredName", nvrtcGetLoweredName(kernel_program, kernel_name.c_str(), &lowered_kernel_name));

    // Get pointer to kernel from PTX
    CUmodule module;
    CUDA_SAFE_CALL("cuModuleLoadDataEx", cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    delete[] ptx; 
    CUDA_SAFE_CALL("cuModuleGetFunction", cuModuleGetFunction(&kern_func, module, lowered_kernel_name));

    // Set shared memory configuration
    CUDA_SAFE_CALL("cuFuncSetSharedMemConfig", cuFuncSetSharedMemConfig(kern_func, CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));

    // Destroy program
    NVRTC_SAFE_CALL("nvrtcDestroyProgram", nvrtcDestroyProgram(&kernel_program));
}


//===========================================================================
int libcusmm_process_d(int *param_stack, int stack_size, CUstream stream, int m, int n, int k, double *a_data, double *b_data, double *c_data){

    CUfunction kern_func;
    int threads, grouping; 

    // Look up the kernel in the table of already JITed kernels
    Triplet h_mnk = { m, n, k };
    auto kernel_it = kernel_handles.find(h_mnk);
    if (kernel_it != kernel_handles.end()){  // the kernel has already been JITed

        // Retrieve kernel launching parameters
        kern_func = kernel_it->second.kernel_function; 
        threads = kernel_it->second.threads; 
        grouping = kernel_it->second.grouping;

    } else {	// the kernel has not been JIT-ed yet

	// Check whether autotuned parameters are given for this kernel, and if so, retrieve them 
        auto params_it = ht.find(h_mnk);
	if (params_it != ht.end()){

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
            jit_kernel(kern_func, algo, tile_m, tile_n, w, v, threads, grouping, minblocks, m, n, k);
            validate_kernel(kern_func, stream, threads, grouping, m, n, k);

            // Store the handle to the JIT-ed kernel
            kernel_handles.emplace(h_mnk, kernel_launcher(kern_func, threads, grouping));

        } else { // there exist no autotuned parameters for this (m, n, k)-triplet

            return -2; // fall back to CPU

        }

    }

    // Construct argument pointer list and launch kernel
    void *args[] = { &param_stack, &stack_size, &a_data, &b_data, &c_data };
    return launch_kernel_from_handle(kern_func, ((stack_size + grouping - 1) / grouping), threads, stream, args);

}


//===========================================================================
extern "C" int libsmm_acc_process (void *param_stack, int stack_size, int nparams, int datatype, void *a_data, void *b_data, void *c_data, int m, int n, int k, int def_mnk, void *stream){
    if(def_mnk!=1)
        return(-1); // inhomogenous stacks not supported
    if(datatype==dbcsr_type_real_8) {
      if(m>MAX_BLOCK_DIM || n>MAX_BLOCK_DIM || k>MAX_BLOCK_DIM)
	return(-1); // maximum size over any dimention
      else
        return (libcusmm_process_d ((int *) param_stack, stack_size, *((CUstream *) stream), m, n, k, (double *) a_data, (double *) b_data, (double *) c_data));
    }
    return(-1); // datatype not supported
};


//===========================================================================
void jit_transpose_handle(CUfunction& kern_func, int m, int n){

    // Create nvrtcProgram
    nvrtcProgram kernel_program;
    std::string transpose_code = cusmm_common + cusmm_transpose; 
    NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&kernel_program, transpose_code.c_str(), "transpose_kernel.cu", 0, NULL, NULL));

    // Add lowered name
    std::string kernel_name = "transpose_d<" + std::to_string(m) + ", " + std::to_string(n) + ">";
    NVRTC_SAFE_CALL("nvrtcAddNameExpression", nvrtcAddNameExpression(kernel_program, kernel_name.c_str()));

    // (JIT-)compile
    size_t nOptions = 2;
    const std::string arch_opt = "--gpu-architecture=compute_" + std::to_string(ARCH_NUMBER);
    const char *compileOptions[] = {"-w", arch_opt.c_str()};
    NVRTC_SAFE_CALL("nvrtcCompileProgram", nvrtcCompileProgram(kernel_program, nOptions, compileOptions));

    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL("nvrtcGetPTXsize", nvrtcGetPTXSize(kernel_program, &ptxSize));
    char *ptx = new char[ptxSize];
    NVRTC_SAFE_CALL("nvrtcGetPTX", nvrtcGetPTX(kernel_program, ptx));

    // Get lowered name
    const char *lowered_kernel_name;
    NVRTC_SAFE_CALL("nvrtcGetLoweredName", nvrtcGetLoweredName(kernel_program, kernel_name.c_str(), &lowered_kernel_name));

    // Get pointer to kernel from PTX
    CUmodule module;
    CUDA_SAFE_CALL("cuModuleLoadDataEx", cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    delete[] ptx;
    CUDA_SAFE_CALL("cuModuleGetFunction", cuModuleGetFunction(&kern_func, module, lowered_kernel_name));

    // Set shared memory configuration
    CUDA_SAFE_CALL("cuFuncSetSharedMemConfig", cuFuncSetSharedMemConfig(kern_func, CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));

    // Destroy program
    NVRTC_SAFE_CALL("nvrtcDestroyProgram", nvrtcDestroyProgram(&kernel_program));
}


//===========================================================================
int libcusmm_transpose_d(int *trs_stack, int offset, int nblks,
                         double *buffer, int m, int n, CUstream stream) {

    CUfunction kern_func;

    // Look up the kernel in the table of already JITed kernels
    Triplet h_mnk = { m, n, 0 };
    auto kernel_it = transpose_handles.find(h_mnk); 
    if(kernel_it != transpose_handles.end()){  // the kernel has already been JITed

        kern_func = kernel_it->second; // retrieve handle

    } else {    // the kernel has not been JIT-ed yet

        // JIT and store a kernel for this transposition
        jit_transpose_handle(kern_func, m, n);
        transpose_handles.emplace(h_mnk, kern_func);

    }
    
    // Construct argument pointer list and lauch function
    int* trs_stack_ = trs_stack + offset; 
    void *args[] = { &trs_stack_, &nblks, &buffer};
    int res = launch_kernel_from_handle(kern_func, nblks, 128, stream, args); 

    return(cudaGetLastError());

}


//===========================================================================
extern "C" int libsmm_acc_transpose (void *trs_stack, int offset, int nblks, void *buffer,int datatype, int m, int n, void* stream) {
    cudaStream_t* custream = (cudaStream_t*) stream;
    if(datatype != dbcsr_type_real_8)
        return 0; //transpose not needed
    if(m>MAX_BLOCK_DIM || n>MAX_BLOCK_DIM)
      return 0; // maximum size over any dimention
    return libcusmm_transpose_d((int *) trs_stack, offset, nblks, (double *) buffer, m, n, *((CUstream *) stream));
};

//EOF
