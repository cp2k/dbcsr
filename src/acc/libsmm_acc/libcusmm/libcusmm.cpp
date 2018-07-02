/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2018  CP2K developers group                         *
 *****************************************************************************/
#include "../include/libsmm_acc.h"
#include "libcusmm.h"
#include "libcusmm_benchmark.h"
#include "parameters.h"
#include "cusmm_kernels.h"

#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <algorithm>

#define dbcsr_type_real_4     1
#define dbcsr_type_real_8     3
#define dbcsr_type_complex_4  5
#define dbcsr_type_complex_8  7
#define KERNEL_FILES_PATH DBCSRHOME "/src/acc/libsmm_acc/libcusmm/kernels/"

#ifdef TIMING
#include <chrono>
typedef std::chrono::high_resolution_clock timer_clock;
#endif

// Hash function constants
inline int hash(int m, int n, int k){
    return (m*P + n)*Q + k;
}
std::vector<int> hash_back(int hash){

    int PQ = P*Q; 
    int m = hash / PQ; 
    hash -= PQ*m;
    int n = hash / Q; 
    hash -= Q*n; 
    int k = hash; 
    return std::vector<int>({m, n, k}); 

}


//===========================================================================
inline int launch_kernel_from_handle(CUfunction const& kern_func, int nblks, int threads, CUstream stream, void** args){

#ifdef LOGGING
    printf("launch_kernel_from_handle: about to launch JIT-ted kernel\n");
#endif
    CUDA_SAFE_CALL(
            "cuLaunchKernel",
            cuLaunchKernel(kern_func,           // CUfunction
                           nblks, 1, 1,      	// grid dimension x, y, z
                           threads, 1, 1,	// block dimension x, y, z
                           0, stream,           // shared memory size and stream
                           args, NULL));        // arguments
    CUDA_SAFE_CALL("cuCtxSynchronize", cuCtxSynchronize());
#ifdef LOGGING
    printf("launch_kernel_from_handle: kernel launched, context synchronized\n");
#endif
    return(0);

}


//===========================================================================
inline void validate_kernel(CUfunction& kern_func, CUstream stream, int threads, int grouping, int m, int n, int k){

#ifdef LOGGING
    printf("validate_kernel: start kernel validation check (libcusmm_benchmark) with (%ix%ix%i)\n", m, n, k);
#endif
    libcusmm_benchmark_t* h;
    libcusmm_benchmark_init(&h, false, m, n, k);

    // Compute the matrix-matrix multiplication on the CPU
    memset(h->mat_c, 0, h->n_c * m * n * sizeof(double));
    matInit(h->mat_a, h->n_a, m, k, 42);
    matInit(h->mat_b, h->n_b, k, n, 24);
    stackInit(h->stack, h->n_stack, h->n_c, h->mat_c, h->n_a, h->mat_a, h->n_b, h->mat_b, m, n, k);

#ifdef LOGGING
    printf("validate_kernel: launch validation kernel (CPU - %ix%ix%i)\n", m, n, k);
#endif
    stackCalc(h->stack, h->n_stack, h->mat_c, h->mat_a, h->mat_b, m, n, k);
    double sumCPU = checkSum(h->mat_c, h->n_c, m, n);

    // Run the matrix-matrix multiplication kernel on the GPU
    cudaMemcpy(h->d_mat_a, h->mat_a, h->n_a * m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_mat_b, h->mat_b, h->n_b * k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_stack, h->stack, h->n_stack * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(h->d_mat_c, 0, h->n_c * m * n * sizeof(double));

#ifdef LOGGING
    printf("validate_kernel: launch validation kernel (GPU - %ix%ix%i)\n", m, n, k);
#endif
    void *args[] = { &h->d_stack, &h->n_stack, &h->d_mat_a, &h->d_mat_b, &h->d_mat_c };
    int res = launch_kernel_from_handle(kern_func, ((h->n_stack + grouping - 1) / grouping), threads, stream, args);
    cudaMemcpy(h->mat_c, h->d_mat_c, h->n_c * m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess){
        printf("validate_kernel: cuda_error: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }

    // Compare results
    double sumGPU =  checkSum(h->mat_c, h->n_c, m, n);
    if(sumGPU != sumCPU){
        printf("validate_kernel: checksum_diff: %g\n", sumGPU-sumCPU);
        exit(1);
    }
    libcusmm_benchmark_finalize(h);
#ifdef LOGGING
    printf("validate_kernel: OK\n");
#endif
}


//===========================================================================
inline void jit_kernel(CUfunction& kern_func, libcusmm_algo algo, int tile_m, int tile_n, int w, int v, int threads, int grouping, int minblocks, int m, int n, int k){
#ifdef LOGGING
    printf("jit_kernel: start\n");
#endif

        // Get the code and the lowered name of the kernel to launch
        std::string kernel_code, kernel_name;
#ifdef TIMING
    auto switch_name_overhead_start = timer_clock::now();
#endif
        switch(algo) {
            case 1:
                kernel_code = cusmm_dnt_largeDB1; 
                kernel_name = "cusmm_dnt_largeDB1<" +
                              std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                              std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                              std::to_string(w) + ", " + std::to_string(v) + ", " +
                              std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
                break;
            case 2:
                kernel_code = cusmm_dnt_largeDB2; 
                kernel_name = "cusmm_dnt_largeDB2<" +
                              std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                              std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                              std::to_string(w) + ", " + std::to_string(v) + ", " +
                              std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
                break;
            case 3:
                kernel_code = cusmm_dnt_medium; 
                kernel_name = "cusmm_dnt_medium<" +
                              std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                              std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                              std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
                break;
            case 4:
                kernel_code = cusmm_dnt_small; 
                kernel_name = "cusmm_dnt_small<" +
                              std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                              std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                              std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
                break;
            case 5:
                kernel_code = cusmm_dnt_tiny; 
                kernel_name = "cusmm_dnt_tiny<" +
                              std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                              std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
                break;
            default:
                printf("\nerror: algorithm number %i is not encoded.", algo);
                exit(1);
        }
#ifdef TIMING
    auto switch_name_overhead_stop = timer_clock::now();
    printf("  ##TIMER## jit: switch name overhead: %g us\n",
           std::chrono::duration<double, std::micro>(switch_name_overhead_stop - switch_name_overhead_start).count());
#endif

        // Create nvrtcProgram
        nvrtcProgram kernel_program;
        NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&kernel_program, kernel_code.c_str(), "smm_kernel.cu", 0, NULL, NULL));

        // Add lowered name
        NVRTC_SAFE_CALL("nvrtcAddNameExpression", nvrtcAddNameExpression(kernel_program, kernel_name.c_str()));

        // (JIT-)compile kernel program
        const std::string kernel_files_path = KERNEL_FILES_PATH;
        const std::string include_opt = "-I=" + kernel_files_path;
        const char *compileOptions[] = {include_opt.c_str()};
        size_t nOptions = 1;
        nvrtcResult compileResult = nvrtcCompileProgram(kernel_program, nOptions, compileOptions);

#ifdef LOGGING
        // Obtain compilation log
        size_t logSize;
        NVRTC_SAFE_CALL("nvrtcGetProgramLogSize", nvrtcGetProgramLogSize(kernel_program, &logSize));
        char *log = new char[logSize];
        NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(kernel_program, log));
        printf("\ncompilation log ---\n");
        printf(log);
        printf("\n--- end log\n\n");
        delete[] log;
        if (compileResult != NVRTC_SUCCESS) {
            printf("NVRTC compilation failed\n");
            exit(1);
        }
#endif

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
#ifdef LOGGING
    printf("jit_kernel: kernel jitted\n");
#endif
        // Set shared memory configuration
        CUDA_SAFE_CALL("cuFuncSetSharedMemConfig", cuFuncSetSharedMemConfig(kern_func, CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));

        // Destroy program
        NVRTC_SAFE_CALL("nvrtcDestroyProgram", nvrtcDestroyProgram(&kernel_program));
}


//===========================================================================
int libcusmm_process_d(int *param_stack, int stack_size, CUstream stream, int m, int n, int k, double *a_data, double *b_data, double *c_data){

#ifdef LOGGING
    printf("-------------------------------------------------------------------------------------------\n");
    printf("libcusmm_process_d: CUSMM (%ix%ix%i)\n", m, n, k);
#endif

#ifdef TIMING
    auto JITed_kernel_lookup_start = timer_clock::now();
#endif
    CUfunction kern_func;
    int threads, grouping; 
    int h_mnk = hash(m, n, k);
    auto kernel_it = kernel_handles.find(h_mnk); 
#ifdef TIMING
    auto JITed_kernel_lookup_stop = timer_clock::now();
    printf("##TIMER## JITed kernel lookup: %g us\n",
           std::chrono::duration<double, std::micro>(JITed_kernel_lookup_stop - JITed_kernel_lookup_start).count());
#endif
    if (kernel_it != kernel_handles.end()){

        kern_func = kernel_it->second; 
#ifdef LOGGING
        printf("libcusmm_process_d: found a handle to (%i, %i, %i) kernel in table at hash %i...\n", m, n, k, h_mnk);
#endif

        // Retrieve launching parameters
#ifdef TIMING
        auto launchpar_lookup_start = timer_clock::now();
#endif
        auto launchpar = kernel_launching_parameters[h_mnk];
        threads = launchpar.first; 
        grouping = launchpar.second;
#ifdef TIMING
    auto launchpar_lookup_stop = timer_clock::now();
    printf("##TIMER## launching parameters lookup: %g us\n", 
           std::chrono::duration<double, std::micro>(launchpar_lookup_stop - launchpar_lookup_start).count());
#endif
#ifdef LOGGING
        printf("libcusmm_process_d: launching parameters:\nthreads = %i, grouping = %i\n", threads, grouping);
#endif

    } else {	// this kernel has not been JIT-ed yet

#ifdef LOGGING
        printf("libcusmm_process_d: no handle found to (%i, %i, %i) kernel...\n", m, n, k); 
#endif

	// Check whether autotuned parameters are given for this kernel, and if so, retrieve them 
#ifdef TIMING
    auto autotuned_pars_lookup_start = timer_clock::now();
#endif
        auto params_it = ht.find(h_mnk);
#ifdef TIMING
    auto autotuned_pars_lookup_stop = timer_clock::now();
    printf("##TIMER## Autotuned parameters lookup: %g us\n",
           std::chrono::duration<double, std::micro>(autotuned_pars_lookup_stop - autotuned_pars_lookup_start).count());
#endif
	if (params_it != ht.end()){
            // Retrieve launching parameters
#ifdef TIMING
    auto autotuned_pars_retrieval_start = timer_clock::now();
#endif
            std::vector<int> params = ht[h_mnk];
            libcusmm_algo algo = libcusmm_algo(params[0]); // enum {largeDB1, largeDB2, medium, small, tiny}
            int tile_m = params[1];
            int tile_n = params[2];
            int w = params[3]; 
            int v = params[4]; 
            threads = params[5];
            grouping = params[6];
            int minblocks =  params[7];
#ifdef TIMING
    auto autotuned_pars_retrieval_stop = timer_clock::now();
    printf("##TIMER## Autotuned parameters retrieval: %g us\n",
           std::chrono::duration<double, std::micro>(autotuned_pars_retrieval_stop - autotuned_pars_retrieval_start).count());
#endif
#ifdef LOGGING
            printf("libcusmm_process_d: CUSMM with parameters:\nalgo = %i,\ntile_m = %i, tile_n = %i,\nw = %i, v = %i,\nthreads = %i, grouping = %i,\nminblocks = %i, stack_size = %i,\nm = %i, n = %i, k = %i\n\n",
                   algo, tile_m, tile_n, w, v, threads, grouping, minblocks, stack_size, m, n, k);
#endif

#ifdef TIMING
    auto JIT_overhead_start = timer_clock::now();
#endif
            // JIT and validate the kernel
            jit_kernel(kern_func, algo, tile_m, tile_n, w, v, threads, grouping, minblocks, m, n, k);
#ifdef TIMING
    auto JIT_overhead_stop = timer_clock::now();
    printf("##TIMER## JIT overhead: %g us\n",
           std::chrono::duration<double, std::micro>(JIT_overhead_stop - JIT_overhead_start).count());
#endif
#ifdef TIMING
    auto val_overhead_start = timer_clock::now();
#endif
            validate_kernel(kern_func, stream, threads, grouping, m, n, k);
#ifdef TIMING
    auto val_overhead_stop = timer_clock::now();
    printf("##TIMER## validation overhead: %g us\n",
           std::chrono::duration<double, std::micro>(val_overhead_stop - val_overhead_start).count());
#endif

            // Store the handle to the JIT-ed kernel and to its launching parameters
#ifdef TIMING
    auto handle_store_overhead_start = timer_clock::now();
#endif
            kernel_handles.emplace(h_mnk, kern_func);
            kernel_launching_parameters.emplace(h_mnk, std::make_pair(threads, grouping));
#ifdef TIMING
    auto handle_store_overhead_stop = timer_clock::now();
    printf("##TIMER## handle store overhead: %g us\n",
           std::chrono::duration<double, std::micro>(handle_store_overhead_stop - handle_store_overhead_start).count());
#endif
#ifdef LOGGING
            printf("libcusmm_process_d: store handle to kernel (%i, %i, %i) in table at hash %i\n", m, n, k, h_mnk);
#endif

        } else { // there exist no autotuned parameters for this (m, n, k)-triplet

            // Fall back to CPU
#ifdef LOGGING
            printf("libcusmm_process_d: fallback to CPU\n");
#endif
            return -2; 

        }

    }

    // Construct argument pointer list and launch kernel
#ifdef TIMING
    auto assemble_args_start = timer_clock::now();
#endif
    void *args[] = { &param_stack, &stack_size, &a_data, &b_data, &c_data };
#ifdef TIMING
    auto assemble_args_stop = timer_clock::now();
    printf("##TIMER## assemble args overhead: %g us\n",
           std::chrono::duration<double, std::micro>(assemble_args_stop - assemble_args_start).count());
#endif
#ifdef TIMING
    auto kernel_start = timer_clock::now();
#endif
    int ret = launch_kernel_from_handle(kern_func, ((stack_size + grouping - 1) / grouping), threads, stream, args);
#ifdef TIMING
    auto kernel_stop = timer_clock::now();
    printf("##TIMER## launch and run kernel: %g us\n",
           std::chrono::duration<double, std::micro>(kernel_stop - kernel_start).count());
#endif
    return ret; 

}


//===========================================================================
extern "C" int libsmm_acc_process (void *param_stack, int stack_size, int nparams, int datatype, void *a_data, void *b_data, void *c_data, int m, int n, int k, int def_mnk, void *stream){
#ifdef TIMING
    auto libcusmm_smm_start = timer_clock::now();
#endif
    if(def_mnk!=1)
        return(-1); // inhomogenous stacks not supported
    if(datatype==dbcsr_type_real_8){
        int res = (libcusmm_process_d ((int *) param_stack, stack_size, *((CUstream *) stream), m, n, k, (double *) a_data, (double *) b_data, (double *) c_data));
#ifdef TIMING
    auto libcusmm_smm_stop = timer_clock::now();
    printf("##TIMER_LIBCUSMM## libcusmm_smm: %g us\n",
           std::chrono::duration<double, std::micro>(libcusmm_smm_stop - libcusmm_smm_start).count());
#endif
	return res; 
    }
    return(-1); // datatype not supported
};


//===========================================================================
void jit_transpose_handle(CUfunction& kern_func, int m, int n){

    // Create nvrtcProgram
    nvrtcProgram kernel_program;
    NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&kernel_program, cusmm_transpose.c_str(), "transpose_kernel.cu", 0, NULL, NULL));

    // Add lowered name
    std::string kernel_name = "transpose_d<" + std::to_string(m) + ", " + std::to_string(n) + ">";
    NVRTC_SAFE_CALL("nvrtcAddNameExpression", nvrtcAddNameExpression(kernel_program, kernel_name.c_str()));

    // (JIT-)compile
    const char *compileOptions[] = {};
    nvrtcResult compileResult = nvrtcCompileProgram(kernel_program, 0, compileOptions);

#ifdef LOGGING
    // Obtain compilation log
    size_t logSize;
    NVRTC_SAFE_CALL("nvrtcGetProgramLogSize", nvrtcGetProgramLogSize(kernel_program, &logSize));
    char *log = new char[logSize];
    NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(kernel_program, log));
    printf("\ncompilation log ---\n");
    printf(log);
    printf("\n--- end log\n\n");
    delete[] log;
    if (compileResult != NVRTC_SUCCESS) {
        printf("NVRTC compilation failed\n");
        exit(1);
    }
#endif 

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

#ifdef LOGGING
   printf("-------------------------------------------------------------------------------------------\n");
   printf("libcusmm_transpose_d: CUSMM-transpose with parameters:\nm = %i, n = %i,\nnblks = %i, offset = %i\n\n", m, n, nblks, offset);
#endif

    int h_mnk = hash(m, n, 0);
    CUfunction kern_func;
    auto kernel_it = transpose_handles.find(h_mnk); 
    if(kernel_it != transpose_handles.end()){

        kern_func = kernel_it->second;
#ifdef LOGGING
        printf("libcusmm_transpose_d: found a handle to (%i, %i) transpose in table at hash %i...\n", m, n, hash(m, n, 0));
#endif

    } else {

#ifdef LOGGING
    printf("libcusmm_transpose_d: no handle to (%i, %i) transpose found in table at hash %i...\n", m, n, hash(m, n, 0));
#endif
        // JIT and store a kernel for this transposition
        jit_transpose_handle(kern_func, m, n);
        transpose_handles.emplace(hash(m, n, 0), kern_func);
#ifdef LOGGING
        printf("libcusmm_transpose_d: store handle to kernel (%i, %i) in table at hash %i\n", m, n, hash(m, n, 0));
#endif   

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
    return libcusmm_transpose_d((int *) trs_stack, offset, nblks, (double *) buffer, m, n, *((CUstream *) stream));
};

//EOF
