/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2018  CP2K developers group                         *
 *****************************************************************************/
#include "../include/libsmm_acc.h"
#include "libcusmm.h"
#include "parameters.h"

#include <sstream>
#include <fstream>
#include <string>

#define dbcsr_type_real_4     1
#define dbcsr_type_real_8     3
#define dbcsr_type_complex_4  5
#define dbcsr_type_complex_8  7
#define KERNEL_FILES_PATH DBCSRHOME "/src/libsmm_acc/libcusmm/kernels/"
#define P 999
#define Q 999

int hash(int m, int n, int k){
    return (m*P + n)*Q + k;
}

void get_kernel_handle(CUfunction& kern_func, libcusmm_algo algo, int M /*tile_m*/, int N /*tile_n*/, int w /*w*/, int v/*v*/, int threads, int grouping, int minblocks, int m_max /*m*/, int n_max /*n*/, int k_max /*k*/){
    
    // Check whether this kernel has already been JITed and a handle exists
    auto kernel_it = kernel_handles.find(hash(m_max, n_max, k_max));
    if (kernel_it != kernel_handles.end()){
#ifdef LOGGING
        printf("Found a handle to (%i, %i, %i) kernel in table at hash %i...\n", m_max, n_max, k_max, hash(m_max, n_max, k_max)); 
#endif
        kern_func = kernel_it->second;
#ifdef LOGGING
        printf("Got handle\n"); 
#endif
    } else {
#ifdef LOGGING
        printf("No handle to (%i, %i, %i) kernel found in table at hash %i...\n", m_max, n_max, k_max, hash(m_max, n_max, k_max)); 
#endif
        // Get the file path corresponding to the kernel to launch
        const std::string kernel_files_path = KERNEL_FILES_PATH;
        std::string kernel_file_name;
        std::string kernel_name;
        switch(algo) {
            case 1:
                kernel_file_name = kernel_files_path + "cusmm_dnt_largeDB1.h";
                kernel_name = "cusmm_dnt_largeDB1<" +
                              std::to_string(m_max) + ", " + std::to_string(n_max) + ", " + std::to_string(k_max) + ", " +
                              std::to_string(M) + ", " + std::to_string(N) + ", " +
                              std::to_string(w) + ", " + std::to_string(v) + ", " +
                              std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
                break;
            case 2:
                kernel_file_name = kernel_files_path + "cusmm_dnt_largeDB2.h";
                kernel_name = "cusmm_dnt_largeDB2<" +
                              std::to_string(m_max) + ", " + std::to_string(n_max) + ", " + std::to_string(k_max) + ", " +
                              std::to_string(M) + ", " + std::to_string(N) + ", " +
                              std::to_string(w) + ", " + std::to_string(v) + ", " +
                              std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
                break;
            case 3:
                kernel_file_name = kernel_files_path + "cusmm_dnt_medium.h";
                kernel_name = "cusmm_dnt_medium<" +
                              std::to_string(m_max) + ", " + std::to_string(n_max) + ", " + std::to_string(k_max) + ", " +
                              std::to_string(M) + ", " + std::to_string(N) + ", " +
                              std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
                break;
            case 4:
                kernel_file_name = kernel_files_path + "cusmm_dnt_small.h";
                kernel_name = "cusmm_dnt_small<" +
                              std::to_string(m_max) + ", " + std::to_string(n_max) + ", " + std::to_string(k_max) + ", " +
                              std::to_string(M) + ", " + std::to_string(N) + ", " +
                              std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
                break;
            case 5:
                kernel_file_name = kernel_files_path + "cusmm_dnt_tiny.h";
                kernel_name = "cusmm_dnt_tiny<" +
                              std::to_string(m_max) + ", " + std::to_string(n_max) + ", " + std::to_string(k_max) + ", " +
                              std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
                break;
            default:
                printf("\nerror: algorithm number %i is not encoded.");
                exit(1);
        }

        // Read file containing kernel
        std::ifstream kernel_file(kernel_file_name.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

        if (!kernel_file.is_open()){
            printf("\nerror: cannot open %s for reading\n", kernel_file_name.c_str());
            exit(1);
        }

        std::streampos pos = kernel_file.tellg();
        size_t kernel_code_size = (size_t)pos;
        char * kernel_code = new char [kernel_code_size + 1];

        kernel_file.seekg (0, std::ios::beg);
        kernel_file.read (kernel_code, kernel_code_size);
        kernel_file.close();
        kernel_code[kernel_code_size] = '\x0';
        ///printf("Read program of length %i characters\n", kernel_code_size);
        ///printf(kernel_code);

        // Create nvrtcProgram
        nvrtcProgram kernel_program;
        NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&kernel_program, kernel_code, "smm_kernel.cu", 0, NULL, NULL));

        // Add lowered name
        NVRTC_SAFE_CALL("nvrtcAddNameExpression", nvrtcAddNameExpression(kernel_program, kernel_name.c_str()));

        // (JIT-)compile kernel program
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
        ///printf("Get PTX of size %i characters\n", ptxSize);
        ///printf(ptx);

        // Get lowered name
        const char *lowered_kernel_name;
        NVRTC_SAFE_CALL("nvrtcGetLoweredName", nvrtcGetLoweredName(kernel_program, kernel_name.c_str(), &lowered_kernel_name));

        // Get pointer to kernel from PTX
        CUmodule module;
        CUDA_SAFE_CALL("cuModuleLoadDataEx", cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
        CUDA_SAFE_CALL("cuModuleGetFunction", cuModuleGetFunction(&kern_func, module, lowered_kernel_name));

        // Set shared memory configuration
        CUDA_SAFE_CALL("cuFuncSetSharedMemConfig", cuFuncSetSharedMemConfig(kern_func, CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));

        // Validate the JIt-ted kernel
        // TODO

        // store the handle
        kernel_handles.emplace(hash(m_max, n_max, k_max), kern_func); // that's not hw std::copy is used, find smth
#ifdef LOGGING
        printf("Store handle to kernel (%i, %i, %i) in table at hash %i\n", m_max, n_max, k_max, hash(m_max, n_max, k_max)); 
#endif
    }

}

int launch_cusmm_kernel(libcusmm_algo algo, int M /*tile_m*/, int N /*tile_n*/, int w /*w*/, int v/*v*/, int threads, int grouping, int minblocks, int *param_stack, int stack_size, CUstream stream, int m_max /*m*/, int n_max /*n*/, int k_max /*k*/, double *a_data, double *b_data, double *c_data){

#ifdef LOGGING
    printf("CUSMM with parameters:\nalgo = %i,\ntile_m = %i, tile_n = %i,\nw = %i, v = %i,\nthreads = %i, grouping = %i,\nminblocks = %i,stack_size = %i,\nm = %i, n = %i, k = %i\n\n", 
           algo, M, N, w, v, threads, grouping, minblocks, stack_size, m_max, n_max, k_max);
#endif 

    // Get handle to JIT-ted kernel
    CUfunction kern_func;
    get_kernel_handle(kern_func, algo, M, N, w, v, threads, grouping, minblocks, m_max, n_max, k_max); 
#ifdef LOGGING
    printf("get_kernel_handle returned\n"); 
#endif

    // Arguments
    int shared_size = 0;
    void *args[] = { &param_stack, &stack_size, &a_data, &b_data, &c_data };
    
    // Launch JITed kernel
#ifdef LOGGING
    printf("About to launch JIT-ted kernel on stream %i\n", stream);
#endif
    CUDA_SAFE_CALL(
            "cuLaunchKernel", 
            cuLaunchKernel(kern_func,						// CUfunction
                           ((stack_size + grouping - 1) / grouping), 1, 1,	// grid dim x, y, z
                           threads, 1, 1,       				// block dim x, y, z
                           shared_size, stream,		  			// shared mem and stream
                           args, NULL));      					// arguments
    CUDA_SAFE_CALL("cuCtxSynchronize", cuCtxSynchronize());
#ifdef LOGGING
    printf("Kernel launched, context synchronized\n"); 
#endif
    return(0);

}


int libcusmm_process_d(int *param_stack, int stack_size, CUstream stream, int m, int n, int k, double *a_data, double *b_data, double *c_data){

    // Retrieve launching parameters from parameters.h
    int* params = ht[m][n][k];

    // call launch_kernel with parameters
    return launch_cusmm_kernel(libcusmm_algo(params[0]), // enum {largeDB1, largeDB2, medium, small, tiny}
                               params[1], // tile_m
                               params[2], // tile_n
                               params[3], // w
                               params[4], // v
                               params[5], // threads
                               params[6], // grouping
                               params[7], // minblocks
                               param_stack,
                               stack_size,
                               stream,
                               m, // m_max
                               n, // n_max
                               k, // k_max
                               a_data,
                               b_data,
                               c_data);
}


extern "C" int libsmm_acc_process (void *param_stack, int stack_size, int nparams, int datatype, void *a_data, void *b_data, void *c_data, int m_max, int n_max, int k_max, int def_mnk, void *stream){
    if(def_mnk!=1)
        return(-1); // inhomogenous stacks not supported
    if(datatype==dbcsr_type_real_8)
        return(libcusmm_process_d ((int *) param_stack, stack_size, *((CUstream *) stream), m_max, n_max, k_max,(double *) a_data, (double *) b_data, (double *) c_data));

    return(-1); // datatype not supported
};

const char *transpose_d = "                                             \n\
template <int m, int n>                                                 \n\
__global__ void transpose_d(int *trs_stack, int nblks, double* mat){    \n\
 __shared__ double buf[m*n];						\n\
 int offset = trs_stack[blockIdx.x];                                    \n\
 for(int i=threadIdx.x; i < m*n; i+=blockDim.x){                        \n\
     buf[i] = mat[offset + i];                                          \n\
 }                                                                      \n\
 syncthreads();                                                         \n\
                                                                        \n\
 for(int i=threadIdx.x; i < m*n; i+=blockDim.x){                        \n\
     int r_out = i % n;                                                 \n\
     int c_out = i / n;                                                 \n\
     int idx = r_out * m + c_out;                                       \n\
     mat[offset + i] = buf[idx];                                        \n\
 }                                                                      \n\
}                                                                       \n";

int libcusmm_transpose_d(int *trs_stack, int offset, int nblks,
                         double *buffer, int m, int n, CUstream stream) {

#ifdef LOGGING
   printf("CUSMM-transpose with parameters:\nm = %i, n = %i,\nnblks = %i, offset = %i\n\n",
           m, n, nblks, offset);
#endif 

    // Check whether this kernel has already been JITed and a handle exists
    // if so, retrieve handle
    // if not, JIT kernel and store a handle to it

    // Create nvrtcProgram
    nvrtcProgram kernel_program;
    NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&kernel_program, transpose_d, "transpose_kernel.cu", 0, NULL, NULL));
    ///printf("NVRTC program created\n");

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
    ///printf("Get PTX of size %i characters\n", ptxSize);

    // Get lowered name
    ///printf("Getting lowered name\n"); 
    const char *lowered_kernel_name;
    NVRTC_SAFE_CALL("nvrtcGetLoweredName", nvrtcGetLoweredName(kernel_program, kernel_name.c_str(), &lowered_kernel_name));

    // Get pointer to kernel from PTX
    CUmodule module;
    CUDA_SAFE_CALL("cuModuleLoadDataEx", cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    CUfunction kern_func; // actual handle to executable kernel
    CUDA_SAFE_CALL("cuModuleGetFunction", cuModuleGetFunction(&kern_func, module, lowered_kernel_name));

    // Set shared memory configuration
    CUDA_SAFE_CALL("cuFuncSetSharedMemConfig", cuFuncSetSharedMemConfig(kern_func, CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));

    // Arguments
    int* trs_stack_ = trs_stack + offset; 
    void *args[] = { &trs_stack_, &nblks, &buffer};

    // Launch JITed kernel
    ///printf("About to launch JIT-ted kernel on stream %i\n", stream);
    CUDA_SAFE_CALL(
            "cuLaunchKernel",
            cuLaunchKernel(kern_func,                                           // CUfunction
                           nblks, 1, 1,      // grid dim x, y, z
                           128, 1, 1,                                       // block dim x, y, z
                           0, stream,                                 // shared mem and stream
                           args, NULL));                                        // arguments
    CUDA_SAFE_CALL("cuCtxSynchronize", cuCtxSynchronize());
    ///printf("Context synchronized\n");

    return(cudaGetLastError());

}

extern "C" int libsmm_acc_transpose (void *trs_stack, int offset, int nblks, void *buffer,int datatype, int m, int n, void* stream) {
    cudaStream_t* custream = (cudaStream_t*) stream;
    if(datatype != dbcsr_type_real_8)
        return 0; //transpose not needed
    return libcusmm_transpose_d((int *) trs_stack, offset, nblks, (double *) buffer, m, n, *((CUstream *) stream));
};

//EOF
