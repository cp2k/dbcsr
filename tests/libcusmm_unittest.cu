/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2018  CP2K developers group                         *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include "acc/libsmm_acc/libcusmm/libcusmm_benchmark.h"
#include "acc/libsmm_acc/libcusmm/libcusmm.h"

/****************************************************************************\
 \brief Checks correctness of every libcusmm kernel and measures its performance.
\****************************************************************************/

int main(int argc, char** argv){

    printf("In libcusmm unit test (toy)\n"); 
    KernelLauncher launcher = libcusmm_process_d;

    char buffer[1000];
    char * kernel_descr[1] = {buffer};

    std::vector<int> v;

    /*int const p_min = 4;
    int const p_max = 32;
    for(size_t m=p_min; m<=p_max; m++){
        for(size_t n=p_min; n<=p_max; n++){
            for(size_t k=p_min; k<=p_max; k++){
                v.push_back(m); v.push_back(n); v.push_back(k);
            }
        }
     }*/

    v.push_back(6); v.push_back(6); v.push_back(6);
    v.push_back(6); v.push_back(6); v.push_back(64);
    v.push_back(6); v.push_back(64); v.push_back(6);
    v.push_back(64); v.push_back(6); v.push_back(6);
    v.push_back(6); v.push_back(64); v.push_back(64);
    v.push_back(64); v.push_back(6); v.push_back(64);
    v.push_back(64); v.push_back(64); v.push_back(6);
    v.push_back(64); v.push_back(64); v.push_back(64);
    
    int n_blocksizes = v.size()/3;
    const int *blocksizes = &v[0];
    printf("# Libcusmm has %d blocksizes compiled in...\n", n_blocksizes);

    int max_m=0, max_n=0, max_k=0;
    for(int i=0; i<n_blocksizes; i++){
        max_m = std::max(max_m, blocksizes[3*i + 0]);
        max_n = std::max(max_n, blocksizes[3*i + 1]);
        max_k = std::max(max_k, blocksizes[3*i + 2]);
    }

    libcusmm_benchmark_t* handle;
    libcusmm_benchmark_init(&handle, false, max_m, max_n, max_k);
    printf("Initialized benchmarking\n");

    int errors = 0;
    for(int i=0; i<n_blocksizes; i++){
        int m = blocksizes[3*i + 0];
        int n = blocksizes[3*i + 1];
        int k = blocksizes[3*i + 2];
        sprintf(buffer, "%d x %d x %d", m, n, k);
        errors += libcusmm_benchmark(handle, m, n, k, 1, &launcher, kernel_descr);
    }
    libcusmm_benchmark_finalize(handle);

    printf("# Done, found %d errors.\n", errors);
    return(errors);
}

//EOF
