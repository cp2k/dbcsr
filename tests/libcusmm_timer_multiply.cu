/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <algorithm>
#include "acc/libsmm_acc/libcusmm/libcusmm_benchmark.h"
#include "acc/libsmm_acc/libcusmm/libcusmm.h"
#include "acc/libsmm_acc/libcusmm/parameters.h"

std::vector<Triplet> combinations(std::vector<int> to_combine){
    
    std::vector<Triplet> v; 
    int len = to_combine.size();
    for(size_t i=0; i<len; i++){
        for(size_t j=0; j<len; j++){
            for(size_t k=0; k<len; k++){
                v.push_back({to_combine[i], to_combine[j], to_combine[k]}); 
            }
        }
    }
    return v;
}

/****************************************************************************\
 \brief Checks correctness of and measures performance of common libcusmm multiplication kernels
\****************************************************************************/

int main(int argc, char** argv){

    KernelLauncher launcher = libcusmm_process_d;
    char buffer[1000];
    char * kernel_descr[1] = {buffer};

    // Choose which triplets to include in timer
    std::vector<Triplet> triplets = {
        {23, 23, 23}, 
        {6, 6, 6}, 
        {64, 64, 64}, 
        {78, 78, 78}, 
        {12, 12, 12}
    };
    std::vector<Triplet> to_add = combinations(std::vector<int>({14, 16, 29, 32}));    
    triplets.insert(triplets.end(), to_add.begin(), to_add.end());
    to_add = combinations(std::vector<int>({5, 32, 13, 24, 26}));
    triplets.insert(triplets.end(), to_add.begin(), to_add.end());
    to_add = combinations(std::vector<int>({9, 32, 22}));
    triplets.insert(triplets.end(), to_add.begin(), to_add.end());
    to_add = combinations(std::vector<int>({13, 14, 25, 26, 32}));
    triplets.insert(triplets.end(), to_add.begin(), to_add.end());

    // Remove triplets for which no autotuned parameters exist
    std::vector<Triplet> blocksizes;
    get_libcusmm_triplets(blocksizes, ht);
    for(int i=0; i<triplets.size(); i++){
        auto it = std::find(std::begin(blocksizes), std::end(blocksizes), triplets[i]);
        if(it == std::end(blocksizes))
            triplets[i] = {0, 0, 0};
    }
    triplets.erase(std::remove_if(triplets.begin(), 
                                  triplets.end(),
                                  [](Triplet const& t) {return t == Triplet({0, 0, 0});}), 
                   triplets.end());

    int n_triplets = triplets.size();
    printf("# Time %d blocksizes ...\n", n_triplets);

    int errors = 0;
    libcusmm_benchmark_t* handle;

    for(int i=0; i<n_triplets; i++){
        printf("\n\n");
        int m = triplets[i][0];
        int n = triplets[i][1];
        int k = triplets[i][2];
        sprintf(buffer, "%d x %d x %d", m, n, k);
        libcusmm_benchmark_init(&handle, timing, m, n, k);
        errors += libcusmm_benchmark(handle, m, n, k, 1, &launcher, kernel_descr);
        libcusmm_benchmark_finalize(handle);
    }

    printf("# Done, found %d errors.\n", errors);
    return(errors);
}
