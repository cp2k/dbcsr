/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <utility>
#include "acc/libsmm_acc/libcusmm/libcusmm_benchmark.h"
#include "acc/libsmm_acc/libcusmm/libcusmm.h"
#include "acc/libsmm_acc/libcusmm/parameters.h"


/****************************************************************************\
 \brief Checks correctness of every libcusmm transpose kernel.
\****************************************************************************/

int main(int argc, char** argv){

    TransposeLauncher launcher_tr = libcusmm_transpose_d;

    char buffer[1000];
    char * kernel_descr[1] = {buffer};

    // Get all blocksizes available in libcusmm
    std::vector<Triplet> libcusmm_triplets;
    get_libcusmm_triplets(libcusmm_triplets, ht);
    int n_triplets = libcusmm_triplets.size();

    int max_m=0, max_n=0, max_k=0;
    for(int i=0; i<n_triplets; i++){
        max_m = max(max_n, libcusmm_triplets[i][0]);
        max_n = max(max_m, libcusmm_triplets[i][1]);
        max_k = max(max_k, libcusmm_triplets[i][2]);
    }

    libcusmm_benchmark_t* handle;
    libcusmm_benchmark_init(&handle, test, max_m, max_n, max_k);

    // Get (m,n) pairs to test transposition
    std::vector<std::pair<int,int> > libcusmm_transpose_pairs; 
    for(int i=0; i<n_triplets; i++){
        int m = libcusmm_triplets[i][0];
        int n = libcusmm_triplets[i][1];
        int k = libcusmm_triplets[i][2];
        libcusmm_transpose_pairs.push_back(std::make_pair(m, k));
        libcusmm_transpose_pairs.push_back(std::make_pair(k, n));
    }
    std::sort(libcusmm_transpose_pairs.begin(), libcusmm_transpose_pairs.end(), 
              [](std::pair<int,int> a, std::pair<int,int> b) {
        return (a.first > b.first) || (a.first == b.first && a.second > b.second) ;   
    });
    auto last = std::unique(libcusmm_transpose_pairs.begin(), libcusmm_transpose_pairs.end());
    libcusmm_transpose_pairs.erase(last, libcusmm_transpose_pairs.end()); 
    int n_pairs = libcusmm_transpose_pairs.size();
    printf("# Libcusmm has %d blocksizes for transposition\n", n_pairs);

    int errors = 0;
    for(int i=0; i<n_pairs; i++){
        int m = libcusmm_transpose_pairs[i].first;
        int n = libcusmm_transpose_pairs[i].second;
        sprintf(buffer, "%d x %d", m, n);
        errors += libcusmm_benchmark_transpose(handle, m, n, &launcher_tr, kernel_descr);
    }
    libcusmm_benchmark_finalize(handle);

    printf("# Done, found %d transpose errors.\n", errors);
    return errors;
}
