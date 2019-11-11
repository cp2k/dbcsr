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
#include <vector>
#include <array>
#include <utility>
#include "acc/libsmm_acc/libsmm_benchmark.h"
#include "acc/libsmm_acc/libsmm.h"
#include "acc/libsmm_acc/parameters.h"


/****************************************************************************\
 \brief Checks correctness of all libsmm transpose kernels
\****************************************************************************/

int main(int argc, char** argv){

    TransposeLauncher launcher_tr = libsmm_transpose_d;

    char buffer[1000];
    char * kernel_descr[1] = {buffer};

    // Get all blocksizes available in libsmm
    std::vector<Triplet> libsmm_triplets;
    get_libsmm_triplets(libsmm_triplets, ht);
    int n_triplets = libsmm_triplets.size();

    int max_m=0, max_n=0, max_k=0;
    for(int i=0; i<n_triplets; i++){
        max_m = std::max(max_n, libsmm_triplets[i][0]);
        max_n = std::max(max_m, libsmm_triplets[i][1]);
        max_k = std::max(max_k, libsmm_triplets[i][2]);
    }

    libsmm_benchmark_t* handle;
    libsmm_benchmark_init(&handle, test, max_m, max_n, max_k);

    // Get (m,n) pairs to test transposition
    std::vector<std::pair<int,int> > libsmm_transpose_pairs;
    for(int i=0; i<n_triplets; i++){
        int m = libsmm_triplets[i][0];
        int n = libsmm_triplets[i][1];
        int k = libsmm_triplets[i][2];
        libsmm_transpose_pairs.push_back(std::make_pair(m, k));
        libsmm_transpose_pairs.push_back(std::make_pair(k, n));
    }
    std::sort(libsmm_transpose_pairs.begin(), libsmm_transpose_pairs.end(),
              [](std::pair<int,int> a, std::pair<int,int> b) {
        return (a.first > b.first) || (a.first == b.first && a.second > b.second) ;
    });
    auto last = std::unique(libsmm_transpose_pairs.begin(), libsmm_transpose_pairs.end());
    libsmm_transpose_pairs.erase(last, libsmm_transpose_pairs.end());
    int n_pairs = libsmm_transpose_pairs.size();
    printf("# Libsmm has %d blocksizes for transposition\n", n_pairs);

    // Sort (m,n) pairs in growing order
    std::sort(libsmm_transpose_pairs.begin(), libsmm_transpose_pairs.end(),
              [](std::pair<int, int> mn1, std::pair<int, int> mn2){
                  if(mn1.first != mn2.first){
                      return mn1.first < mn2.first;
                  } else {
                      return mn1.second < mn2.second;
                  }
              });

    int errors = 0;
    for(int i=0; i<n_pairs; i++){
        int m = libsmm_transpose_pairs[i].first;
        int n = libsmm_transpose_pairs[i].second;
        sprintf(buffer, "%d x %d", m, n);
        errors += libsmm_benchmark_transpose(handle, m, n, &launcher_tr, kernel_descr);
    }
    libsmm_benchmark_finalize(handle);

    printf("# Done, found %d transpose errors.\n", errors);
    return errors;
}
