/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifndef LIBCUSMM_H
#define LIBCUSMM_H

#include "parameters_utils.h"

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <unordered_map>
#include <vector>

// Macros for CUDA error handling
// Wrap calls to CUDA NVRTC API
#define NVRTC_SAFE_CALL(name, x)                                  \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      printf("\nNVRTC ERROR: %s failed with error %s\n",          \
             name, nvrtcGetErrorString(result));                  \
      exit(1);                                                    \
    }                                                             \
  } while(0)

enum libcusmm_algo {
    largeDB1 = 1,
    largeDB2 = 2,
    medium = 3,
    small = 4,
    tiny = 5
};

struct kernel_launcher {
    CUfunction kernel_function;
    int threads;
    int grouping;
    kernel_launcher(CUfunction const& kf, int th, int gp): kernel_function(kf), threads(th), grouping (gp) {}
};

static std::unordered_map<Triplet, kernel_launcher> kernel_handles;

int libcusmm_process_d(int *param_stack, int stack_size,
    CUstream stream, int m, int n, int k,
    double * a_data, double * b_data, double * c_data);

static std::unordered_map<Triplet, CUfunction> transpose_handles;

int libcusmm_transpose_d(int *trs_stack, int offset, int nblks, double *buffer,
                         int m, int n, CUstream stream);

#endif
