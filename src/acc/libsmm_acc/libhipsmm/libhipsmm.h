/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifndef LIBHIPSMM_H
#define LIBHIPSMM_H

#include "parameters_utils.h"

#include <cstdio>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <unordered_map>
#include <vector>

// Macros for HIP error handling
// Wrap calls to HIPRTC API
#define HIPRTC_SAFE_CALL(name, x)                                 \
  do {                                                            \
    hiprtcResult result = x;                                      \
    if (result != HIPRTC_SUCCESS) {                               \
      printf("\nHIPRTC ERROR: %s failed with error %s\n",         \
             name, hiprtcGetErrorString(result));                 \
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
    hipFunction_t kernel_function;
    int threads;
    int grouping;
    kernel_launcher(hipFunction_t const& kf, int th, int gp): kernel_function(kf), threads(th), grouping (gp) {}
};

static std::unordered_map<Triplet, kernel_launcher> kernel_handles;

int libcusmm_process_d(int *param_stack, int stack_size,
    hipStream_t stream, int m, int n, int k,
    double * a_data, double * b_data, double * c_data);

static std::unordered_map<Triplet, hipFunction_t> transpose_handles;

int libcusmm_transpose_d(int *trs_stack, int offset, int nblks, double *buffer,
                         int m, int n, hipStream_t stream);

#endif
