/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2018  CP2K developers group                         *
 *****************************************************************************/
#ifndef LIBCUSMM_H
#define LIBCUSMM_H

#include "parameters_utils.h"

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <unordered_map>
#include <vector>

#define NVRTC_SAFE_CALL(name, x)                                  \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      printf("\nerror: %s failed with error %s\n",                \
             name, nvrtcGetErrorString(result));                  \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(name, x)                                   \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      printf("\nerror: %s failed with error %s\n",                \
             name, msg);                                          \
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
