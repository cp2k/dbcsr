/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2018  CP2K developers group                         *
 *****************************************************************************/
#ifndef LIBCUSMM_H
#define LIBCUSMM_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <unordered_map>

#define NVRTC_SAFE_CALL(name, x)                                  \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      printf("\nerror: %s failed with error %s",                  \
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
      printf("\nerror: %s failed with error %s",                  \
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


static std::unordered_map<int, CUfunction> kernel_handles;
static std::unordered_map<int, std::pair<int, int> > kernel_launching_parameters;

int libcusmm_process_d(int *param_stack, int stack_size,
    CUstream stream, int m, int n, int k,
    double * a_data, double * b_data, double * c_data);

static std::unordered_map<int, CUfunction> transpose_handles;

int libcusmm_transpose_d(int *trs_stack, int offset, int nblks, double *buffer,
                         int m, int n, cudaStream_t * stream);

#endif
//EOF
